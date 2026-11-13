# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under the MIT License; see gdn_chunk64_LICENSE.txt.
# Source-scoped extraction from 1Cat-vLLM v1.5.2 (wheel build 9f970d628).
# Only the three SM70 generators are retained. The upstream gemm_v1 alias
# is spelled gemm directly for pinned TileLang0.1.10, without patching its module.
# No upstream import-time CUDA query, serving code, wrappers or compiler override.

import tilelang
import tilelang.language as T


@tilelang.jit(
    # out_idx=[-1],
)
def tilelang_chunk_local_cumsum(
    H,
    chunk_size,
    accum_dtype,
    g_dtype,
    seqlen_dtype,
    is_varlen,
    reverse,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    g_shape = (data_batch_size, num_tokens, H)

    @T.macro
    def kernel_body(
        bb,
        bc,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        g_raw,
        g_cumsum,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        g_fragment = T.alloc_fragment((H, block_S), dtype=accum_dtype)
        gT_fragment = T.alloc_fragment((block_S, H), dtype=g_dtype)
        gT_shared = T.alloc_shared((block_S, H + 1), dtype=g_dtype)

        if right <= seq_end_idx:
            T.copy(g_raw[bb, left:right, 0:H], gT_fragment)
        else:
            for j, i in T.Parallel(block_S, H):
                if left + j < seq_end_idx:
                    gT_fragment[j, i] = g_raw[bb, left + j, i]
                else:
                    gT_fragment[j, i] = 0
        T.copy(gT_fragment, gT_shared[:, :H])

        for i, j in T.Parallel(H, block_S):
            g_fragment[i, j] = gT_shared[j, i]

        T.cumsum(g_fragment, dim=1, reverse=reverse)

        for i, j in T.Parallel(H, block_S):
            gT_shared[j, i] = g_fragment[i, j]

        T.copy(gT_shared[:, :H], gT_fragment)
        if right <= seq_end_idx:
            T.copy(gT_fragment, g_cumsum[bb, left:right, 0:H])
        else:
            for j, i in T.Parallel(block_S, H):
                if left + j < seq_end_idx:
                    g_cumsum[bb, left + j, i] = gT_fragment[j, i]

    if is_varlen:

        @T.prim_func
        def tilelang_chunk_local_cumsum_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
        ):
            with T.Kernel(num_chunks, threads=128) as (bc,):
                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    bb,
                    bc,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    g_raw,
                    g_cumsum,
                )

    else:

        @T.prim_func
        def tilelang_chunk_local_cumsum_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks, threads=128) as (bc,):
                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb,
                    bc,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    g_raw,
                    g_cumsum,
                )

    return tilelang_chunk_local_cumsum_kernel

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_kkt_solve_sm70(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    b_dtype,
    seqlen_dtype,
    is_varlen,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    k_shape = (data_batch_size, num_tokens, Hg, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)

    @T.macro
    def kernel_body(
        bb,
        bc,
        bh,
        bhg,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        k,
        b,
        a,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
        a64_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

        a16i_row = T.alloc_fragment((4, 16), dtype=accum_dtype)
        a16i_sum = T.alloc_fragment((4, 16), dtype=accum_dtype)

        a16i_shared = T.alloc_shared((4, 17, 16), dtype=accum_dtype)
        a16o_shared = T.alloc_shared((2, 17, 16), dtype=accum_dtype)
        a16o_fragment = T.alloc_fragment((2, 16, 16), dtype=accum_dtype)

        a32i_fragment = T.alloc_fragment((2, 32, 32), dtype=accum_dtype)
        a32i0_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32i1_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)

        a64_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

        T.annotate_layout(
            {
                a16i_shared: tilelang.layout.make_linear_layout(a16i_shared),
                a16o_shared: tilelang.layout.make_linear_layout(a16o_shared),
            }
        )

        if right <= seq_end_idx:
            T.copy(k[bb, left:right, bhg, 0:DK], k_shared)
            for j_s in T.Parallel(block_S):
                b_shared[j_s] = b[bb, left + j_s, bh]
        else:
            for j_s, j_d in T.Parallel(block_S, DK):
                if left + j_s < seq_end_idx:
                    k_shared[j_s, j_d] = k[bb, left + j_s, bhg, j_d]
                else:
                    k_shared[j_s, j_d] = 0
            for j_s in T.Parallel(block_S):
                if left + j_s < seq_end_idx:
                    b_shared[j_s] = b[bb, left + j_s, bh]
                else:
                    b_shared[j_s] = 0
        T.sync_threads()

        T.gemm(k_shared, k_shared, a64_fragment, transpose_B=True, clear_accum=True)

        for j_s, j_t in T.Parallel(block_S, block_S):
            a64_fragment[j_s, j_t] *= b_shared[j_s]

        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s < j_t:
                a64_fragment[j_s, j_t] = 0
            elif j_s == j_t:
                a64_fragment[j_s, j_t] = 1

        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s >= 32 and j_t < 32:
                a32o_shared[j_s - 32, j_t] = -a64_fragment[j_s, j_t]
            elif (j_s // 16) == (j_t // 16) + 1:
                a16o_shared[j_s // 32, j_s % 16, j_t % 16] = -a64_fragment[j_s, j_t]
            elif (j_s // 16) == (j_t // 16):
                a16i_shared[j_s // 16, j_s % 16, j_t % 16] = a64_fragment[j_s, j_t]

        T.clear(a16i_row)
        for k_s in T.unroll(1, 16):
            for j_s, k_t in T.Parallel(4, 16):
                if k_t < k_s:
                    a16i_row[j_s, k_t] = a16i_shared[j_s, k_s, k_t]
            T.clear(a16i_sum)
            for k_r in T.unroll(k_s):
                for j_s, k_t in T.Parallel(4, 16):
                    a16i_sum[j_s, k_t] -= (
                        a16i_shared[j_s, k_r, k_t] * a16i_row[j_s, k_r]
                    )
            for j_s, k_t in T.Parallel(4, 16):
                if k_t < k_s:
                    a16i_shared[j_s, k_s, k_t] = a16i_sum[j_s, k_t]

        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                a16o_fragment[j_s, k_s, k_t] += (
                    a16i_shared[j_s * 2 + 1, k_s, k_r] * a16o_shared[j_s, k_r, k_t]
                )
        for j_s, k_s, k_t in T.Parallel(2, 16, 16):
            a16o_shared[j_s, k_t, k_s] = a16o_fragment[j_s, k_s, k_t]
        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                a16o_fragment[j_s, k_s, k_t] += (
                    a16o_shared[j_s, k_r, k_s] * a16i_shared[j_s * 2, k_r, k_t]
                )
        T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])

        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s < 16 and k_t >= 16:
                a32i_fragment[j_s, k_s, k_t] = 0
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s >= 16 and k_t < 16:
                a32i_fragment[j_s, k_s, k_t] = a16o_shared[j_s, k_s - 16, k_t]
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s // 16 == k_t // 16:
                a32i_fragment[j_s, k_s, k_t] = a16i_shared[
                    j_s * 2 + k_s // 16, k_s % 16, k_t % 16
                ]
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if j_s == 0:
                a32i0_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]
            else:
                a32i1_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]

        T.clear(a32o_fragment)
        for k_r in T.serial(32):
            for k_s, k_t in T.Parallel(32, 32):
                a32o_fragment[k_s, k_t] += (
                    a32i1_shared[k_s, k_r] * a32o_shared[k_r, k_t]
                )
        T.copy(a32o_fragment, a32o_shared)

        T.clear(a32o_fragment)
        for k_r in T.serial(32):
            for k_s, k_t in T.Parallel(32, 32):
                a32o_fragment[k_s, k_t] += (
                    a32o_shared[k_s, k_r] * a32i0_shared[k_r, k_t]
                )

        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            a64_shared[j_s * 32 + k_s, j_s * 32 + k_t] = a32i_fragment[j_s, k_s, k_t]
        for k_s, k_t in T.Parallel(32, 32):
            a64_shared[32 + k_s, k_t] = a32o_fragment[k_s, k_t]
        for k_s, k_t in T.Parallel(32, 32):
            a64_shared[k_s, 32 + k_t] = 0
        T.sync_threads()

        if right <= seq_end_idx:
            T.copy(a64_shared, a[bb, left:right, bh, 0:block_S])
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    a[bb, left + j_s, bh, j_t] = a64_shared[j_s, j_t]

    if is_varlen:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    bb,
                    bc,
                    bh,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    b,
                    a,
                )

    else:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb,
                    bc,
                    bh,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    b,
                    a,
                )

    return tilelang_kkt_solve_kernel

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_chunk_gdr_fwd_sm70(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    h_dtype,
    o_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    store_h,
    store_o,
    is_varlen,
    is_cp,
    state_layout_vlk,
    indexed_state=False,
    block_DV=128,
    threads=128,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    raw_batch_size = T.dynamic("raw_batch_size")
    state_cache_size = T.dynamic("state_cache_size")
    block_S = chunk_size

    if is_varlen:
        q_shape = (1, num_tokens, Hg, DK)
        k_shape = (1, num_tokens, Hg, DK)
        v_shape = (1, num_tokens, H, DV)
        o_shape = (1, num_tokens, H, DV)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
        h_shape = (1, num_chunks, H, DK, DV)
    else:
        q_shape = (batch_size, num_tokens, Hg, DK)
        k_shape = (batch_size, num_tokens, Hg, DK)
        v_shape = (batch_size, num_tokens, H, DV)
        o_shape = (batch_size, num_tokens, H, DV)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
        h_shape = (batch_size, num_chunks, H, DK, DV)
    h0_batch = state_cache_size if indexed_state else batch_size
    h0_shape = (h0_batch, H, DK, DV)
    ht_shape = (raw_batch_size, H, DK, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        cp_seq_map: T.Tensor([batch_size], dtype=seqlen_dtype),
        raw_cu_seqlens: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
        state_indices: T.Tensor([batch_size], dtype="int32"),
        has_initial_state: T.Tensor([batch_size], dtype="bool"),
        o: T.Tensor(o_shape, dtype=o_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(
            T.ceildiv(DV, block_DV) * batch_size * H, threads=threads
        ) as (bbhv,):
            bbh, bv = bbhv // T.ceildiv(DV, block_DV), bbhv % T.ceildiv(DV, block_DV)
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            chunk_start_idx = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            chunk_start_idx = chunk_offsets[bb] if is_varlen else 0

            raw_batch_idx = T.alloc_var("int32")
            raw_seq_end_idx = T.alloc_var("int32")
            need_store_final_state = T.alloc_var("bool")
            raw_batch_idx = cp_seq_map[bb] if is_cp else bb
            raw_seq_end_idx = (
                raw_cu_seqlens[raw_batch_idx + 1] if is_cp else seq_end_idx
            )
            need_store_final_state = store_final_state & (
                raw_seq_end_idx == seq_end_idx
            )

            state_slot = T.alloc_var("int32")
            state_slot = state_indices[bb] if indexed_state else bb

            num_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S)

            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            kt_shared = T.alloc_shared((DK, block_S), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            vn_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            if use_initial_state:
                if indexed_state and state_layout_vlk:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] = T.if_then_else(
                            has_initial_state[bb],
                            h0[
                                state_slot,
                                bh,
                                bv * block_DV + j_v,
                                j_k,
                            ],
                            0.0,
                        )
                elif indexed_state:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] = T.if_then_else(
                            has_initial_state[bb],
                            h0[
                                state_slot,
                                bh,
                                j_k,
                                bv * block_DV + j_v,
                            ],
                            0.0,
                        )
                elif state_layout_vlk:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] = h0[
                            bb,
                            bh,
                            bv * block_DV + j_v,
                            j_k,
                        ]
                else:
                    T.copy(
                        h0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV],
                        h_fragment,
                    )
            else:
                T.clear(h_fragment)

            for i_s in T.serial(num_iters):
                left = seq_start_idx + i_s * block_S
                right = left + block_S

                if right <= seq_end_idx:
                    T.copy(q[batch_idx, left:right, bhg, 0:DK], q_shared)
                    T.copy(k[batch_idx, left:right, bhg, 0:DK], k_shared)
                    T.copy(
                        v[
                            batch_idx,
                            left:right,
                            bh,
                            bv * block_DV : (bv + 1) * block_DV,
                        ],
                        v_shared,
                    )
                    T.copy(a[batch_idx, left:right, bh, 0:block_S], a_shared)
                    for j_s in T.Parallel(block_S):
                        g_shared[j_s] = g[batch_idx, left + j_s, bh]
                    for j_s in T.Parallel(block_S):
                        b_shared[j_s] = b[batch_idx, left + j_s, bh]
                else:
                    for j_s, j_k in T.Parallel(block_S, DK):
                        if left + j_s < seq_end_idx:
                            q_shared[j_s, j_k] = q[batch_idx, left + j_s, bhg, j_k]
                            k_shared[j_s, j_k] = k[batch_idx, left + j_s, bhg, j_k]
                        else:
                            q_shared[j_s, j_k] = 0
                            k_shared[j_s, j_k] = 0
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        if left + j_s < seq_end_idx:
                            v_shared[j_s, j_v] = v[
                                batch_idx,
                                left + j_s,
                                bh,
                                bv * block_DV + j_v,
                            ]
                        else:
                            v_shared[j_s, j_v] = 0
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if left + j_s < seq_end_idx:
                            a_shared[j_s, j_t] = a[batch_idx, left + j_s, bh, j_t]
                        else:
                            a_shared[j_s, j_t] = 0
                    for j_s in T.Parallel(block_S):
                        if left + j_s < seq_end_idx:
                            g_shared[j_s] = g[batch_idx, left + j_s, bh]
                            b_shared[j_s] = b[batch_idx, left + j_s, bh]
                        else:
                            g_shared[j_s] = g[batch_idx, seq_end_idx - 1, bh]
                            b_shared[j_s] = 0
                T.sync_threads()
                for j_s, j_k in T.Parallel(block_S, DK):
                    kt_shared[j_k, j_s] = k_shared[j_s, j_k]
                T.sync_threads()

                T.copy(h_fragment, h_shared)
                if store_h:
                    T.copy(
                        h_shared,
                        h[
                            batch_idx,
                            chunk_start_idx + i_s,
                            bh,
                            0:DK,
                            bv * block_DV : (bv + 1) * block_DV,
                        ],
                    )
                T.sync_threads()

                for j_s in T.Parallel(block_S):
                    g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                for j_s in T.Parallel(block_S):
                    g_rev_exp_shared[j_s] = T.if_then_else(
                        left + j_s < seq_end_idx,
                        T.exp2((g_shared[block_S - 1] - g_shared[j_s]) * 1.442695),
                        0.0,
                    )
                T.sync_threads()

                T.gemm(
                    q_shared, k_shared, p_fragment, transpose_B=True, clear_accum=True
                )

                for j_s, j_t in T.Parallel(block_S, block_S):
                    g_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if j_s >= j_t:
                        g_fragment[j_s, j_t] = T.exp2(g_fragment[j_s, j_t] * 1.442695)
                    else:
                        g_fragment[j_s, j_t] = 0
                for j_s, j_t in T.Parallel(block_S, block_S):
                    a_fragment[j_s, j_t] = a_shared[j_s, j_t]
                for j_s, j_t in T.Parallel(block_S, block_S):
                    a_fragment[j_s, j_t] *= g_fragment[j_s, j_t]
                for j_s, j_t in T.Parallel(block_S, block_S):
                    a_fragment[j_s, j_t] *= b_shared[j_t]
                T.copy(a_fragment, a_shared)
                T.sync_threads()

                T.gemm(k_shared, h_shared, u_fragment, clear_accum=True)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    u_fragment[j_s, j_v] += v_shared[j_s, j_v]
                T.copy(u_fragment, v_shared)
                T.sync_threads()

                T.gemm(a_shared, v_shared, v_fragment, clear_accum=True)
                T.copy(v_fragment, vd_shared)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                T.copy(v_fragment, vn_shared)
                T.sync_threads()

                T.gemm(q_shared, h_shared, o_fragment, clear_accum=True)
                for j_s, j_t in T.Parallel(block_S, block_S):
                    p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                T.copy(p_fragment, p_shared)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
                T.sync_threads()

                T.gemm(p_shared, vd_shared, o_fragment, clear_accum=False)

                if store_o:
                    if right <= seq_end_idx:
                        T.copy(
                            o_fragment,
                            o[
                                batch_idx,
                                left:right,
                                bh,
                                bv * block_DV : (bv + 1) * block_DV,
                            ],
                        )
                    else:
                        for j_s, j_v in T.Parallel(block_S, block_DV):
                            if left + j_s < seq_end_idx:
                                o[
                                    batch_idx,
                                    left + j_s,
                                    bh,
                                    bv * block_DV + j_v,
                                ] = o_fragment[j_s, j_v]

                g_last_local[0] = g_exp_shared[block_S - 1]
                for j_k, j_v in T.Parallel(DK, block_DV):
                    h_fragment[j_k, j_v] *= g_last_local[0]
                T.gemm(kt_shared, vn_shared, h_fragment, clear_accum=False)

            if need_store_final_state:
                if indexed_state and state_layout_vlk:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h0[state_slot, bh, bv * block_DV + j_v, j_k] = h_fragment[
                            j_k, j_v
                        ]
                elif indexed_state:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h0[state_slot, bh, j_k, bv * block_DV + j_v] = h_fragment[
                            j_k, j_v
                        ]
                elif state_layout_vlk:
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        ht[raw_batch_idx, bh, bv * block_DV + j_v, j_k] = h_fragment[
                            j_k, j_v
                        ]
                else:
                    T.copy(
                        h_fragment,
                        ht[
                            raw_batch_idx,
                            bh,
                            0:DK,
                            bv * block_DV : (bv + 1) * block_DV,
                        ],
                    )

    return tilelang_fused_chunk_gdr_fwd_kernel
