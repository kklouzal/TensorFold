"""Pure metadata/owner bounds shared by the unconnected Sym8 consumer/loader.

Format byte counts are exact. Pack temporary bounds sum named NumPy owners and
expression intermediates without reusing storage credits; allocator/runtime
RSS and pageable CUDA staging remain measured/admitted separately.
"""
SCHEMA = 'tensorfold.bf16.sym8.v1'
SENTINEL = 0xffffff
MAX_DIM = (1 << 31) - 257
MAX_I64 = (1 << 63) - 1
PACK_CHUNK_GROUPS = 4096


def geometry(shape):
    if (type(shape) is not tuple or len(shape) != 2 or any(type(v) is not int for v in shape)
            or not 0 < shape[0] <= MAX_DIM or not 0 < shape[1] <= MAX_DIM or shape[1] % 32):
        raise ValueError('positive signed32 dimensions and K multiple32 required')
    if shape[0] * shape[1] > MAX_I64 // 2:
        raise ValueError('original BF16 byte count exceeds signed64')
    return shape[0] * (shape[1] // 32)


def allocation_bytes(groups, raw_count):
    if (type(groups) is not int or groups <= 0 or type(raw_count) is not int
            or not 0 <= raw_count <= min(groups, SENTINEL)):
        raise ValueError('valid group count and24-bit raw pool required')
    size = 36 * groups + 64 * raw_count
    if size > MAX_I64:
        raise ValueError('encoded byte count exceeds signed64')
    return size


def header_bounds(shape, *, chunk_groups=PACK_CHUNK_GROUPS):
    groups = geometry(shape)
    if type(chunk_groups) is not int or chunk_groups <= 0:
        raise ValueError('positive integer chunk_groups required')
    g = min(groups, chunk_groups)
    w, p = 32 * g, 8 * g
    # _plan: all named outputs + original sorted-word/word-count owners,
    # boolean comparison/expression temporaries. dtype widths explicit.
    inventory = {
        'plan_magnitude_u16_words': 2 * w,
        'plan_sorted_words_u16': 2 * w,
        'plan_word_count_intp64': 8 * g,
        'plan_sorted_magnitudes_u16': 2 * w,
        'plan_unique_bool_words': w,
        'plan_magnitude_count_intp64': 8 * g,
        'plan_zero_negative_valid_bool_groups': 3 * g,
        'plan_adjacent_nonzero_comparisons_bool': 2 * w,
        'plan_zero_word_comparisons_bool': 2 * w,
        'plan_valid_expression_bool_groups': 5 * g,
        'encode_palette_u16x8': 16 * g,
        'encode_rank_cumsum_and_subtract_int64': 16 * w,
        'encode_rank_compare_and_unique_mask_bool': 2 * w,
        'encode_nonzero_coordinates_two_intp64': 16 * p,
        'encode_palette_assignment_rank_int64_value_u16': 10 * p,
        'encode_extra_and_needlast_bool_intermediates': 4 * g,
        'encode_single_negative_bool_x8': 16 * g,
        'encode_sign_detection_comparisons_and_results': 2 * w + 10 * g,
        'encode_slot_prior_pos_neg_bool': 2 * g,
        'encode_slot_xor_bool': g,
        'encode_choose_where_int64x8_and_max_int64': 72 * g,
        'encode_choose_error_bool': 2 * g,
        'encode_rows_and_selected_intp64': 16 * g,
        'encode_palette_swap_u16_values_and_copy': 6 * g,
        'encode_extra_assignment_bool': g,
        'encode_codes_u8_words': w,
        'encode_sign_shift_multiply_u16_and_u8': 5 * w,
        'encode_match_bool_intermediates_and_previous': 4 * w,
        'encode_code_values_u8_and_selected_result': 2 * w,
        # np.where with literal8 and uint8 sign returns uint8 in the pinned
        # NumPy target; astype(uint8) owns its separate returned copy.
        'encode_last_where_u8_and_cast_u8': 2 * w,
        'encode_flag_u32_expression_intermediates': 24 * g,
        'encode_packed_nibble_shift_and_result_u8': w,
        'caller_previous_valid_palette_codes_flags_indices': 41 * g,
        'caller_current_raw_indices_u32': 4 * g,
        'caller_raw_indices_arange_u32': 4 * g,
        'caller_escape_mask_and_raw_selection_u16': g + 2 * w,
        'small_fixed_slot_arange_int64x8': 64,
    }
    return {'groups': groups, 'raw_source_bytes': shape[0] * shape[1] * 2,
            'max_valid_encoded_bytes': allocation_bytes(groups, min(groups, SENTINEL)),
            'pack_chunk_groups': g, 'pack_temporary_inventory': inventory,
            'pack_temporary_bytes': sum(inventory.values()),
            'cpu_finite_chunk_bytes': 5 * w + 5,
            'scope': 'logical named owners; no allocator/pageable-driver-overhead or reuse credit'}
