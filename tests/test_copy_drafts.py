"""The retained copy index proposes earlier continuations without importing a GPU runtime."""

import ast
from collections.abc import Sequence
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen3_5/cuda/decode.py"
parsed = ast.parse(SOURCE.read_text())
definitions = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
               and node.name in {"CopyIndex", "copy_chain"}]
namespace = {"Sequence": Sequence}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
CopyIndex, copy_chain = namespace["CopyIndex"], namespace["copy_chain"]
COPY_MATCH = 8

A = list(range(100, 140))                 # 40 distinct tokens
B = list(range(200, 212))                 # 12 more


def test_no_copy_no_chain():
    assert CopyIndex().propose(A, 15) == []
    assert CopyIndex().propose(A + A[:COPY_MATCH - 1], 15) == []  # the tail repeats fewer than COPY_MATCH tokens


def test_a_repeated_tail_drafts_its_continuation():
    ctx = A + B + A[:COPY_MATCH]                                    # the tail is A's first 8 tokens again
    assert CopyIndex().propose(ctx, 15) == A[COPY_MATCH:COPY_MATCH + 15]
    assert CopyIndex().propose(ctx, 40) == A[COPY_MATCH:] + B[:8]  # up to max_nodes, across what followed


def test_a_round_without_room_for_a_chain_gets_none():
    ctx = A + B + A[:COPY_MATCH]
    assert CopyIndex().propose(ctx, COPY_MATCH - 1) == []  # the MTP chain takes such a round
    assert CopyIndex().propose(ctx, COPY_MATCH) == A[COPY_MATCH:2 * COPY_MATCH]


def test_latest_equal_length_copy_wins_and_the_needle_is_not_a_copy():
    first = A[:COPY_MATCH] + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    second = A[:COPY_MATCH] + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ctx = first + second + A[:COPY_MATCH]
    assert CopyIndex().propose(ctx, 10) == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # a tail repeated back to back: the earlier copy is followed by the needle, which is drafted (the pattern repeats)
    assert CopyIndex().propose(A[:COPY_MATCH] + A[:COPY_MATCH], 15) == A[:COPY_MATCH]


def test_an_older_copy_wins_when_its_continuation_is_longer():
    earlier = A[:COPY_MATCH] + list(range(300, 324))
    latest = A[:COPY_MATCH] + list(range(400, 408))
    context = earlier + latest + A[:COPY_MATCH]
    assert CopyIndex().propose(context, 24) == copy_chain(context, 24) == list(range(300, 324))


def test_incremental_updates_equal_rebuilding_and_full_scan():
    ctx = A + B + A[:COPY_MATCH + 4]
    grown = CopyIndex()
    for length in range(30, len(ctx) + 1):
        prefix = ctx[:length]
        assert grown.propose(prefix, 15) == CopyIndex().propose(prefix, 15) == copy_chain(prefix, 15)
    assert grown.propose(ctx, 15) == A[COPY_MATCH + 4:COPY_MATCH + 4 + 15]


def _grown(rng, vocab, size):
    context = [rng.randrange(vocab) for _ in range(size)]
    for _ in range(40):                                      # repeated stretches, so matches exist
        at = rng.randrange(len(context) - 300)
        context += context[at:at + rng.randint(8, 300)]
    return context


def test_a_long_prompts_bulk_index_proposes_what_the_full_scan_does_as_it_grows():
    import random

    rng = random.Random(9)
    for vocab, size in ((6, 6_000), (50, 9_000), (248_000, 20_000)):
        context, index = _grown(rng, vocab, size), CopyIndex()
        assert len(context) - COPY_MATCH > CopyIndex.BULK
        for step in range(60):                               # the first propose builds the bulk; later grams join
            assert index.propose(context, 64) == copy_chain(context, 64), (vocab, step)
            context += context[-rng.randint(1, 40):] if rng.random() < 0.5 else [rng.randrange(vocab)]
        assert index.hashes is not None and index.positions


def test_hash_collisions_are_never_proposed():
    import random

    class Colliding(CopyIndex):                              # a hash with few values: almost every gram collides
        @staticmethod
        def _hashes(tokens, n):
            return CopyIndex._hashes(tokens, n) % 5

    rng = random.Random(3)
    context, index = _grown(rng, 40, 6_000), Colliding()
    for step in range(40):
        assert index.propose(context, 64) == copy_chain(context, 64), step
        context += [rng.randrange(40) for _ in range(rng.randint(1, 5))]
