"""Qwen3.6 MoE on CUDA: MTP drafting and chain-graph settings."""

DEPTH = 3            # most MTP drafts a round
CONFIDENCE = 0.3     # a chain ends after a draft the MTP head gives less than this
CHAIN_BUCKET = 8192  # smallest context span a verify/draft graph covers; larger contexts take the next power of two
