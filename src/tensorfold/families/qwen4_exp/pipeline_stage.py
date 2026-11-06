"""Private fixed PP2 block/endpoint ownership; no transport or serving activation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStage:
    index: int

    def __post_init__(self):
        if type(self.index) is not int or self.index not in (0, 1):
            raise ValueError("pipeline stage must be index zero or one")

    @property
    def first(self):
        return self.index * 24

    @property
    def stop(self):
        return self.first + 24

    @property
    def layer_ids(self):
        return range(self.first, self.stop)

    @property
    def owns_input(self):
        return self.index == 0

    @property
    def owns_output(self):
        return self.index == 1

    def identity(self):
        return {"schema": "tensorfold.qwen4_exp.pipeline_stage.v1", "index": self.index,
                "pipeline_parallel_size": 2, "global_layers": 48,
                "first": self.first, "stop": self.stop,
                "owns_input": self.owns_input, "owns_output": self.owns_output}


def for_weights(weights):
    """Validate actual local blocks/endpoints, never infer a stage from a slice."""
    identity = weights.meta.get("pipeline_stage")
    if identity is None:
        return None
    if type(identity) is not dict or "index" not in identity:
        raise ValueError("complete pipeline stage identity required")
    stage = PipelineStage(identity["index"])
    expected = stage.identity()
    if identity.keys() != expected.keys() or any(type(identity[k]) is not type(v) or identity[k] != v
                                                for k, v in expected.items()):
        raise ValueError("pipeline stage identity differs from the fixed PP2 profile")
    if (type(weights.meta.get("world")) is not int or weights.meta["world"] != 8
            or type(weights.meta.get("rank")) is not int or not 0 <= weights.meta["rank"] < 8
            or weights.cfg.quant != "gguf" or weights.cfg.execution_dtype != "float16"
            or weights.cfg.streams != 4 or weights.meta.get("execution_provider") != "sm70"
            or weights.meta.get("execution_dtype") != "float16"
            or weights.meta.get("history_layout") != "unique" or weights.meta.get("attention_degree") != 4
            or weights.meta.get("indexer_history") != "gpu"):
        raise ValueError("private pipeline components require the GGUF FP16 TP8 stage-local profile")
    if stage.owns_input and weights.mtp is not None:
        raise ValueError("input pipeline stage cannot own MTP")
    full = weights.meta.get("full")
    if (type(weights.cfg.layers) is not int or weights.cfg.layers != 48
            or len(weights.cfg.layer_types) != 48 or full is None or full.layers != 48
            or full.layer_types != weights.cfg.layer_types
            or [layer.index for layer in weights.layers] != list(stage.layer_ids)):
        raise ValueError("pipeline weights must retain global48 config and exact local global block IDs")
    if ((weights.embed is not None) != stage.owns_input
            or (weights.mixer is not None) != stage.owns_output
            or (weights.head is not None) != stage.owns_output):
        raise ValueError("pipeline endpoint weights do not match their sole owner")
    if not stage.owns_input and (weights.cfg.ple_layers or any(layer.ple is not None for layer in weights.layers)):
        raise ValueError("output stage cannot own PLE before the required owner-row service")
    return stage
