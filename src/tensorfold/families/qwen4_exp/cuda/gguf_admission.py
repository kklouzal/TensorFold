"""GGUF loader allocation plans adapted to the shared CUDA admission contract."""

from tensorfold.cuda.capacity import Weights


def capacity_options(model_dir, rank, world, mtp=False, mtp_model_dir=None, draft_ids=None, prefill_plan=None):
    from .gguf_weights import model_plan, weight_byte_plan

    plan = weight_byte_plan(model_dir, tp=(rank, world), mtp=mtp, draft_ids=draft_ids,
                            **({"prefill_plan": prefill_plan} if prefill_plan is not None else {}))
    if mtp:
        if mtp_model_dir is None:
            raise ValueError("GGUF native MTP requires the matching stock draft checkpoint")
        from .sm70_mtp_weights import byte_plan

        prototype, _ = model_plan(model_dir, (rank, world))
        draft = byte_plan(prototype, mtp_model_dir, tp=(rank, world))
        plan["resident_bytes"] += draft["resident_bytes"]
        for field in ("staging_bytes", "host_staging_bytes"):
            plan[field] = max(plan[field], draft[field])
        plan["workspace_bytes"] += draft["workspace_bytes"]

    def estimate(_directory):
        return (Weights(int(plan["resident_bytes"]), int(plan["staging_bytes"]), 0),
                int(plan["host_staging_bytes"]))

    return {"weight_estimator": estimate,
            "host_resident": int(plan["host_resident_bytes"]),
            "host_extra_staging": 0,
            "device_workspace_bytes": int(plan.get("workspace_bytes", 0))}
