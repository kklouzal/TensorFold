"""One resident SM70 FP16 vision encoder using the pinned library's exact tiling.

The source-built Torch efficient-attention backend has native SM70 FP16/D72
dispatch and linear output/LSE/optional accumulator storage. Only that backend
is enabled; unsupported execution refuses rather than allocating a math-backend
score matrix. Actual V100 numerical/resource/performance qualification is still
required. Placement and largest-unit bounds come from the admitted media profile.
"""

from dataclasses import dataclass

import torch

from .mmproj_weights import load_cpu_vision_model


@dataclass(frozen=True)
class EncodedUnit:
    features: torch.Tensor
    done: object
    grids: tuple


def _grids(value, layout, max_patches):
    if (type(max_patches) is not int or not 0 < max_patches <= 2**31 - 1
            or not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.int64
            or value.ndim != 2 or value.shape[1] != 3 or not 0 < value.shape[0] <= max_patches):
        raise ValueError("vision units require a bounded CPU int64 THW grid")
    result = tuple(tuple(int(v) for v in row) for row in value.tolist())
    if any(min(row) <= 0 or row[1] % layout.merge or row[2] % layout.merge for row in result):
        raise ValueError("vision grids require positive merge-aligned spatial dimensions")
    count = sum(t * h * w for t, h, w in result)
    if count > max_patches:
        raise ValueError("the complete vision attention unit exceeds its admitted patch bound")
    return result, count


def vision_arithmetic():
    result = {"attention": "torch-efficient-sdpa-sm70-fp16-f32acc",
              "linear_fp16_accumulation": torch.backends.cuda.matmul.allow_fp16_accumulation,
              "linear_fp16_reduced_precision": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
              "linear_fp16_split_k": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction_split_k,
              "axial_frequency": "pinned-hf-float32", "parameters": "float16"}
    if result["linear_fp16_accumulation"] or result["linear_fp16_reduced_precision"]:
        raise ValueError("vision requires the admitted FP32 GEMM accumulation/reduction policy")
    return result


class SM70VisionEncoder:
    @classmethod
    def create(cls, plan, reader, *, staging_bytes, device, max_unit_patches, software_binding, register):
        """Journal the actual partial owner before any fallible module/device work."""
        if (not callable(register) or type(max_unit_patches) is not int
                or not 0 < max_unit_patches <= 2**31 - 1):
            raise ValueError("vision construction requires an owner journal and explicit unit bound")
        owner = cls.__new__(cls)
        owner.model = owner.done = owner.inputs = None
        owner.closed = owner.failed = owner.gpu_started = False
        owner.device = torch.device(device)
        owner.plan, owner.max_unit_patches = plan, max_unit_patches
        owner.software_binding = software_binding
        register(owner)
        try:
            owner._initialize(reader, staging_bytes)
        except BaseException:
            owner.failed = True
            raise
        return owner

    def _initialize(self, reader, staging_bytes):
        from tensorfold.cuda.sm70.media_software_binding import verify_media_binding

        verify_media_binding(self.software_binding)
        if self.device.type != "cuda":
            raise ValueError("the resident vision encoder requires its fixed CUDA device")
        self.arithmetic = vision_arithmetic()
        self.gpu_started = True
        if torch.cuda.get_device_capability(self.device) != (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
            raise ValueError("this vision profile requires an actual SM70-enabled framework/device")
        if self.plan.layout.hidden // self.plan.layout.heads != 72 or self.plan.layout.heads != 16:
            raise ValueError("this initial native vision profile requires the declared H16/D72 geometry")
        with torch.cuda.device(self.device):
            probe = torch.empty((1, 16, 4, 72), dtype=torch.float16, device=self.device)
            parameters = torch.backends.cuda.SDPAParams(probe, probe, probe, None, 0., False, False)
            if not torch.backends.cuda.can_use_efficient_attention(parameters):
                raise ValueError("the required SM70 FP16 efficient-attention backend is unavailable")
        self.model = load_cpu_vision_model(self.plan, reader, staging_bytes=staging_bytes)
        self.model.to(device=self.device)  # Preserve FP16 parameters and FP32 frequency buffers.

    def _retire_inputs(self):
        try:
            if self.done is not None:
                self.done.synchronize()
        except BaseException:
            self.failed = True
            raise
        self.inputs = self.done = None

    def encode_unit(self, pixels, grid):
        """Encode whole independent segments; never spatially truncate attention edges."""
        return self._encode_unit(pixels, grid, transferred=False)

    def encode_device_unit(self, pixels, grid):
        """Borrow admitted FP16 patches delivered by the TP input transport.

        The transport owns unchanged processor rows and their one FP16 cast,
        has joined the receive onto this stream, and retains the buffer through
        the returned event. Grid metadata remains authoritative CPU int64.
        """
        return self._encode_unit(pixels, grid, transferred=True)

    def _encode_unit(self, pixels, grid, *, transferred):
        if self.closed or self.failed or self.model is None:
            raise ValueError("vision encoder is closed, failed or not initialized")
        grids, count = _grids(grid, self.plan.layout, self.max_unit_patches)
        layout = self.plan.layout
        width = layout.channels * layout.temporal * layout.patch ** 2
        expected_device = self.device if transferred else torch.device("cpu")
        expected_dtype = torch.float16 if transferred else torch.float32
        if (not isinstance(pixels, torch.Tensor) or pixels.device != expected_device or pixels.dtype != expected_dtype
                or transferred and expected_device.type != "cuda"
                or pixels.shape != (count, width) or not pixels.is_contiguous()):
            raise ValueError("vision unit pixels differ from their admitted CPU FP32 or transferred CUDA FP16 contract")
        self._retire_inputs()  # One bounded encoder working set at a time.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers.vision_utils import (
            get_vision_cu_seqlens, get_vision_interpolation_indices_and_weights, get_vision_position_ids,
        )

        # CPU metadata remains authoritative; library data-dependent helpers do
        # not force GPU scalar readback during encoding. Video temporal groups
        # preserve exactly the library's noncausal segment boundaries.
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid, self.model.num_grid_per_side, mode="bilinear", align_corners=True,
            spatial_merge_size=layout.merge)
        positions = get_vision_position_ids(grid, layout.merge)
        lengths = get_vision_cu_seqlens(grid)
        self.inputs = [pixels, grid, indices, weights, positions, lengths]
        try:
            with torch.cuda.device(self.device), torch.inference_mode(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                # Journal each materialized input before the next fallible copy.
                # All three grid-derived values are supplied below; the pinned
                # forward never reads grid again. Keep that metadata on CPU.
                for index in (0, 2, 3, 4):
                    dtype = torch.float16 if index == 0 else None
                    self.inputs[index] = self.inputs[index].to(device=self.device, dtype=dtype)
                result = self.model(self.inputs[0], self.inputs[1], interp_indices=self.inputs[2],
                                    interp_weights=self.inputs[3], position_ids=self.inputs[4],
                                    cu_seqlens=lengths)
                features = result.pooler_output
                if features.shape != (count // layout.merge ** 2, layout.output) or features.dtype != torch.float16:
                    raise ValueError("vision output differs from the declared merged row/precision contract")
                self.done = torch.cuda.Event()
                self.done.record(torch.cuda.current_stream(self.device))
            return EncodedUnit(features, self.done, grids)
        except BaseException:
            self.failed = True
            raise  # Engine journal retains this owner/inputs through coordinated fencing.

    def adopt_after_warm(self):
        """Startup/adoption boundary, never per request; immutable anchor cannot change."""
        from tensorfold.cuda.sm70.media_software_binding import reissue_media_binding, verify_media_binding

        if self.closed or self.failed or self.model is None:
            raise ValueError("uninitialized or failed vision encoder cannot be adopted")
        self._retire_inputs()
        self.software_binding = reissue_media_binding(self.software_binding)
        return verify_media_binding(self.software_binding)

    def close(self):
        if self.closed:
            return
        try:
            if self.gpu_started:
                torch.cuda.synchronize(self.device)
        except BaseException:
            self.failed = True
            raise
        self.inputs = self.done = self.model = self.software_binding = None
        self.closed = True
