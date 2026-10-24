"""Cold-catalogue native allocation bounds for immutable-limit State lanes.

This deliberately declines the configurable (1,10)MiB segment band. Outside
that band the pinned native allocator uses2MiB small segments or2MiB large
rounding, independent of its unexposed large_segment_size setting. Its
unlimited large-block splitting can retain at most a1MiB remainder. Cached
storage is never credited as a new allocation grant.

Source contract: PyTorch5c488690 c10/core/AllocatorConfig.h and
c10/cuda/CUDACachingAllocator.cpp::get_allocation_size/should_split.
"""


MIB = 1 << 20


class LaneAllocator:
    def __init__(self, identity, divisions):
        self.identity, self.divisions = identity, divisions

    def bounds(self, sizes):
        from tensorfold.cuda.allocation_bound import _rounded

        resident = physical = 0
        for size in sizes:
            if type(size) is not int or size < 0:
                raise ValueError("lane allocation sizes must be nonnegative integers")
            if not size:
                continue
            rounded = _rounded(size, self.divisions)
            if rounded <= MIB:
                resident += rounded
                physical += 2 * MIB
            elif rounded >= 10 * MIB:
                resident += rounded + MIB  # Native keeps an unsplit remainder <=1MiB.
                physical += -(-rounded // (2 * MIB)) * (2 * MIB)
            else:
                return None               # Effective medium-segment quantum is not exposed.
        return resident, physical

    def unchanged(self, torch):
        from tensorfold.cuda.allocation_bound import UnsupportedAllocator, _capture

        try:
            _, settings = _capture(torch)
        except UnsupportedAllocator:
            return False
        return settings == (self.identity, False, self.divisions)


def prepare(torch):
    from tensorfold.cuda.allocation_bound import UnsupportedAllocator, _capture

    try:
        snapshot, (identity, expandable, divisions) = _capture(torch)
    except UnsupportedAllocator:
        return None
    # Custom limited splitting could retain a much larger existing block.
    if expandable or snapshot["allocator_settings"].get("max_split_size") != -1:
        return None
    return LaneAllocator(identity, divisions)
