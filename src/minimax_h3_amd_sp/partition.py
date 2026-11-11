from dataclasses import dataclass


@dataclass(frozen=True)
class SequencePartition:
    total: int
    world_size: int = 2

    def __post_init__(self):
        if self.total < 0:
            raise ValueError("total must be non-negative")
        if self.world_size != 2:
            raise ValueError("the optimized runtime currently supports exactly two ranks")

    @property
    def first_size(self):
        return (self.total + 1) // 2

    def bounds(self, rank):
        if rank == 0:
            return 0, self.first_size
        if rank == 1:
            return self.first_size, self.total
        raise ValueError(f"rank must be 0 or 1, got {rank}")

    def size(self, rank):
        start, stop = self.bounds(rank)
        return stop - start

    def peer(self, rank):
        if rank not in (0, 1):
            raise ValueError(f"rank must be 0 or 1, got {rank}")
        return 1 - rank


def localize_segments(segments, start, stop):
    local = []
    for seg_start, seg_stop, row in segments:
        left = max(seg_start, start)
        right = min(seg_stop, stop)
        if left < right:
            local.append((left - start, right - start, row))
    return local
