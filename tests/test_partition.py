from minimax_h3_amd_sp.partition import SequencePartition, localize_segments


def test_odd_partition_keeps_rank_zero_larger():
    partition = SequencePartition(11)
    assert partition.bounds(0) == (0, 6)
    assert partition.bounds(1) == (6, 11)


def test_segments_are_clipped_and_rebased():
    segments = [(0, 3, 10), (3, 8, 20), (8, 11, 30)]
    assert localize_segments(segments, 0, 6) == [(0, 3, 10), (3, 6, 20)]
    assert localize_segments(segments, 6, 11) == [(0, 2, 20), (2, 5, 30)]
