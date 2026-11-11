import torch
import torch.distributed as dist

from .partition import SequencePartition


def _exchange(send, receive, peer, group=None):
    operations = [
        dist.P2POp(dist.isend, send.contiguous(), peer, group=group),
        dist.P2POp(dist.irecv, receive, peer, group=group),
    ]
    requests = dist.batch_isend_irecv(operations)
    for request in requests:
        request.wait()


def send_tensor(tensor, peer, group=None):
    requests = dist.batch_isend_irecv(
        [dist.P2POp(dist.isend, tensor.contiguous(), peer, group=group)]
    )
    for request in requests:
        request.wait()


def receive_tensor(tensor, peer, group=None):
    requests = dist.batch_isend_irecv(
        [dist.P2POp(dist.irecv, tensor, peer, group=group)]
    )
    for request in requests:
        request.wait()
    return tensor


def exchange_sequence(local, partition, rank, group=None):
    if not isinstance(partition, SequencePartition):
        raise TypeError("partition must be a SequencePartition")
    peer = partition.peer(rank)
    receive = torch.empty(
        (partition.size(peer), *local.shape[1:]),
        dtype=local.dtype,
        device=local.device,
    )
    _exchange(local, receive, peer, group=group)
    return (local, receive) if rank == 0 else (receive, local)


def reduce_scatter_sequence(partial, partition, rank, group=None):
    if partial.shape[0] != partition.total:
        raise ValueError(
            f"partial output has {partial.shape[0]} rows, expected {partition.total}"
        )
    start, stop = partition.bounds(rank)
    peer_start, peer_stop = partition.bounds(partition.peer(rank))
    own = partial[start:stop].contiguous()
    receive = torch.empty_like(own)
    _exchange(partial[peer_start:peer_stop], receive, partition.peer(rank), group=group)
    own.add_(receive)
    return own
