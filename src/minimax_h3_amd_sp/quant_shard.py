import copy
import dataclasses

import torch


def _is_quantized_tensor(tensor):
    return type(tensor).__name__ == "QuantizedTensor"


def _replace_quant_params(weight, qdata, shape, index=None):
    updates = {"orig_shape": tuple(shape)}
    scale = getattr(weight._params, "scale", None)
    if index is not None and isinstance(scale, torch.Tensor):
        if scale.ndim > 0 and scale.numel() == weight.shape[0]:
            updates["scale"] = scale.reshape(-1)[index.to(scale.device)].contiguous()
    return dataclasses.replace(weight._params, **updates)


def _wrap_parameter(weight):
    return torch.nn.Parameter(weight, requires_grad=False)


def shard_linear_rows(linear, index):
    index = index.to(linear.weight.device)
    result = copy.copy(linear)
    result._parameters = dict(linear._parameters)

    weight = linear.weight
    if _is_quantized_tensor(weight):
        qdata = weight._qdata[index].contiguous()
        params = _replace_quant_params(
            weight,
            qdata,
            (index.numel(), weight.shape[1]),
            index=index,
        )
        sharded_weight = type(weight)(qdata, weight._layout_cls, params)
    else:
        sharded_weight = weight[index].contiguous()
    result._parameters["weight"] = _wrap_parameter(sharded_weight)

    if linear.bias is not None:
        result._parameters["bias"] = _wrap_parameter(linear.bias[index].contiguous())
    result.out_features = index.numel()
    if hasattr(result, "_orig_shape"):
        result._orig_shape = (result.out_features, result.in_features)
    return result


def shard_linear_columns(linear, start, stop):
    if linear.bias is not None:
        raise ValueError("column-sharded linear must not contain a bias")
    if not 0 <= start < stop <= linear.in_features:
        raise ValueError(
            f"invalid column range [{start}, {stop}) for {linear.in_features} inputs"
        )

    result = copy.copy(linear)
    result._parameters = dict(linear._parameters)
    weight = linear.weight

    if _is_quantized_tensor(weight):
        if tuple(weight._qdata.shape) != tuple(weight.shape):
            raise NotImplementedError(
                f"packed layout {weight._layout_cls} cannot be column-sharded safely"
            )
        group_size = getattr(weight._params, "convrot_groupsize", 1)
        if getattr(weight._params, "convrot", False):
            if start % group_size or stop % group_size:
                raise ValueError(
                    f"ConvRot column shard [{start}, {stop}) must align to group size "
                    f"{group_size}"
                )
        qdata = weight._qdata[:, start:stop].contiguous()
        params = _replace_quant_params(
            weight,
            qdata,
            (weight.shape[0], stop - start),
        )
        sharded_weight = type(weight)(qdata, weight._layout_cls, params)
    else:
        sharded_weight = weight[:, start:stop].contiguous()
    result._parameters["weight"] = _wrap_parameter(sharded_weight)

    result.in_features = stop - start
    if hasattr(result, "_orig_shape"):
        result._orig_shape = (result.out_features, result.in_features)

    for name in ("pre_quant_scale", "input_scale"):
        value = getattr(linear, name, None)
        if isinstance(value, torch.Tensor) and value.numel() == linear.in_features:
            setattr(result, name, _wrap_parameter(value.reshape(-1)[start:stop].contiguous()))
    return result


def attention_head_indices(heads, head_dim, rank, world_size=2, device=None):
    if world_size != 2:
        raise ValueError("only two-rank attention sharding is currently supported")
    if heads % world_size:
        raise ValueError(f"{heads} heads cannot be divided across {world_size} ranks")
    local_heads = heads // world_size
    inner = heads * head_dim
    first_head = rank * local_heads
    start = first_head * head_dim
    stop = start + local_heads * head_dim
    parts = [
        torch.arange(start, stop, device=device) + section * inner
        for section in range(3)
    ]
    return torch.cat(parts), start, stop
