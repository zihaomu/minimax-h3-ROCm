from dataclasses import dataclass

import torch
import torch.distributed as dist

import comfy.ldm.common_dit
import comfy.model_management
from comfy.ldm.minimax.model import (
    AUDIO_COND_TIMESTEP,
    VISUAL_COND_TIMESTEP,
    PackedLayout,
    pack_audio,
    patchify_video,
    rope_rotation_table,
    time_shift_sigma,
    time_shift_slope,
    unpack_audio,
    unpatchify_video,
)

from .parallel_blocks import run_blocks
from .partition import SequencePartition
from .peer_ops import receive_tensor, send_tensor


@dataclass
class ForwardState:
    orig_t: int
    orig_h: int
    orig_w: int
    latent_t: int
    latent_h: int
    latent_w: int
    layout: object
    t_rows: dict
    segment_times: dict
    sigma_video: torch.Tensor
    shift_video: float
    shift_audio: float


def _build_segments(layout, payload, video_time, audio_time, visual_aug, audio_aug):
    has_visual_condition = any(
        kind in ("cond", "ref_img") for _, _, kind in layout.segments
    )
    has_audio_condition = any(kind == "ref_audio" for _, _, kind in layout.segments)
    segment_times = {
        "text": video_time,
        "video": video_time,
        "audio": audio_time,
        "cond": max(video_time, visual_aug),
        "ref_img": max(video_time, visual_aug),
        "ref_audio": max(audio_time, audio_aug),
    }
    unique_times = sorted(
        {video_time, audio_time}
        | ({segment_times["cond"]} if has_visual_condition else set())
        | ({segment_times["ref_audio"]} if has_audio_condition else set())
    )
    time_rows = {value: index for index, value in enumerate(unique_times)}
    modality = {
        "text": 1,
        "video": 0,
        "audio": 2,
        "cond": 0,
        "ref_img": 0,
        "ref_audio": 2,
    }

    text_tags = payload.get("text_token_tags")
    modulation_segments = []
    for start, stop, kind in layout.segments:
        row_base = time_rows[segment_times[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for offset in range(1, stop - start + 1):
                if offset == stop - start or tags[offset] != tags[run_start]:
                    modulation_segments.append(
                        (
                            start + run_start,
                            start + offset,
                            row_base + int(tags[run_start]),
                        )
                    )
                    run_start = offset
        else:
            modulation_segments.append(
                (start, stop, row_base + modality[kind])
            )
    return unique_times, time_rows, segment_times, modulation_segments


def prepare_rank_zero(
    dit,
    x,
    timestep,
    context,
    transformer_options,
    minimax_payload,
):
    video, audio = x[0], x[1]
    orig_t, orig_h, orig_w = video.shape[2:]
    video = comfy.ldm.common_dit.pad_to_patch_size(video, dit.patch_size)
    if video.shape[0] != 1:
        raise ValueError("MiniMax H3 sequence parallelism supports batch size 1")

    payload = minimax_payload or {}
    device = video.device
    dtype = context.dtype
    latent_t, latent_h, latent_w = video.shape[2:]
    audio_t = audio.shape[-1]
    text_length = context.shape[1]

    layout = payload.get("layout")
    signature = (text_length, latent_t, latent_h, latent_w, audio_t)
    if layout is None or layout.signature != signature:
        layout = PackedLayout(
            text_length,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
            frame_count=payload.get("frame_count"),
        )

    shift_video = float(
        transformer_options.get(
            "minimax_h3_sigma_shift_video",
            dit.sigma_shift_video,
        )
    )
    shift_audio = float(
        transformer_options.get(
            "minimax_h3_sigma_shift_audio",
            dit.sigma_shift_audio,
        )
    )
    sigma_video = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    video_time = float(1.0 - sigma_video)
    audio_time = float(
        1.0 - time_shift_sigma(sigma_video, shift_video, shift_audio)
    )
    visual_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    audio_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    unique_times, time_rows, segment_times, modulation_segments = _build_segments(
        layout,
        payload,
        video_time,
        audio_time,
        visual_aug,
        audio_aug,
    )

    image_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = patchify_video(video.to(torch.float32), dit.patch_size)
    audio_rows = pack_audio(audio.to(torch.float32))
    conditioned_video = dit._cond_video_rows(payload, device)
    conditioned_audio = dit._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if conditioned_video is not None:
        all_video_rows = torch.empty(
            image_update.shape[0],
            video_rows.shape[1],
            dtype=torch.float32,
            device=device,
        )
        all_video_rows[~image_update] = conditioned_video
        all_video_rows[image_update] = video_rows

    all_audio_rows = audio_rows
    if conditioned_audio is not None:
        all_audio_rows = torch.empty(
            audio_update.shape[0],
            audio_rows.shape[1],
            dtype=torch.float32,
            device=device,
        )
        all_audio_rows[~audio_update] = conditioned_audio
        all_audio_rows[audio_update] = audio_rows

    video_embedding = dit.video_patch_proj(all_video_rows).to(dtype)
    audio_embedding = dit.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != dit.hidden_size:
        text_states = dit.token_refiner(
            dit.condition_proj(text_states),
            transformer_options=transformer_options,
        )

    hidden = torch.empty(
        layout.seq_len,
        dit.hidden_size,
        dtype=dtype,
        device=device,
    )
    video_offset = 0
    audio_offset = 0
    for start, stop, kind in layout.segments:
        rows = stop - start
        if kind == "text":
            hidden[start:stop] = text_states
        elif kind in ("cond", "ref_img", "video"):
            hidden[start:stop] = video_embedding[video_offset:video_offset + rows]
            video_offset += rows
        else:
            hidden[start:stop] = audio_embedding[audio_offset:audio_offset + rows]
            audio_offset += rows

    time_values = torch.tensor(unique_times, dtype=torch.float32, device=device)
    if dit.use_adaln_curves:
        table = comfy.model_management.cast_to(dit.adaln_t_table, device=device)
        position = time_values.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        time_embedding = torch.lerp(
            table[lower],
            table[lower + 1],
            (position - lower).unsqueeze(1),
        )
    else:
        time_embedding = dit.time_embedder(time_values).to(dtype)

    rope = rope_rotation_table(
        dit.rope_freqs(layout.position_ids, device),
        dtype,
    )
    state = ForwardState(
        orig_t,
        orig_h,
        orig_w,
        latent_t,
        latent_h,
        latent_w,
        layout,
        time_rows,
        segment_times,
        sigma_video,
        shift_video,
        shift_audio,
    )
    return hidden, time_embedding, rope, modulation_segments, state


def finish_rank_zero(dit, hidden, time_embedding, state, video_dtype, audio_dtype):
    video_segment = next(
        (
            start,
            stop,
            state.t_rows[state.segment_times["video"]],
        )
        for start, stop, kind in state.layout.segments
        if kind == "video"
    )
    audio_segment = next(
        (
            start,
            stop,
            state.t_rows[state.segment_times["audio"]],
        )
        for start, stop, kind in state.layout.segments
        if kind == "audio"
    )
    video_rows, audio_rows = dit.final_layer(
        hidden,
        time_embedding,
        video_segment,
        audio_segment,
    )
    video_output = unpatchify_video(
        video_rows,
        state.latent_t,
        state.latent_h // 2,
        state.latent_w // 2,
        dit.latents_dim,
        dit.patch_size,
    )
    video_output = video_output[
        :,
        :,
        :state.orig_t,
        :state.orig_h,
        :state.orig_w,
    ]
    audio_output = unpack_audio(audio_rows)
    audio_slope = time_shift_slope(
        state.sigma_video,
        state.shift_video,
        state.shift_audio,
    ).to(audio_output.dtype)
    return [
        -video_output.to(video_dtype),
        (-audio_slope) * audio_output.to(audio_dtype),
    ]


def send_rank_one_inputs(
    hidden,
    time_embedding,
    rope,
    segments,
    object_group,
    reset_cache=False,
):
    partition = SequencePartition(hidden.shape[0])
    metadata = {
        "op": "forward",
        "total_rows": hidden.shape[0],
        "hidden_width": hidden.shape[1],
        "hidden_dtype": str(hidden.dtype).replace("torch.", ""),
        "time_shape": tuple(time_embedding.shape),
        "time_dtype": str(time_embedding.dtype).replace("torch.", ""),
        "rope_shape": tuple(rope.shape),
        "rope_dtype": str(rope.dtype).replace("torch.", ""),
        "segments": segments,
        "reset_cache": reset_cache,
    }
    dist.broadcast_object_list([metadata], src=0, group=object_group)
    start, stop = partition.bounds(1)
    send_tensor(hidden[start:stop], peer=1)
    dist.broadcast(time_embedding.contiguous(), src=0)
    dist.broadcast(rope.contiguous(), src=0)
    return partition


def gather_rank_one_output(local_hidden, partition):
    peer_start, peer_stop = partition.bounds(1)
    peer_hidden = torch.empty(
        (peer_stop - peer_start, local_hidden.shape[1]),
        dtype=local_hidden.dtype,
        device=local_hidden.device,
    )
    receive_tensor(peer_hidden, peer=1)
    return torch.cat((local_hidden, peer_hidden), dim=0)


def run_rank_zero(
    dit,
    x,
    timestep,
    context,
    transformer_options,
    minimax_payload,
    object_group,
    reset_cache=False,
    profile=False,
):
    hidden, time_embedding, rope, segments, state = prepare_rank_zero(
        dit,
        x,
        timestep,
        context,
        transformer_options,
        minimax_payload,
    )
    partition = send_rank_one_inputs(
        hidden,
        time_embedding,
        rope,
        segments,
        object_group,
        reset_cache=reset_cache,
    )
    start, stop = partition.bounds(0)
    local_hidden = run_blocks(
        dit.blocks,
        hidden[start:stop].contiguous(),
        time_embedding,
        segments,
        rope,
        hidden.shape[0],
        rank=0,
        transformer_options=transformer_options,
        profile=profile,
    )
    hidden = gather_rank_one_output(local_hidden, partition)
    return finish_rank_zero(
        dit,
        hidden,
        time_embedding,
        state,
        x[0].dtype,
        x[1].dtype,
    )
