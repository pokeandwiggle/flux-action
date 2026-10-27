# Custom tasks and embodiments

Complete [setup](setup.md), including the base and encoder download.
The [gaming example](../examples/games/README.md) trains one policy on two browser games.

`flux-action train` reads a DROID or LeRobot index. To train on other data, provide the windows yourself
and configure the dataset, schedule and action dimensions below. The [DROID guide](droid-finetune.md)
describes the shared training, resume and export workflow.

## The window

Each sample is one dict: `images.<camera>` `(frames, 3, H, W)` uint8 with `frames = chunk_size + 1` (one
observed frame, then the frames the actions lead to), `state (D,)` float32, `action (chunk_size, D)` float32,
`task` (the caption, a string), `window_seed` (int), and optionally `action_mask (D,)`.

## The config

**`dataset = "pkg.module:callable"`.** The callable receives the `TrainConfig` and `seed, epoch, rank, world_size, num_workers, windows_per_rank, skip_batches, grad_accumulation`, and returns an iterable dataset with a `batches_per_rank` attribute that yields the windows above. Derive the sampling from `(seed, epoch, rank, position)` and skip the first `skip_batches`; resume is exact when you do. `index_dir` is ignored. A 60-line reference: `tests/training/test_embodiment_plug.py`.

**`frozen_steps`, `trunk_warmup_steps`, `heads_warmup_steps`.** The defaults (1000 / 2000 / 1000) fit a 30k-step
run. For 3000 steps use 200 / 600 / 200; with the defaults the trunk never unfreezes.

**`action_mask`,** only if embodiments with different action widths share one head. Zero-pad the narrower action
to `D` and set the mask per window; the loss averages over unmasked dims. The caption tells them apart.

**`export-checkpoint --dtype bfloat16`.** Export bf16 weights for inference; the default export keeps
the fp32 master weights. For a policy that conditions on one frame, `PolicyConfig.single_frame_encode`
encodes that frame alone instead of padding it to the VAE’s 45-frame chunk. This changes the encoded
latents and is off by default; retain the setting used by your checkpoint.

## Real-time chunking

A robot that replans while it still executes the previous chunk has already committed to its first few
actions. **`rtc_max_delay`** trains for that: each window draws `d` from `0..rtc_max_delay`,
keeps its first `d` action tokens clean (timestep 0) and leaves them out of the loss. `d` is uniform unless
**`rtc_delay_probs`** gives one probability per delay; a server that always pins `rtc_max_delay` actions trains
best on that delay and on 0 alone, e.g. `[0.75, 0, …, 0, 0.25]`. At inference,
`predict_action_chunk(batch, action_prefix=...)` takes those committed commands, `(B, d, D)` in dataset units
with `d <= rtc_max_delay`; every denoiser call sees them clean and the chunk returns them unchanged. It needs
the absolute action parameterization, and either the reference transformer (an unprepared policy) or the FP8r
backend (see [Precision](setup.md#precision)).

[`examples/games`](../examples/games/README.md) does all of the above on two browser games.
