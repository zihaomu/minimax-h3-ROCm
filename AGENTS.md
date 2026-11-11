# MiniMax H3 ROCm

The parent workspace ROCm instructions also apply.

- Public source is src/, integrations/, tools/, tests/, configs/ and third_party/.
  Root __init__.py/nodes.py are ComfyUI discovery shims.
- The complete local research entrance is vibe/README.md. Read vibe/STATE.md
  before resuming experiments. Full theory, tensor inventories, logs, outputs,
  historical documents and migration records remain under vibe/ and archive/.
  These are protected local data, not public documentation or cache to delete.
- GitHub docs/ contains only RESULTS.md and results.json: curated key numbers,
  workload and qualification scope. Do not add raw logs, traces, full parameter
  inventories, output dumps or local research notes there.
- vibe/, doc/, archive/, models/, .runtime/ and runs/ must never enter the public
  tree or its ancestor commits. Preserve old histories on local/* or standard-opt/*
  branches; publish only main, never --all/--mirror or routine force pushes.
- Models remain under models/. Shared dependencies/caches/engines remain under
  .runtime/. Hidden .cache/.deps/.tools/.venv links preserve native paths;
  .standard-opt-state is the shared GPU-lock entrance. Do not split lock namespaces.
- Preserve shared files, model permissions and the 18 frozen hashes in
  archive/legacy-20261111/scripts/standard_opt/baseline_manifest.json. The original
  archive verifier remains archive/legacy-20261111/doc/verify_archive.py.
- Default attention is reduced_error with the retained frozen encoding/math,
  FP32 PV and BF16 output. Other backends are explicit controls, without automatic
  fallback. Precision exploration remains stopped by user choice.
- Keep configs/runtime-lock.json Torch/Triton/HIP/image identities. Source/path
  changes require real-input replay before Standard timing; retain numerical
  gates, fresh TE, 10 NFE, previews, audio and complete media checks.
- GPU3 is excluded. Select available non-GPU3 cards through the common workspace
  launcher and shared reservations, device identity, occupancy and ownership gates.
  Local restructuring and CPU checks do not launch GPU workloads.
- Historical 284.680954811s formal and 191.437884654s engineering records have
  different qualification scopes. Do not relabel them as current-main scores.
