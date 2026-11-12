#!/usr/bin/env bash
set -euo pipefail
# Consumer-owned selection and tracing flags; the SDK remains the producer.
exec /opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/bin/rocprofv3 \
  --selected-regions --memory-copy-trace --memory-allocation-trace --rccl-trace "$@"
