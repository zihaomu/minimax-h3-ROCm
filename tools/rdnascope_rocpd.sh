#!/usr/bin/env bash
set -euo pipefail
sdk=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core
export LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The official converter creates an extra merged rocpd working DB in its cwd.
# Keep that artifact beside the RDNAScope run, never in the public source tree.
if [[ ${1:-} == convert ]]; then
  args=("$@")
  for ((i=0; i<${#args[@]}-1; i++)); do
    if [[ ${args[i]} == -d || ${args[i]} == --output-directory ]]; then
      cd "$(dirname "${args[i+1]}")"
      break
    fi
  done
fi
exec /usr/bin/python3 "$sdk/bin/rocpd" "$@"
