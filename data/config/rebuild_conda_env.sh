#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <env-name>" >&2
  echo "Available envs:" >&2
  find /workspace/data/config/conda -mindepth 1 -maxdepth 1 -type d -printf '  %f
' | sort >&2
  exit 2
fi

env_name="$1"
env_dir="/workspace/data/config/conda/$env_name"
explicit_file="$env_dir/explicit.txt"
pip_file="$env_dir/pip-freeze.txt"

if [ ! -f "$explicit_file" ]; then
  echo "Missing explicit conda spec: $explicit_file" >&2
  exit 1
fi

conda create -n "$env_name" --file "$explicit_file"

if [ -s "$pip_file" ]; then
  conda run -n "$env_name" python -m pip install --no-deps -r "$pip_file"
fi
