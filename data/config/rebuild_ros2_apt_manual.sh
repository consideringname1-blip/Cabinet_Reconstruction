#!/usr/bin/env bash
set -euo pipefail

list_file="/workspace/data/config/system/apt-manual-versioned.txt"
if [ ! -f "$list_file" ]; then
  echo "Missing apt list: $list_file" >&2
  exit 1
fi

apt-get update
xargs -a "$list_file" apt-get install -y --no-install-recommends
