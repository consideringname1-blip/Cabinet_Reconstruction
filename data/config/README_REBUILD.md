# Current Server Environment Snapshot

Generated from the currently working container/server state. The goal is to rebuild the same conda environments and Docker/system package layer without letting solvers silently downgrade or upgrade dependencies.

## Directory Layout

- `conda/<env>/explicit.txt`: exact conda package URLs for this Linux platform. Use this as the primary conda lock file.
- `conda/<env>/pip-freeze.txt`: pip packages observed after the conda environment is active.
- `conda/<env>/environment.yml`: human-readable reference export. Do not use it as the first choice for exact rebuilds because it invokes the solver.
- `system/dpkg-installed.tsv`: all installed Debian packages with exact versions.
- `system/apt-manual-versioned.txt`: manually installed apt package roots with exact versions.
- `system/apt-installed-versioned.txt`: all apt packages with exact versions.
- `ros2/ros-humble-apt-versioned.txt`: installed ROS Humble apt packages with exact versions.
- `docker/Dockerfile.current` and `docker/docker-compose.current.yml`: Docker definition files that were present in this project.
- `docker/Dockerfile.shigure_core` and `docker/docker-compose.shigure_core.yml`: Docker definition files copied from the current branch's `code/reconstruction/shigure_core`.
- `runtime-env-summary.env`: important ROS2/CUDA/PYTHONPATH runtime variables. Full process environment is intentionally not captured; `docker/environment.txt` is a whitelist to avoid storing secrets.
- `source-git-manifest.tsv`: current branch Git index manifest with mode, object, size, and symlink target information. Git symlinks are recorded as mode `120000`.
- `source-symlinks.tsv`: runtime symlinks visible in this worktree, including tracked links and external model/checkpoint links.
- `source-submodules.txt`: recursive submodule commits for the current branch.
- `source-status.txt`: branch/status summary captured when this snapshot was generated.

## Rebuild Conda Environments

Prefer exact conda specs, then install pip packages without dependency resolution:

```bash
# example: rebuild server
bash /workspace/data/config/rebuild_conda_env.sh server
```

Equivalent manual commands:

```bash
conda create -n server --file /workspace/data/config/conda/server/explicit.txt
conda run -n server python -m pip install --no-deps -r /workspace/data/config/conda/server/pip-freeze.txt
```

Use `environment.yml` only for inspection or if the explicit spec cannot be used on the target platform.

## Rebuild Docker/System Layer

The current container has ROS2 Humble installed in `/opt/ros/humble` via apt/system packages, not as a conda env. Current runtime values include:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=10
export ROS_LOCALHOST_ONLY=0
```

For system packages, start from the same base image in `docker/Dockerfile.current`, restore apt sources, then install versioned packages. A safer first pass is:

```bash
apt-get update
xargs -a /workspace/data/config/system/apt-manual-versioned.txt apt-get install -y --no-install-recommends
```

If apt reports a version is unavailable, compare against `system/dpkg-installed.tsv` and the configured sources in `system/apt-sources.txt`. Avoid mixing unpinned newer repositories with these files; that is how ROS/Python packages get upgraded or downgraded unexpectedly.

## ROS2 / Shigurei Receive Workspace

The Shigurei receive workspace lives at:

```bash
/workspace/code/ros2/shigure_recv_ws
```

Use:

```bash
source /workspace/code/ros2/shigure_recv_ws/setup_env.sh
ros2 topic list -t
```

The generated `build/`, `install/`, and `log/` directories are intentionally ignored in that workspace.
