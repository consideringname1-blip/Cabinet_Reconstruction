from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _bootstrap import CODE_ROOT
from config import ENV_CONFIG_ROOT


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        check=True,
    )


def list_conda_envs() -> list[tuple[str, str]]:
    result = _run_command(["conda", "env", "list", "--json"])
    data = json.loads(result.stdout)

    env_paths: list[str] = data.get("envs", [])
    default_prefix: str = data.get("default_prefix", "")

    envs: list[tuple[str, str]] = []
    for env_path in env_paths:
        if env_path == default_prefix:
            env_name = "base"
        else:
            env_name = Path(env_path).name
        envs.append((env_name, env_path))
    return envs


def make_unique_file_prefixes(envs: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    counts: dict[str, int] = {}
    result: list[tuple[str, str, str]] = []

    for env_name, env_path in envs:
        counts[env_name] = counts.get(env_name, 0) + 1
        idx = counts[env_name]
        file_prefix = env_name if idx == 1 else f"{env_name}_{idx}"
        result.append((env_name, env_path, file_prefix))
    return result


def export_conda_and_pip(
    env_path: str,
    file_prefix: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    conda_file = output_dir / f"{file_prefix}_conda_environment.yml"
    pip_file = output_dir / f"{file_prefix}_pip_requirements.txt"

    with conda_file.open("w", encoding="utf-8", newline="\n") as f:
        subprocess.run(
            ["conda", "env", "export", "-p", env_path, "--no-builds"],
            stdout=f,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    with pip_file.open("w", encoding="utf-8", newline="\n") as f:
        subprocess.run(
            ["conda", "run", "-p", env_path, "python", "-m", "pip", "freeze"],
            stdout=f,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    return conda_file, pip_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all conda/pip environment files."
    )
    parser.parse_args()

    envs = make_unique_file_prefixes(list_conda_envs())
    if not envs:
        print("No conda environments found.")
        return

    for env_name, env_path, file_prefix in envs:
        conda_file, pip_file = export_conda_and_pip(
            env_path=env_path,
            file_prefix=file_prefix,
            output_dir=ENV_CONFIG_ROOT,
        )
        print(f"[{env_name}] Conda environment exported: {conda_file}")
        print(f"[{env_name}] Pip requirements exported: {pip_file}")


if __name__ == "__main__":
    main()
