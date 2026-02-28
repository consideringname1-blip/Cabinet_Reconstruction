# task_worker.py
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional

from config import (
    IMESH_PY,
    INSTANTMESH_DIR,
    INSTANTMESH_CONFIG,
    INSTANTMESH_RUN_PY,
    OUTPUT_ROOT,
    INSTANTMESH_OUTPUT_MESHES,
    INSTANTMESH_OUTPUT_IMAGES,
    BLENDER_BIN,
    CONVERT_SCRIPT,
    BLENDER_FBX_DIR,
)

# 任务状态存储
tasks: Dict[str, Dict[str, Any]] = {}
task_queue = deque()
task_lock = threading.Lock()
_current_task_id: Optional[str] = None


def find_task_file(folder: Path, task_id: str, ext: str) -> Optional[Path]:
    """根据 task_id 在指定目录中查找最新的某类型文件。"""
    files = list(folder.glob(f"{task_id}*.{ext}"))
    return max(files, key=lambda p: p.stat().st_ctime) if files else None


def _process_tasks_loop():
    """后台线程循环，从队列中取任务执行 InstantMesh + Blender。"""
    global _current_task_id
    while True:
        task_id = None
        with task_lock:
            if task_queue:
                task_id = task_queue.pop()
                _current_task_id = task_id
                task_data = tasks[task_id]

        if not task_id:
            time.sleep(1)
            continue

        try:
            print(f"Processing task {task_id}")
            # 你原来的缩放公式，后面可以按需要再改
            scale = 1.5 + 4.5 * task_data["center_depth"]

            # ==== 调用 InstantMesh ====
            cmd = [
                IMESH_PY,
                str(INSTANTMESH_RUN_PY),
                str(INSTANTMESH_CONFIG),
                str(task_data["upload_path"]),
                "--output_path",
                str(OUTPUT_ROOT),
                "--save_video",
                "--export_texmap",
                # "--scale", str(scale),
                # "--real_scale", str(task_data["center_depth"]),
            ]
            print(">>> InstantMesh CMD:", " ".join(cmd))

            # 关键：给子进程单独准备 env，把 imesh 的 bin 加到 PATH 前面
            env = os.environ.copy()
            env["PATH"] = "/opt/miniconda/envs/imesh/bin:" + env.get("PATH", "")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(INSTANTMESH_DIR),  # 在 InstantMesh 目录下跑
                env=env,                   # 使用带有 ninja 的 PATH
            )
            print(f"Command output: {result.stdout}")

            # 查找 InstantMesh 输出
            mesh_path = find_task_file(INSTANTMESH_OUTPUT_MESHES, task_id, "obj")
            mtl_path = find_task_file(INSTANTMESH_OUTPUT_MESHES, task_id, "mtl")
            image_path = find_task_file(INSTANTMESH_OUTPUT_IMAGES, task_id, "png")

            if not mesh_path or not mtl_path or not image_path:
                raise RuntimeError(f"Output files not found for task {task_id}")

            # ==== 调用 Blender 转 FBX ====
            fbx_path = BLENDER_FBX_DIR / (mesh_path.stem + ".fbx")
            fbx_generated = False
            try:
                result = subprocess.run(
                    [
                        BLENDER_BIN,
                        "--background",
                        "--python",
                        str(CONVERT_SCRIPT),
                        "--",
                        str(mesh_path),
                        str(image_path),
                        str(fbx_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"FBX生成成功: {result.stdout}")
                if os.path.exists(fbx_path):
                    fbx_generated = True
                else:
                    print("FBX文件未生成。")
            except subprocess.CalledProcessError as e:
                print(f"FBX生成失败: {e.stderr}")

            # 写回任务状态
            with task_lock:
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["result"] = {
                    "mesh_path": mesh_path,
                    "mtl_path": mtl_path,
                    "image_path": image_path,
                    "fbx_path": fbx_path if fbx_generated else None,
                }
            print(f"Task {task_id} completed")

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or str(e)
            print(f"Task {task_id} failed: {error_msg}")
            with task_lock:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = error_msg
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            with task_lock:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = str(e)
        finally:
            with task_lock:
                _current_task_id = None

        time.sleep(1)


def start_worker():
    """在 Flask 启动时调用，启动后台线程。"""
    t = threading.Thread(target=_process_tasks_loop, daemon=True)
    t.start()
    return t


def create_task(task_id, upload_path: Path, center_depth: float,
                device_pose: Optional[dict], object_pose: Optional[dict]):
    """由 Flask 接口创建一个新任务并入队。"""
    with task_lock:
        tasks[task_id] = {
            "status": "pending",
            "upload_path": upload_path,
            "center_depth": center_depth,
            "created_at": time.time(),
            "device_pose": device_pose,
            "object_pose": object_pose,
        }
        task_queue.append(task_id)


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """根据 task_id 获取任务信息。"""
    with task_lock:
        return tasks.get(task_id)


def get_current_task_id() -> Optional[str]:
    with task_lock:
        return _current_task_id


def get_queue_snapshot():
    """返回当前队列的一个快照（list），供接口查询排队位置用。"""
    with task_lock:
        return list(task_queue)
