from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


def _is_vertex_line(line: str) -> bool:
    return line.startswith("v ") or line.startswith("v\t")


def _is_face_line(line: str) -> bool:
    return line.startswith("f ") or line.startswith("f\t")


def _resolve_obj_index(raw_index: int, total_count: int) -> int:
    if raw_index == 0:
        raise ValueError("OBJ indices are 1-based; got 0")
    if raw_index < 0:
        return total_count + raw_index + 1
    return raw_index


def _parse_face_vertex_indices(line: str, vertex_count: int) -> list[int]:
    indices: list[int] = []
    for token in line.split()[1:]:
        vertex_part = token.split("/", 1)[0]
        if not vertex_part:
            continue
        indices.append(_resolve_obj_index(int(vertex_part), vertex_count))
    return indices


def read_obj_geometry(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[list[float]] = []
    face_lines: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if _is_vertex_line(line):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif _is_face_line(line):
            face_lines.append(line)

    if not vertices:
        raise ValueError(f"No OBJ vertices found: {path}")

    vertex_count = len(vertices)
    faces = [_parse_face_vertex_indices(line, vertex_count) for line in face_lines]
    faces_zero_based = [[index - 1 for index in face] for face in faces if len(face) >= 3]
    return np.asarray(vertices, dtype=np.float32), faces_zero_based


class _DisjointSet:
    def __init__(self, count: int):
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _build_face_components(face_vertex_indices: list[list[int]]) -> list[list[int]]:
    if not face_vertex_indices:
        return []

    dsu = _DisjointSet(len(face_vertex_indices))
    first_face_for_vertex: dict[int, int] = {}
    for face_index, vertex_indices in enumerate(face_vertex_indices):
        for vertex_index in vertex_indices:
            previous_face = first_face_for_vertex.get(vertex_index)
            if previous_face is None:
                first_face_for_vertex[vertex_index] = face_index
            else:
                dsu.union(previous_face, face_index)

    components_by_root: dict[int, list[int]] = {}
    for face_index in range(len(face_vertex_indices)):
        root = dsu.find(face_index)
        components_by_root.setdefault(root, []).append(face_index)
    components = list(components_by_root.values())
    components.sort(key=len, reverse=True)
    return components


def _remap_face_line(line: str, vertex_mapping: dict[int, int], vertex_count: int) -> str:
    remapped_tokens: list[str] = []
    for token in line.split()[1:]:
        parts = token.split("/")
        old_vertex_index = _resolve_obj_index(int(parts[0]), vertex_count)
        parts[0] = str(vertex_mapping[old_vertex_index])
        remapped_tokens.append("/".join(parts))
    return "f " + " ".join(remapped_tokens)


def _write_filtered_obj(
    *,
    source_path: Path,
    target_path: Path,
    vertices: list[list[float]],
    face_lines: list[str],
    keep_face_indices: set[int],
) -> None:
    used_vertices = sorted(
        {
            vertex_index
            for face_index in keep_face_indices
            for vertex_index in _parse_face_vertex_indices(face_lines[face_index], len(vertices))
        }
    )
    vertex_mapping = {old_index: new_index for new_index, old_index in enumerate(used_vertices, start=1)}
    new_vertex_lines = [
        "v " + " ".join(f"{coord:.9g}" for coord in vertices[old_index - 1])
        for old_index in used_vertices
    ]

    output_lines: list[str] = []
    wrote_vertices = False
    face_index = 0
    for line in source_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if _is_vertex_line(line):
            if not wrote_vertices:
                output_lines.extend(new_vertex_lines)
                wrote_vertices = True
            continue
        if _is_face_line(line):
            if face_index in keep_face_indices:
                output_lines.append(_remap_face_line(line, vertex_mapping, len(vertices)))
            face_index += 1
            continue
        output_lines.append(line)

    if not wrote_vertices:
        output_lines = new_vertex_lines + output_lines

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def clean_obj_connected_components(
    source_path: Path,
    target_path: Path,
    *,
    min_face_ratio: float,
    min_faces: int,
) -> dict:
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    vertices: list[list[float]] = []
    face_lines: list[str] = []
    for line in lines:
        if _is_vertex_line(line):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif _is_face_line(line):
            face_lines.append(line)

    if not vertices or not face_lines:
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        return {
            "enabled": True,
            "source_mesh": source_path.name,
            "clean_mesh": target_path.name,
            "component_count": 0,
            "kept_component_count": 0,
            "removed_component_count": 0,
            "original_vertices": len(vertices),
            "clean_vertices": len(vertices),
            "original_faces": len(face_lines),
            "clean_faces": len(face_lines),
            "removed_faces": 0,
        }

    face_vertex_indices = [
        _parse_face_vertex_indices(line, len(vertices))
        for line in face_lines
    ]
    components = _build_face_components(face_vertex_indices)
    if not components:
        shutil.copy2(source_path, target_path)
        return {
            "enabled": True,
            "source_mesh": source_path.name,
            "clean_mesh": target_path.name,
            "component_count": 0,
            "kept_component_count": 0,
            "removed_component_count": 0,
            "original_vertices": len(vertices),
            "clean_vertices": len(vertices),
            "original_faces": len(face_lines),
            "clean_faces": len(face_lines),
            "removed_faces": 0,
        }

    largest_face_count = len(components[0])
    threshold = max(int(min_faces), int(round(largest_face_count * float(min_face_ratio))))
    keep_components = [
        component
        for component in components
        if component is components[0] or len(component) >= threshold
    ]
    keep_face_indices = {face_index for component in keep_components for face_index in component}
    removed_faces = len(face_lines) - len(keep_face_indices)
    if removed_faces > 0:
        _write_filtered_obj(
            source_path=source_path,
            target_path=target_path,
            vertices=vertices,
            face_lines=face_lines,
            keep_face_indices=keep_face_indices,
        )
        clean_vertices, _clean_faces = read_obj_geometry(target_path)
        clean_vertex_count = int(len(clean_vertices))
    else:
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        clean_vertex_count = len(vertices)

    return {
        "enabled": True,
        "source_mesh": source_path.name,
        "clean_mesh": target_path.name,
        "component_count": len(components),
        "kept_component_count": len(keep_components),
        "removed_component_count": len(components) - len(keep_components),
        "component_face_counts": [int(len(component)) for component in components[:12]],
        "min_face_ratio": float(min_face_ratio),
        "min_faces": int(min_faces),
        "face_threshold": int(threshold),
        "original_vertices": len(vertices),
        "clean_vertices": clean_vertex_count,
        "original_faces": len(face_lines),
        "clean_faces": len(keep_face_indices),
        "removed_faces": int(removed_faces),
    }
