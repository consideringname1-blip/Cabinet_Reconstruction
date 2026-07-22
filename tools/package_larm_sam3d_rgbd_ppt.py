import json
import shutil
import textwrap
import zipfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/workspace_whz")
PKG = ROOT / "data/output/report_packages/larm_sam3d_rgbd_ppt_package"
ASSETS = PKG / "assets"
MODELS = PKG / "models"
REPORTS = PKG / "reports"

SRC = {
    "source_color": ROOT / "data/upload/larm_captures/20260622_081031_636398Z/color.png",
    "door_mask": ROOT / "data/output/geometric_joint_estimate_sam3door/door_mask.png",
    "base_mask": ROOT / "data/output/geometric_joint_estimate_sam3door/base_mask.png",
    "sam3_door_overlay": ROOT / "data/output/geometric_joint_estimate/sam3_prompt_candidates/door_overlay.png",
    "sam3_base_overlay": ROOT / "data/output/geometric_joint_estimate_sam3door/sam3d_parts_input/qpos1_base_sam3mask_overlay.png",
    "wrong_existing_projection_overlay": ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/prismatic_existing_fitted_visible/qpos1_existing_fitted_open_visible_projection_overlay.png",
    "wrong_silhouette_projection_overlay": ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/prismatic_silhouette_aligned/qpos1_open_mesh_projection_overlay.png",
    "final_combined_overlay": ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/combined_rgbd_similarity_centered_overlay.png",
    "final_base_overlay": ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/base_rgbd_similarity_centered_overlay.png",
    "final_drawer_overlay": ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/drawer_rgbd_similarity_centered_overlay.png",
    "final_combined_glb": ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/cabinet_drawer_rgbd_similarity_centered_open.glb",
    "final_base_glb": ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/base_rgbd_similarity_centered.glb",
    "final_drawer_glb": ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/drawer_rgbd_similarity_centered.glb",
    "final_report": ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/rgbd_similarity_centered_report.json",
    "pre_center_report": ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit/rgbd_similarity_fit_report.json",
}


def copy_if_exists(src: Path, dst_dir: Path) -> str | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return str(dst.relative_to(PKG))


def read_final_metrics() -> dict:
    report = json.loads(SRC["final_report"].read_text(encoding="utf-8"))
    out = {}
    for name, part in report["parts"].items():
        out[name] = {
            "target_bbox_xyxy": part["target_bbox_xyxy"],
            "fitted_bbox_xyxy": part["bbox_after_xyxy"],
            "target_center_px": part["target_bbox_center_px"],
            "fitted_center_px": part["bbox_center_after_px"],
            "scale_uniform": part["scale_uniform"],
            "xy_translation_delta_m": part["xy_translation_delta_m"],
        }
    return out


def make_contact_sheet(image_paths: list[Path], labels: list[str], output: Path) -> None:
    thumbs = []
    for path, label in zip(image_paths, labels):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (640, 34), (255, 255, 255), -1)
        cv2.putText(img, label, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (30, 30, 30), 2, cv2.LINE_AA)
        thumbs.append(img)
    if not thumbs:
        return
    while len(thumbs) % 2:
        thumbs.append(np.zeros_like(thumbs[0]) + 255)
    rows = []
    for i in range(0, len(thumbs), 2):
        rows.append(np.hstack([thumbs[i], thumbs[i + 1]]))
    sheet = np.vstack(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def main() -> None:
    if PKG.exists():
        shutil.rmtree(PKG)
    ASSETS.mkdir(parents=True)
    MODELS.mkdir()
    REPORTS.mkdir()

    manifest = {"package": str(PKG), "assets": {}, "models": {}, "reports": {}}
    for key, path in SRC.items():
        if key.endswith("_glb"):
            rel = copy_if_exists(path, MODELS)
            if rel:
                manifest["models"][key] = rel
        elif key.endswith("_report"):
            rel = copy_if_exists(path, REPORTS)
            if rel:
                manifest["reports"][key] = rel
        else:
            rel = copy_if_exists(path, ASSETS)
            if rel:
                manifest["assets"][key] = rel

    contact = ASSETS / "contact_sheet_pipeline.png"
    make_contact_sheet(
        [
            SRC["source_color"],
            SRC["sam3_door_overlay"],
            SRC["wrong_existing_projection_overlay"],
            SRC["final_combined_overlay"],
        ],
        [
            "Input RGB-D capture",
            "SAM mask prompt result",
            "Failed/diagnostic alignment",
            "Final RGB-D similarity fit",
        ],
        contact,
    )
    manifest["assets"]["contact_sheet_pipeline"] = str(contact.relative_to(PKG))

    metrics = read_final_metrics()
    summary_json = {
        "title": "From LARM failure to RGB-D constrained SAM3DObject fitting",
        "final_method": "Fit raw SAM3DObject base/drawer meshes to fixed-camera SAM masks and aligned depth using 7DoF similarity only: rotation, uniform scale, translation.",
        "important_negative_results": [
            "LARM direct input produced unreliable part masks for the white drawer/cabinet.",
            "AABB/PCA fitting required anisotropic scale and changed shape, so it was rejected.",
            "Projection-only and silhouette-only alignment could place parts in the image but failed 3D consistency.",
        ],
        "final_metrics": metrics,
        "key_outputs": manifest,
    }
    (PKG / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    write_text(
        PKG / "README.md",
        f"""
        # LARM / SAM3D / RGB-D Reconstruction PPT Package

        这个目录是一份汇报材料包，整理了从 LARM 失败、SAM3DObject 分件、错误对齐尝试，到最终 RGB-D/mask constrained similarity fitting 的过程。

        ## 推荐使用的最终结果

        - `models/cabinet_drawer_rgbd_similarity_centered_open.glb`
        - `assets/combined_rgbd_similarity_centered_overlay.png`
        - `reports/rgbd_similarity_centered_report.json`

        ## 最终方法一句话

        固定 HoloLens 相机内参，使用 SAM mask 和 aligned depth 生成每个 part 的观测约束；对 SAM3DObject 生成的 base/drawer mesh 只优化 3D rotation、uniform scale、camera-frame translation，不做非等比例缩放，不改拓扑。

        ## 关键指标

        - Drawer target bbox: `{metrics['drawer']['target_bbox_xyxy']}`
        - Drawer fitted bbox: `{metrics['drawer']['fitted_bbox_xyxy']}`
        - Base target center: `{metrics['base']['target_center_px']}`
        - Base fitted center: `{metrics['base']['fitted_center_px']}`

        ## 目录

        - `assets/`: PPT 可直接插入的 PNG 图
        - `models/`: 可检查的 GLB 模型
        - `reports/`: JSON fit report
        - `slides_outline.md`: 建议 PPT 页结构和每页讲稿
        - `prompts.md`: 给 PPT/论文/继续实验用的提示词
        - `speaker_notes.md`: 口头汇报稿
        """,
    )

    write_text(
        PKG / "slides_outline.md",
        f"""
        # PPT Outline: RGB-D Constrained Articulated Cabinet Reconstruction

        ## Slide 1. Title
        **RGB-D Constrained Reconstruction of a Drawer-like Cabinet from HoloLens Captures**

        Subtitle: Why LARM failed, and how SAM3DObject + fixed-camera RGB-D fitting recovered usable part geometry.

        Suggested visual: `assets/contact_sheet_pipeline.png`

        ## Slide 2. Problem Setup
        - Input: six LARM-style captures of a white cabinet/drawer.
        - Object moved between states; camera was not the moving baseline.
        - Goal: recover two part meshes and a prismatic joint suitable for AR/HoloLens deployment.

        Suggested visual: `assets/color.png`

        ## Slide 3. Why Direct LARM Failed
        - White cabinet on light background caused weak boundaries.
        - LARM part mask became uninformative or blurry.
        - SfM assumptions were not ideal because object motion dominated the captures.

        Suggested visual: previous LARM outputs or `assets/door_overlay.png`

        ## Slide 4. SAM Mask Improved Segmentation
        - SAM mask isolated the drawer and cabinet body better than LARM internal part mask.
        - This provided separate inputs for SAM3DObject part reconstruction.

        Suggested visuals:
        - `assets/door_overlay.png`
        - `assets/qpos1_base_sam3mask_overlay.png`

        ## Slide 5. Negative Result: Geometry-only Fits Were Not Enough
        - AABB/PCA alignment separated parts and could flip orientations.
        - Some fits required anisotropic scaling, visibly changing shape.
        - Closed child-frame meshes were misleading as visual inspection assets.

        Suggested visual: `assets/qpos1_existing_fitted_open_visible_projection_overlay.png`

        ## Slide 6. Key Insight
        **If camera intrinsics, aligned depth, and masks are known, each generated part should be fitted by a 7DoF similarity transform only.**

        The valid transform is:

        ```text
        X_camera = s * R * X_sam3d + t
        ```

        where `s` is uniform. No anisotropic scale is allowed.

        ## Slide 7. Final Method
        1. Generate SAM masks for base and drawer.
        2. Backproject mask pixels with aligned depth to camera-frame point clouds.
        3. Load raw SAM3DObject meshes with vertex/color geometry intact.
        4. Enumerate PCA axis candidates for robust orientation initialization.
        5. Optimize similarity transform with ICP-like nearest-neighbor fitting.
        6. Snap final projected bbox center to the mask center using camera-model x/y translation only.

        Suggested visual: `assets/contact_sheet_pipeline.png`

        ## Slide 8. Quantitative Fit Check
        - Drawer target bbox: `{metrics['drawer']['target_bbox_xyxy']}`
        - Drawer fitted bbox: `{metrics['drawer']['fitted_bbox_xyxy']}`
        - Base target center: `{metrics['base']['target_center_px']}`
        - Base fitted center: `{metrics['base']['fitted_center_px']}`
        - Constraint: rotation + uniform scale + translation only.

        Suggested visual: `assets/combined_rgbd_similarity_centered_overlay.png`

        ## Slide 9. Final 3D Assets
        - Combined open-pose GLB: `models/cabinet_drawer_rgbd_similarity_centered_open.glb`
        - Base GLB: `models/base_rgbd_similarity_centered.glb`
        - Drawer GLB: `models/drawer_rgbd_similarity_centered.glb`

        Suggested visual: screenshot from GLB viewer or Blender.

        ## Slide 10. Remaining Work
        - Estimate prismatic slide axis from qpos0/qpos1 after both states are fitted.
        - Export deployment-ready FBX/Unity prefab with vertex-color shader.
        - Validate scale and transform in HoloLens coordinate system.
        - Replace heuristic center snap with differentiable silhouette/depth rendering if needed.

        ## Slide 11. Takeaways
        - LARM is brittle when internal part segmentation fails.
        - SAM3DObject gives useful part geometry, but canonical pose is arbitrary.
        - RGB-D/mask constraints are the right bridge between image-space segmentation and metric 3D placement.
        - Avoid anisotropic scale: it hides fitting errors by deforming geometry.
        """,
    )

    write_text(
        PKG / "speaker_notes.md",
        f"""
        # Speaker Notes

        今天这组实验的核心问题是：我们有一个白色抽屉式柜子，想从少量 HoloLens RGB-D capture 中恢复可动模型。最初尝试直接走 LARM，但由于柜体和背景都偏白，边界弱，LARM 生成的 part mask 基本不可靠，后续 mesh 和 URDF 都随之失败。

        第二步我们把问题拆开：先用 SAM 得到 base 和 drawer 的 mask，再用 SAM3DObject 分别生成两个 part mesh。SAM3DObject 的视觉效果看起来不错，但它输出的 mesh 在自己的 canonical pose 中，不能直接认为和输入相机坐标一致。

        中间有几条失败路线很重要。AABB/PCA fitting 可以把 mesh 大概放到点云附近，但它会诱导非等比例缩放，形状会变；projection-only 或 silhouette-only fitting 可以让 2D 看起来接近，但 3D 深度和朝向仍然可能错。另一个坑是 closed child-frame mesh 不能拿来当 open 状态可视检查，否则会误以为 drawer 消失或埋进柜子。

        最终可用的方法是固定相机模型：用 HoloLens 内参和 aligned depth，把 SAM mask 反投影成每个 part 的相机坐标点云；然后对 SAM3DObject 的 raw mesh 只优化 7DoF similarity transform，也就是 rotation、uniform scale 和 translation。这个约束非常关键，因为如果需要非等比例缩放才能对上，通常说明坐标系或 pose fitting 错了，而不是模型真的应该变形。

        最终结果里，drawer 的目标 bbox 是 `{metrics['drawer']['target_bbox_xyxy']}`，拟合 bbox 是 `{metrics['drawer']['fitted_bbox_xyxy']}`；base 的目标中心是 `{metrics['base']['target_center_px']}`，拟合中心是 `{metrics['base']['fitted_center_px']}`。这说明两个 part 已经能在输入图像中对齐到合理位置，同时保持统一尺度，不再靠变形糊弄过去。

        下一步是把 qpos0 和 qpos1 都做同样的 RGB-D similarity fitting，再根据两个状态的 drawer 位姿差估计 prismatic slide axis，最后导出 Unity/HoloLens 可用的 FBX 或 prefab。
        """,
    )

    write_text(
        PKG / "prompts.md",
        f"""
        # Prompts

        ## 1. 生成中文汇报 PPT 的提示词

        请根据以下材料生成一份 10-12 页的中文技术汇报 PPT。主题是“从 LARM 失败到 RGB-D 约束的 SAM3DObject 可动柜体重建”。要求风格清晰、工程研究导向，不要营销风。每页包括标题、3-5 个要点、建议插图、讲稿备注。

        背景：输入是 HoloLens RGB-D 捕获的白色抽屉式柜子，目标是恢复 base 和 drawer 两个 part mesh，并最终估计 prismatic joint 用于 HoloLens/Unity 部署。

        过程：
        1. 直接使用 LARM 时，由于白色柜体边缘弱、背景相近，内部 part mask 不可靠，输出 mesh/URDF 失败。
        2. 使用 SAM 得到 drawer/base mask，分开送入 SAM3DObject 生成 part mesh，分割明显改善。
        3. 失败尝试包括 AABB/PCA fitting、projection-only fitting、silhouette-only fitting。这些方法导致 part 分离、朝向错误，甚至需要非等比例缩放改形状。
        4. 最终方法：固定相机内参，使用 aligned depth + SAM mask 反投影成相机坐标点云；对 SAM3DObject raw mesh 只优化 7DoF similarity transform：3D rotation、uniform scale、translation，不允许 anisotropic scale。
        5. 最终检查：drawer target bbox `{metrics['drawer']['target_bbox_xyxy']}`，fitted bbox `{metrics['drawer']['fitted_bbox_xyxy']}`；base target center `{metrics['base']['target_center_px']}`，fitted center `{metrics['base']['fitted_center_px']}`。

        可用图片：
        - `assets/contact_sheet_pipeline.png`
        - `assets/door_overlay.png`
        - `assets/qpos1_base_sam3mask_overlay.png`
        - `assets/combined_rgbd_similarity_centered_overlay.png`

        可用模型：
        - `models/cabinet_drawer_rgbd_similarity_centered_open.glb`

        请突出结论：SAM3DObject 生成的 mesh canonical pose 任意，不能直接当相机坐标；正确做法是用 RGB-D/mask 约束做 rigid+uniform-scale fitting。避免非等比例缩放，因为它会掩盖 pose fitting 错误。

        ## 2. 生成英文论文式摘要的提示词

        Write a concise research-style abstract for a project that reconstructs an articulated drawer-like cabinet from HoloLens RGB-D captures. The pipeline first fails with direct LARM reconstruction due to weak white-object boundaries and unreliable part masks. It then uses SAM masks and SAM3DObject to reconstruct separate base and drawer meshes. Since SAM3DObject outputs meshes in arbitrary canonical poses, the final alignment fits each raw mesh to fixed-camera aligned depth and segmentation masks using only a 7DoF similarity transform: rotation, uniform scale, and translation. Mention that anisotropic scaling was rejected because it deforms geometry and hides pose errors. Include final projection alignment metrics: drawer target bbox `{metrics['drawer']['target_bbox_xyxy']}`, fitted bbox `{metrics['drawer']['fitted_bbox_xyxy']}`.

        ## 3. 继续实验的 Codex 提示词

        在 `/workspace_whz` 中继续当前项目。不要再使用 `prismatic_aabb_aligned`、`prismatic_similarity_aligned`、`prismatic_projection_aligned`、`prismatic_silhouette_aligned` 作为最终结果；这些是失败/诊断路线。当前最可信结果是：

        - `data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/cabinet_drawer_rgbd_similarity_centered_open.glb`
        - `data/output/geometric_joint_estimate_sam3door_sam3d_fitted/rgbd_similarity_fit_centered/rgbd_similarity_centered_report.json`

        请基于 qpos0 和 qpos1 分别做同样的 RGB-D + SAM mask + SAM3DObject raw mesh similarity fitting。约束只能是 rotation + uniform scale + translation，不允许 anisotropic scaling。然后用两个状态的 drawer 位姿差估计 prismatic slide axis，并导出 Unity/HoloLens 可用的 FBX/GLB 和 joint metadata。所有可视检查都必须使用 open-pose mesh，不要把 closed child-frame mesh 当作可视检查。

        ## 4. 生成流程图图片的提示词

        Create a clean technical diagram for a research presentation. Show a pipeline from "HoloLens RGB-D capture" to "SAM masks" to "SAM3DObject part meshes" to "RGB-D constrained 7DoF similarity fitting" to "Aligned base/drawer mesh" to "Prismatic joint estimation". Use a restrained engineering style, white background, thin lines, blue/green accents, no decorative blobs. Include a warning callout: "No anisotropic scaling: rotation + uniform scale + translation only." The diagram should be readable on a 16:9 presentation slide.

        ## 5. 一页总结提示词

        请把以下工作总结成一页中文 PPT：LARM 对白色抽屉柜 part mask 失败；SAM mask + SAM3DObject 得到较好的分件 mesh；AABB/PCA 和 projection-only 对齐会造成朝向错误或非等比例缩放；最终使用固定相机内参、aligned depth 和 mask，将每个 raw SAM3DObject mesh 通过 7DoF similarity transform 对齐到相机坐标。输出最终模型 `cabinet_drawer_rgbd_similarity_centered_open.glb`，drawer bbox 从目标 `[140,143,308,293]` 对齐到 `[141.0,145.8,307.7,295.7]`。
        """,
    )

    write_text(
        PKG / "marp_deck.md",
        f"""
        ---
        marp: true
        theme: default
        paginate: true
        size: 16:9
        ---

        # RGB-D Constrained Reconstruction
        ## From LARM failure to SAM3DObject part fitting

        ![](assets/contact_sheet_pipeline.png)

        ---

        # Problem

        - White drawer-like cabinet, weak object boundaries
        - Six LARM-style HoloLens captures
        - Need base/drawer meshes and a prismatic joint for AR deployment

        ---

        # Direct LARM Failure

        - Part mask was unreliable
        - Mesh/URDF became unstable
        - Object motion made SfM-style assumptions fragile

        ---

        # Segmentation Recovery

        - SAM masks gave usable base and drawer regions
        - SAM3DObject reconstructed separate part meshes
        - Remaining issue: canonical mesh pose is arbitrary

        ![](assets/door_overlay.png)

        ---

        # Failed Alignment Routes

        - AABB/PCA fitting separated parts and changed shape
        - Projection-only fitting looked plausible but was 3D-inconsistent
        - Closed child-frame meshes were misleading for visual inspection

        ---

        # Final Method

        ```text
        X_camera = s * R * X_sam3d + t
        ```

        - Fixed camera intrinsics
        - SAM mask + aligned depth backprojection
        - Rotation + uniform scale + translation only
        - No anisotropic scaling

        ---

        # Final Fit Check

        - Drawer target bbox: `{metrics['drawer']['target_bbox_xyxy']}`
        - Drawer fitted bbox: `{metrics['drawer']['fitted_bbox_xyxy']}`
        - Base target center: `{metrics['base']['target_center_px']}`
        - Base fitted center: `{metrics['base']['fitted_center_px']}`

        ![](assets/combined_rgbd_similarity_centered_overlay.png)

        ---

        # Outputs

        - `models/cabinet_drawer_rgbd_similarity_centered_open.glb`
        - `reports/rgbd_similarity_centered_report.json`
        - `assets/combined_rgbd_similarity_centered_overlay.png`

        ---

        # Next Steps

        - Fit qpos0 and qpos1 with the same method
        - Estimate prismatic slide axis from pose delta
        - Export Unity/HoloLens-ready FBX or prefab
        - Validate metric scale in headset coordinates
        """,
    )

    manifest_path = PKG / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    zip_path = PKG.parent / "larm_sam3d_rgbd_ppt_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in PKG.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(PKG.parent))

    print(json.dumps({"package_dir": str(PKG), "zip": str(zip_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
