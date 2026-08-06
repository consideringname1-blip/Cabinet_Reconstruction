"""Subclass factory that keeps official iTACO source files untouched."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .moving_map import build_moving_map


def make_bundle_adjustment_adapter(
    official_class,
    *,
    chamfer_distance,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    wandb,
):
    class ProposalMovingMapBundleAdjustment(official_class):
        def __init__(
            self,
            *args,
            mode: str,
            valid_support: np.ndarray,
            static_threshold: float,
            moving_threshold: float,
            overlap_aggregation: str,
            logit_epsilon: float,
            **kwargs,
        ):
            self.moving_map_mode = mode
            self.static_threshold = static_threshold
            self.moving_threshold = moving_threshold
            self.overlap_aggregation = overlap_aggregation
            super().__init__(*args, **kwargs)
            if not self.valid:
                return
            support = torch.as_tensor(valid_support, dtype=torch.bool, device=self.device)
            if support.shape != self.old_part_segments_list.shape:
                raise ValueError(
                    f"support shape {tuple(support.shape)} != video shape "
                    f"{tuple(self.old_part_segments_list.shape)}"
                )
            self.sensor_support = support
            self.initial_occupation = self.moving_map_vec.detach().clone()
            if mode == "gate_no_minmax":
                initial = self.initial_occupation.clamp(logit_epsilon, 1.0 - logit_epsilon)
                del self.moving_map_vec
                self.moving_map_logits = torch.nn.Parameter(torch.logit(initial))
                params = (
                    self.camera_pose,
                    self.joint_axis,
                    self.joint_pos,
                    self.joint_state,
                    self.moving_map_logits,
                )
            else:
                params = (
                    self.camera_pose,
                    self.joint_axis,
                    self.joint_pos,
                    self.joint_state,
                    self.moving_map_vec,
                )
            self.optimizer = torch.optim.Adam([{"params": params}], lr=self.lr)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.steps
            )
            self.best_parameter = self._moving_parameter().detach().cpu().numpy()
            self.loss_history: list[float] = []

        def _moving_parameter(self) -> torch.Tensor:
            if self.moving_map_mode == "gate_no_minmax":
                return self.moving_map_logits
            return self.moving_map_vec

        def moving_map_result(
            self, parameter: torch.Tensor | np.ndarray | None = None
        ) -> dict[str, torch.Tensor]:
            if parameter is None:
                parameter = self._moving_parameter()
            if isinstance(parameter, np.ndarray):
                parameter = torch.as_tensor(
                    parameter, dtype=torch.float64, device=self.device
                )
            support = (
                torch.ones_like(self.sensor_support)
                if self.moving_map_mode == "official"
                else self.sensor_support
            )
            return build_moving_map(
                self.part_segments_full,
                parameter,
                support,
                self.moving_map_mode,
                static_threshold=self.static_threshold,
                moving_threshold=self.moving_threshold,
                overlap_aggregation=self.overlap_aggregation,
            )

        def score_for_consumer(
            self, consumer: str, *, best: bool = False
        ) -> torch.Tensor:
            if consumer not in {
                "train", "eval", "save", "visualize", "point_cloud", "adapter"
            }:
                raise ValueError(consumer)
            parameter = self.best_parameter if best else None
            return self.moving_map_result(parameter)["gated_score"]

        def _computable(self, frame: int) -> torch.Tensor:
            mask = self.obj_mask_bool[frame] & self.old_part_segments_list[frame]
            if self.moving_map_mode != "official":
                mask = mask & self.sensor_support[frame]
            return mask.reshape(-1)

        def chamfer_loss(self) -> torch.Tensor:
            if self.moving_map_mode == "official":
                return super().chamfer_loss()
            n, h, w, channels = self.xyz.shape
            moving_map = self.score_for_consumer("train")
            camera_extrinsics = torch.eye(
                4, dtype=torch.float64, device=self.device
            ).repeat(n, 1, 1)
            camera_extrinsics[:, :3, :3] = quaternion_to_matrix(self.camera_pose[:, :4])
            camera_extrinsics[:, :3, 3] = self.camera_pose[:, 4:]
            camera_xyz = (
                torch.matmul(
                    self.xyz.reshape(n, h * w, channels),
                    camera_extrinsics[:, :3, :3].permute(0, 2, 1),
                )
                + camera_extrinsics[:, :3, 3].reshape(n, 1, 3)
            )

            axis = F.normalize(self.joint_axis.reshape(1, 3))
            if self.joint_type == "revolute":
                rotations = axis_angle_to_matrix(axis.repeat(n, 1) * self.joint_state[:, None])
                translations = torch.matmul(
                    torch.eye(3, dtype=torch.float64, device=self.device).repeat(n, 1, 1)
                    - rotations,
                    self.joint_pos,
                )
            else:
                rotations = torch.eye(
                    3, dtype=torch.float64, device=self.device
                ).repeat(n, 1, 1)
                translations = axis.repeat(n, 1) * self.joint_state[:, None]
            joint_xyz = (
                torch.matmul(camera_xyz, rotations.permute(0, 2, 1))
                + translations[:, None, :]
            )
            static_loss = torch.zeros(n, device=self.device)
            dynamic_loss = torch.zeros_like(static_loss)
            flat_score = moving_map.reshape(n, h * w)
            for frame in range(n):
                computable = self._computable(frame)
                count = int(computable.sum().item())
                if count < 2:
                    continue
                sample_count = count // 2
                indices = torch.randint(0, count, (sample_count,), device=self.device)
                weights = flat_score[frame, computable][indices]
                static_dist, _ = chamfer_distance(
                    camera_xyz[frame, computable][indices][None],
                    self.surface_xyz[None],
                    batch_reduction=None,
                    point_reduction=None,
                    single_directional=True,
                )
                if self.loss_func == "hausdorff":
                    static_loss[frame] = torch.max((1 - weights) * static_dist[0])
                else:
                    static_loss[frame] = torch.mean((1 - weights) * static_dist[0])
            for frame in range(n):
                computable = self._computable(frame)
                count = int(computable.sum().item())
                if count < 2:
                    continue
                sample_count = count // 2
                indices = torch.randint(0, count, (sample_count,), device=self.device)
                weights = flat_score[frame, computable][indices]
                dynamic_dist, _ = chamfer_distance(
                    joint_xyz[frame, computable][indices][None],
                    self.surface_xyz[None],
                    batch_reduction=None,
                    point_reduction=None,
                    single_directional=True,
                )
                if self.loss_func == "hausdorff":
                    dynamic_loss[frame] = torch.max(weights * dynamic_dist[0])
                else:
                    dynamic_loss[frame] = torch.mean(weights * dynamic_dist[0])
            if self.train:
                wandb.log(
                    {
                        "Train/static chamfer loss": static_loss.detach().mean().item(),
                        "Train/dynamic chamfer loss": dynamic_loss.detach().mean().item(),
                        "Train/chamfer loss": (static_loss + dynamic_loss).detach().mean().item(),
                    }
                )
            return static_loss + dynamic_loss

        def optimize_adam(self, *unused) -> None:
            from tqdm import tqdm

            progress = tqdm(range(self.steps))
            for step in progress:
                self.current_step = step
                started = time.time()
                self.optimizer.zero_grad()
                loss = self.chamfer_loss().mean()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                value = float(loss.detach().cpu())
                self.loss_history.append(value)
                if value < self.best_loss:
                    self.best_loss = value
                    self.best_joint_axis = self.joint_axis.detach().cpu().numpy()
                    self.best_joint_pos = self.joint_pos.detach().cpu().numpy()
                    self.best_joint_state = self.joint_state.detach().cpu().numpy()
                    self.best_camera_poses = self.camera_pose.detach().cpu().numpy()
                    self.best_parameter = self._moving_parameter().detach().cpu().numpy()
                wandb.log(
                    {
                        f"Eval/{self.loss_func} error": value,
                        "time": time.time() - started,
                        "lr": self.lr_scheduler.get_last_lr()[-1],
                    },
                    step=step + 1,
                )
                progress.set_description(f"Loss: {value:.6f}")

    return ProposalMovingMapBundleAdjustment
