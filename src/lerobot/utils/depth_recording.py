#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for recording depth maps alongside RGB during ``lerobot-record``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras.camera import Camera

if TYPE_CHECKING:
    from lerobot.robots import Robot


def camera_supports_depth(camera: Camera) -> bool:
    """Return whether this camera can provide depth frames."""
    if getattr(camera, "use_depth", False):
        return True
    config = getattr(camera, "config", None)
    return bool(getattr(config, "use_depth", False))


def enable_depth_on_cameras(robot: "Robot") -> None:
    """Turn on ``use_depth`` in camera configs before ``robot.connect()``."""
    if not hasattr(robot, "cameras"):
        return
    for camera in robot.cameras.values():
        config = getattr(camera, "config", None)
        if config is not None and hasattr(config, "use_depth"):
            config.use_depth = True
        if hasattr(camera, "use_depth"):
            camera.use_depth = True


def get_depth_hw_features(robot: "Robot") -> dict[str, tuple[int, int]]:
    """Hardware depth feature specs keyed as ``{camera_name}_depth``."""
    if not hasattr(robot, "cameras"):
        return {}
    depth_features: dict[str, tuple[int, int]] = {}
    for cam_key, camera in robot.cameras.items():
        if not camera_supports_depth(camera):
            continue
        config = getattr(camera, "config", None)
        height = getattr(config, "height", None) or getattr(camera, "capture_height", None)
        width = getattr(config, "width", None) or getattr(camera, "capture_width", None)
        if height is None or width is None:
            continue
        depth_features[f"{cam_key}_depth"] = (int(height), int(width))
    return depth_features


def _depth_max_age_ms(camera: Camera) -> int:
    config = getattr(camera, "config", None)
    cam_fps = getattr(config, "fps", None) or getattr(camera, "fps", None) or 15
    return max(int(4 * 1000 / cam_fps), 1000)


def augment_observation_with_depth(robot: "Robot", observation: dict[str, Any]) -> dict[str, Any]:
    """Append ``{cam}_depth`` arrays to a robot observation dict."""
    if not hasattr(robot, "cameras"):
        return observation
    augmented = dict(observation)
    for cam_key, camera in robot.cameras.items():
        if not camera_supports_depth(camera):
            continue
        max_age_ms = _depth_max_age_ms(camera)
        if hasattr(camera, "read_depth_latest"):
            try:
                augmented[f"{cam_key}_depth"] = camera.read_depth_latest(max_age_ms=max_age_ms)
                continue
            except TimeoutError:
                pass
        if not hasattr(camera, "read_depth"):
            raise AttributeError(
                f"Camera '{cam_key}' is configured for depth but has no read_depth() method."
            )
        augmented[f"{cam_key}_depth"] = camera.read_depth()
    return augmented


def depth_to_rgb_uint8(depth: np.ndarray) -> np.ndarray:
    """Convert a depth map (uint16 mm or float) to uint8 RGB for PNG / video encoding."""
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D, got shape {depth.shape}")

    if depth.dtype == np.uint16:
        gray = (depth.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
    elif depth.dtype == np.uint8:
        gray = depth
    else:
        gray = (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)

    return np.stack([gray, gray, gray], axis=-1)


def dataset_has_depth_features(features: dict[str, dict]) -> bool:
    return any(ft.get("dtype") == "depth" for ft in features.values())


def observation_features_with_depth(robot: "Robot", save_depth: bool) -> dict[str, type | tuple]:
    """Build observation feature dict for dataset schema creation."""
    features = dict(robot.observation_features)
    if save_depth:
        features.update(get_depth_hw_features(robot))
    return features
