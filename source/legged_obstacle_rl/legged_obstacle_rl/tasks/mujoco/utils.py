from typing import Iterable

import mujoco
import numpy as np


def normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize an array along the last dimension."""
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)

    xyz = quat[:, 1:]
    t = np.cross(xyz, vec, axis=-1) * 2
    return (vec - quat[:, 0:1] * t + np.cross(xyz, t, axis=-1)).reshape(shape)


def map_indexes(*, target: Iterable, source: Iterable) -> list:
    """Return the indices of target names in the source list.

    Args:
        target (Iterable): Ordered list of names defining the desired output order.
        source (Iterable): Ordered list of names to search within.

    Returns:
        list[int]: Indices into `source` such that
            ``[source[i] for i in result] == target``.

    Raises:
        ValueError: (implicit) If a name in `target` is not found in `source`,
            the name is silently skipped and the result will be shorter than expected.

    Example:
        >>> find_indices(target=["b", "c", "a"], source=["a", "b", "c"])
        [1, 2, 0]
    """
    res = []
    for name in target:
        for i, sname in enumerate(source):
            if sname == name:
                res.append(i)
    return res


def mujoco_height_scan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    xv: np.ndarray,
    yv: np.ndarray,
    offset_z: float,
    base_offset: float,
) -> np.ndarray:
    """Cast downward rays from a body-aligned grid and return relative terrain heights.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        body_id: Index of the base body (used for position and yaw).
        xv: Meshgrid X coordinates of scan points in local body frame (m).
        yv: Meshgrid Y coordinates of scan points in local body frame (m).
        offset_z: Ray origin height above ground (m).
        base_offset: Vertical offset subtracted from measured height (m).
            Should match the IsaacLab sensor height offset used during training.

    Returns:
        Array of terrain height values clipped to [-1, 1], shape ``(N,)``
        where N = xv.size.
    """
    body_pos = data.xpos[body_id]
    body_mat = data.xmat[body_id].reshape(3, 3)

    yaw = np.arctan2(body_mat[1, 0], body_mat[0, 0])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    yaw_mat = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])

    local_origins = np.stack(
        [xv.flatten(), yv.flatten(), np.full(xv.size, offset_z)],
        axis=-1,
    )
    world_origins = body_pos + local_origins @ yaw_mat.T
    world_direction = np.array([0.0, 0.0, -1.0])

    distances = []
    geom_id = np.zeros(1, dtype=np.int32)
    for origin in world_origins:
        dist = mujoco.mj_ray(
            model,
            data,
            pnt=origin,
            vec=world_direction,
            geomgroup=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
            flg_static=1,
            bodyexclude=-1,
            geomid=geom_id,
        )
        ground = origin[2] - dist
        distances.append(np.clip(body_pos[2] + ground - base_offset, -1.0, 1.0))

    return np.array(distances, dtype=np.float32)
