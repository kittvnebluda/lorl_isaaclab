from typing import Iterable
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
