from importlib.resources import files

import numpy as np

from legged_obstacle_rl.tasks.mujoco.rough.flat_env import Go1RoughFlatEnv

_HFIELD_HALF_X = 5.0  # metres
_HFIELD_HALF_Y = 5.0


class Go1RoughHFieldEnv(Go1RoughFlatEnv):
    """Go1RoughEnv on a randomised MuJoCo heightfield.

    Args:
        max_roughness: Maximum terrain height as a fraction of hfield elevation [0, 1].
            E.g. 0.5 -> max bump = 0.15 m with the default 0.3 m elevation.
        flat_pad_radius: Half-side of the flat starting square around the origin (m).
    """

    def __init__(self, max_roughness: float = 0.5, flat_pad_radius: float = 0.75, **kwargs):
        xml_file = str(files("legged_obstacle_rl").joinpath("tasks/mujoco/unitree_go1/scene_rough.xml"))
        super().__init__(xml_file=xml_file, **kwargs)
        self.max_roughness = max_roughness
        self.flat_pad_radius = flat_pad_radius

    def reset_model(self):
        self._randomise_terrain()
        return super().reset_model()

    def _randomise_terrain(self):
        nrow = int(self.model.hfield_nrow[0])
        ncol = int(self.model.hfield_ncol[0])

        data = np.random.uniform(0.0, self.max_roughness, (nrow, ncol)).astype(np.float32)

        # Zero out the flat starting pad around the origin.
        # MuJoCo hfield: row 0 = +Y edge, row nrow-1 = -Y edge; col 0 = -X edge, col ncol-1 = +X edge.
        cell_x = (2 * _HFIELD_HALF_X) / ncol
        cell_y = (2 * _HFIELD_HALF_Y) / nrow
        pad_cols = int(np.ceil(self.flat_pad_radius / cell_x))
        pad_rows = int(np.ceil(self.flat_pad_radius / cell_y))
        cx, cy = ncol // 2, nrow // 2
        data[cy - pad_rows : cy + pad_rows + 1, cx - pad_cols : cx + pad_cols + 1] = 0.0

        self.model.hfield_data[: nrow * ncol] = data.flatten()
