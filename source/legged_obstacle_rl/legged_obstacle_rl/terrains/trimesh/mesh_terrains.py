import numpy as np
import trimesh
from isaaclab.terrains.trimesh.utils import make_plane


def diamond_walkway_terrain(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a grid of diamond-shaped walkways."""

    # 1. Resolve global parameters
    beam_width = cfg.beam_width_range[1] + (1.0 - difficulty) * (cfg.beam_width_range[0] - cfg.beam_width_range[1])
    beam_height = cfg.beam_height * max(0.2, difficulty)

    # Calculate cell dimensions
    num_x, num_y = cfg.grid_dims
    cell_size_x = cfg.size[0] / num_x
    cell_size_y = cfg.size[1] / num_y

    # Geometry for a single diamond inside a cell
    half_cx = cell_size_x / 2.0
    half_cy = cell_size_y / 2.0
    diag_length = np.sqrt(half_cx**2 + half_cy**2)
    angle = np.arctan2(half_cy, half_cx)

    # Initialize meshes with the base plane
    meshes_list = [make_plane(cfg.size, 0.0, center_zero=False)]

    # 2. Iterate through the grid
    for i in range(num_x):
        for j in range(num_y):
            # Calculate the bottom-left corner offset for this cell
            offset_x = i * cell_size_x
            offset_y = j * cell_size_y

            # Local center of the current cell
            cell_center_x = offset_x + half_cx
            cell_center_y = offset_y + half_cy

            # --- Horizontal Center Beam ---
            # Width is limited to the cell width
            h_beam_dims = (cell_size_x, beam_width, beam_height / 4)
            h_beam_pos = (cell_center_x, cell_center_y, beam_height * 7.0 / 8.0)
            h_beam = trimesh.creation.box(h_beam_dims, trimesh.transformations.translation_matrix(h_beam_pos))
            meshes_list.append(h_beam)

            # --- Diagonal Beams ---
            diag_dims = (diag_length, beam_width, beam_height / 4)

            # Local orientations relative to the cell
            # We use the same logic as before, but add the cell's offset_x/y
            orientations = [
                ([offset_x + half_cx / 2.0, offset_y + 3 * half_cy / 2.0], angle),  # Top-Left
                ([offset_x + 3 * half_cx / 2.0, offset_y + 3 * half_cy / 2.0], -angle),  # Top-Right
                ([offset_x + half_cx / 2.0, offset_y + half_cy / 2.0], -angle),  # Bottom-Left
                ([offset_x + 3 * half_cx / 2.0, offset_y + half_cy / 2.0], angle),  # Bottom-Right
            ]

            for pos_xy, rot_z in orientations:
                matrix = trimesh.transformations.translation_matrix([pos_xy[0], pos_xy[1], beam_height * 7.0 / 8.0])
                matrix = trimesh.transformations.concatenate_matrices(
                    matrix, trimesh.transformations.rotation_matrix(rot_z, [0, 0, 1])
                )
                diag_mesh = trimesh.creation.box(diag_dims, matrix)
                meshes_list.append(diag_mesh)

    origin = np.array([cfg.size[0] / 2, cfg.size[0] / 2, 0.0])

    return meshes_list, origin
