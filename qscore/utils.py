from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interpn

from qscore.mrc_utils import MRCObject


def sample_uniformly_on_sphere(sphere_radius: float, num_points: int) -> np.ndarray:
    points = np.random.randn(num_points, 3)
    points /= np.linalg.norm(points, axis=-1, ord=2, keepdims=True)
    return points * sphere_radius

def get_reference_gaussian_params(map: MRCObject) -> Tuple[float, float]:
    map_max = np.max(map.grid)
    map_min = np.min(map.grid)
    map_mean = np.mean(map.grid)
    map_std = np.std(map.grid)
    high_value = min(map_mean + 10 * map_std, map_max)
    low_value = max(map_mean - map_std, map_min)
    reference_gaussian_height = high_value - low_value
    reference_gaussian_offset = low_value
    return reference_gaussian_height, reference_gaussian_offset


def get_radial_points(atoms: np.ndarray, sphere_radius: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    radial_points = np.zeros((len(atoms), num_points, 3))
    point_exists = np.zeros((len(atoms), num_points), dtype=bool)
    kdtree = cKDTree(atoms)
    for num_try in range(100):
        atoms_left = ~np.all(point_exists, axis=-1)
        num_atoms_left = np.sum(atoms_left)
        sphere_points = sample_uniformly_on_sphere(
            sphere_radius, num_points * num_atoms_left
        ).reshape((num_atoms_left, num_points, 3))
        sphere_points += atoms[atoms_left, None]
        # Check if each sphere point is closest to the atom it originates from
        indices = kdtree.query(sphere_points, k=1, workers=4)[1]
        indices_that_match = indices == np.arange(len(atoms))[atoms_left, None] # A x N
        if num_try > 3:
            sort_idx_pe = np.argsort(~point_exists[atoms_left], axis=1)  # True values first
            point_exists[atoms_left] = np.take_along_axis(point_exists[atoms_left], sort_idx_pe, axis=1)
            radial_points[atoms_left] = np.take_along_axis(radial_points[atoms_left], sort_idx_pe[..., None], axis=1)
            sort_idx_sp = np.argsort(indices_that_match, axis=1)  # False values first
            indices_that_match = np.take_along_axis(indices_that_match, sort_idx_sp, axis=1)
            sphere_points = np.take_along_axis(sphere_points, sort_idx_sp[..., None], axis=1)
        idxs_to_update = np.nonzero(indices_that_match & ~point_exists[atoms_left])  # Not exactly optimal, probably should sort first
        radial_points[np.nonzero(atoms_left)[0][idxs_to_update[0]], idxs_to_update[1]] = sphere_points[idxs_to_update[0], idxs_to_update[1]]
        point_exists[np.nonzero(atoms_left)[0][idxs_to_update[0]], idxs_to_update[1]] = True
        if np.all(point_exists):
            break
    return radial_points, point_exists


def interpolate_grid_at_points(points: np.ndarray, map: MRCObject) -> np.ndarray:
    x = np.arange(map.grid.shape[0])
    y = np.arange(map.grid.shape[1])
    z = np.arange(map.grid.shape[2])
    points = np.flip(points, axis=-1)
    # Origin should be flipped the same way, thank you Sjors
    flipped_origin = np.flip(map.global_origin, axis=-1)
    p = (points - flipped_origin[None]) / map.voxel_size
    return interpn((x, y, z), map.grid, p)
