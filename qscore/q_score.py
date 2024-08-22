import os.path

import numpy as np
from tqdm import tqdm

from qscore.mrc_utils import MRCObject, load_mrc
from qscore.pdb_utils import get_protein_from_file_path, index_to_restype_3, protein_to_cif
from qscore.utils import get_reference_gaussian_params, get_radial_points, interpolate_grid_at_points


def calculate_q_score(
        atoms: np.ndarray,
        map: MRCObject,
        ref_gaussian_width: float = 0.6,
        num_points: int = 8,
        epsilon: float = 1e-6,
) -> np.ndarray:
    num_atoms = len(atoms)
    ref_gaussian_height, ref_gaussian_offset = get_reference_gaussian_params(map)
    reference_gaussian_values = []
    map_values = []
    for R in tqdm(range(21)):
        R /= 10
        radial_points = get_radial_points(atoms, R, num_points)
        map_values_at_points = interpolate_grid_at_points(radial_points[0], map)
        map_values.append(map_values_at_points)
        ref_gaussian_value_at_R = ref_gaussian_height * np.exp(
            - 0.5 * (R/ref_gaussian_width)**2
        ) + ref_gaussian_offset
        reference_gaussian_values.append([ref_gaussian_value_at_R] * num_points)
    v_vector = np.stack(reference_gaussian_values, axis=1).reshape(1, -1)  # N x M
    u_vector = np.stack(map_values, axis=2).reshape(num_atoms, -1)  # A x N x M
    v_norm = v_vector - np.mean(v_vector, axis=1, keepdims=True)
    u_norm = u_vector - np.mean(u_vector, axis=1, keepdims=True)
    Q = np.sum(u_norm * v_norm, axis=-1) / np.sqrt(
            np.sum(u_norm * u_norm, axis=-1) * np.sum(v_norm * v_norm, axis=-1) + epsilon
    )
    return Q


def calculate_per_residue_q_scores(
        structure_path: str,
        map_path: str,
        output_path: str,
):
    prot = get_protein_from_file_path(structure_path)
    map = load_mrc(map_path, False)
    atoms = prot.atom_positions[prot.atom_mask.astype(bool)]
    q_scores = calculate_q_score(atoms, map)
    q_score_per_residue = np.zeros_like(prot.atom_mask, dtype=np.float32)
    q_score_per_residue[prot.atom_mask.astype(bool)] = q_scores
    q_score_per_residue = q_score_per_residue.sum(axis=1) / prot.atom_mask.sum(axis=1)
    avg_q_score = np.mean(q_scores)
    min_q_score = np.min(q_scores)
    max_q_score = np.max(q_scores)
    print(f"Mean: {avg_q_score}, Min: {min_q_score}, Max: {max_q_score}")
    base_path = os.path.splitext(output_path)[0]
    with open(f"{base_path}.csv", "w") as f:
        for resid in range(len(prot.aatype)):
            f.write(
                f"{prot.chain_id[prot.chain_index[resid]]},"
                f"{resid+1},"
                f"{index_to_restype_3[prot.aatype[resid]]},"
                f"{q_score_per_residue[resid]:.6f}\n"
            )

    prot.b_factors = q_score_per_residue * 100
    protein_to_cif(prot, f"{base_path}.cif")
