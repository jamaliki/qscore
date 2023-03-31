import dataclasses
from typing import List

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, MMCIFIO
from Bio.PDB.StructureBuilder import StructureBuilder

index_to_restype_3 = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "DA",
    "DC",
    "DG",
    "DT",
    "A",
    "C",
    "G",
    "U",
]
index_to_restype_1 = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "x",
    "y",
    "z",
    "t",
    "a",
    "c",
    "g",
    "u",
]

num_prot = 20
prot_restype3 = set(index_to_restype_3[:num_prot])

restype3_to_atoms = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "DA": [
        "OP1",
        "P",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N9",
        "C4",
        "N3",
        "C2",
        "N1",
        "C6",
        "C5",
        "N7",
        "C8",
        "N6",
    ],
    "DC": [
        "OP1",
        "P",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
    ],
    "DG": [
        "OP1",
        "P",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N9",
        "C4",
        "N3",
        "C2",
        "N1",
        "C6",
        "C5",
        "N7",
        "C8",
        "N2",
        "O6",
    ],
    "DT": [
        "OP1",
        "P",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C7",
        "C6",
    ],
    "A": [
        "OP1",
        "P",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C1'",
        "C2'",
        "O2'",
        "N1",
        "C2",
        "N3",
        "C4",
        "C5",
        "C6",
        "N6",
        "N7",
        "C8",
        "N9",
    ],
    "C": [
        "OP1",
        "P",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C1'",
        "C2'",
        "O2'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
    ],
    "G": [
        "OP1",
        "P",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C1'",
        "C2'",
        "O2'",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
        "C5",
        "C6",
        "O6",
        "N7",
        "C8",
        "N9",
    ],
    "U": [
        "OP1",
        "P",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C1'",
        "C2'",
        "O2'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C6",
    ],
}

restype_3_to_index = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "DA": 20,
    "DC": 21,
    "DG": 22,
    "DT": 23,
    "A": 24,
    "C": 25,
    "G": 26,
    "U": 27,
}

restype_3to1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "DA": "x",
    "DC": "y",
    "DG": "z",
    "DT": "t",
    "A": "a",
    "C": "c",
    "G": "g",
    "U": "u",
}

restype_num = len(index_to_restype_3)
restype_order = {restype: i for i, restype in enumerate(index_to_restype_1)}

atom_types = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CD",
    "NE",
    "CZ",
    "NH1",
    "NH2",
    "OD1",
    "ND2",
    "OD2",
    "SG",
    "OE1",
    "NE2",
    "OE2",
    "ND1",
    "CD2",
    "CE1",
    "CG1",
    "CG2",
    "CD1",
    "CE",
    "NZ",
    "SD",
    "CE2",
    "OG",
    "OG1",
    "NE1",
    "CE3",
    "CZ2",
    "CZ3",
    "CH2",
    "OH",
    "OP1",
    "P",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
    "N9",
    "C4",
    "N3",
    "C2",
    "N1",
    "C6",
    "C5",
    "N7",
    "C8",
    "N6",
    "O2",
    "N4",
    "N2",
    "O6",
    "O4",
    "C7",
    "O2'",
    "OXT",
]

def restype3_is_prot(restype3: str) -> bool:
    return restype3 in prot_restype3

restype3_to_atoms_index = dict(
    [
        (res, dict([(a, i) for (i, a) in enumerate(atoms)]))
        for (res, atoms) in restype3_to_atoms.items()
    ]
)
for residue in restype3_to_atoms_index:
    if restype3_is_prot(residue):
        restype3_to_atoms_index[residue]["OXT"] = restype3_to_atoms_index[residue]["O"]

atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 65.
num_atoms = atom_type_num
num_atomc = max([len(c) for c in restype3_to_atoms_index.values()])
restype_name_to_atomc_names = {}
for k in restype3_to_atoms:
    res_num_atoms = len(restype3_to_atoms)
    atom_names = restype3_to_atoms[k]
    if len(atom_names) < num_atomc:
        atom_names += [""] * (num_atomc - len(atom_names))
    restype_name_to_atomc_names[k] = atom_names


@dataclasses.dataclass(frozen=False)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # _rc.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, 65, 3]

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # _rc.atom_types, i.e. the first three are N, CA, CB.
    atomc_positions: np.ndarray  # [num_res, _rc.num_atomc, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Same as above, but for atomc
    atomc_mask: np.ndarray

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # The original Chain ID string list that the chain_indices correspond to.
    chain_id: np.ndarray  # [num_chains]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chains to residues
    chain_idx_to_residues: List[np.ndarray]
    # Whether or not the residue is a protein residue
    prot_mask: np.ndarray

def get_protein_from_file_path(file_path: str, chain_id: str = None) -> Protein:
    """Takes a file path containing a PDB/mmCIF file and constructs a Protein object.
    WARNING: All non-standard residue types will be ignored. All
      non-standard atoms will be ignored.
    Args:
      pdb_str: The path to the PDB file
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.
    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    if file_path.split(".")[-1][-3:] == "pdb":
        parser = PDBParser(QUIET=True)
    elif file_path.split(".")[-1][-3:] == "cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise RuntimeError("Unknown type for structure file:", file_path[-3:])
    structure = parser.get_structure("none", file_path)
    models = list(structure.get_models())
    model = models[0]

    atom_positions = []
    atomc_positions = []
    aatype = []
    atom_mask = []
    atomc_mask = []
    residue_index = []
    chain_ids = []
    chain_idx_to_residues = []
    b_factors = []

    # Sequence related
    residue_count = 0

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        chain_seq = []
        chain_res_ids = []
        chain_aatype = []
        for res in chain:
            if res.resname not in restype_3_to_index:
                continue
            res_shortname = restype_3to1[res.resname]
            restype_idx = restype_order.get(res_shortname, restype_num)

            pos = np.zeros((atom_type_num, 3))
            posc = np.zeros((num_atomc, 3))
            mask = np.zeros((atom_type_num,))
            maskc = np.zeros((num_atomc,))
            res_b_factors = np.zeros((atom_type_num,))
            for atom in res:
                if atom.name not in atom_types:
                    continue
                pos[atom_order[atom.name]] = atom.coord
                posc[restype3_to_atoms_index[res.resname][atom.name]] = atom.coord
                mask[atom_order[atom.name]] = 1.0
                maskc[restype3_to_atoms_index[res.resname][atom.name]] = 1.0
                res_b_factors[atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            chain_aatype.append(restype_idx)
            atom_positions.append(pos)
            atomc_positions.append(posc)
            atom_mask.append(mask)
            atomc_mask.append(maskc)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)
            chain_res_ids.append(residue_count)
            residue_count += 1
        aatype.extend(chain_aatype)
    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    atom_positions = np.array(atom_positions)
    atomc_positions = np.array(atomc_positions)
    atom_mask = np.array(atom_mask)
    atomc_mask = np.array(atomc_mask)
    aatype = np.array(aatype)
    residue_index = np.array(residue_index)
    b_factors = np.array(b_factors)
    return Protein(
        atom_positions=atom_positions,
        atomc_positions=atomc_positions,
        atom_mask=atom_mask,
        atomc_mask=atomc_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        chain_id=unique_chain_ids,
        b_factors=b_factors,
        chain_idx_to_residues=chain_idx_to_residues,
        prot_mask=aatype < num_prot,
    )

def protein_to_cif(
    protein: Protein, path_to_save: str,
):
    if protein.b_factors is None:
        bfactors = np.zeros(len(protein.aatype))
    else:
        bfactors = protein.b_factors
    if len(bfactors.shape) > 1:
        bfactors = bfactors[:, 0]
    struct = StructureBuilder()
    curr_chain = 0

    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")
    struct.init_chain(protein.chain_id[0])

    prev_chain = protein.chain_index[0]
    for i in range(protein.aatype.shape[0]):
        res_name_3 = index_to_restype_3[protein.aatype[i]]
        bfactor = bfactors[i]
        atom_names = restype_name_to_atomc_names[res_name_3]
        res_counter = 0
        if prev_chain != protein.chain_index[i]:
            curr_chain += 1
            struct.init_chain(protein.chain_id[protein.chain_index[i]])
        prev_chain = protein.chain_index[i]

        struct.init_residue(res_name_3, " ", i, " ")
        for atom_name, pos, mask in zip(
            atom_names, protein.atomc_positions[i], protein.atomc_mask[i]
        ):
            if mask < 0.5:
                continue
            struct.set_line_counter(i + res_counter)
            struct.init_atom(
                name=atom_name,
                coord=pos,
                b_factor=bfactor,
                occupancy=1,
                altloc=" ",
                fullname=atom_name,
                element=atom_name[0],
            )
            res_counter += 1
    struct = struct.get_structure()
    io = MMCIFIO()
    io.set_structure(struct)
    io.save(path_to_save)
