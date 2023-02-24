import mrcfile
import numpy as np
from collections import namedtuple

MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "global_origin"])

def load_mrc(mrc_fn: str, multiply_global_origin: bool = True) -> MRCObject:
    mrc_file_handle = mrcfile.open(mrc_fn, "r")
    voxel_size = float(mrc_file_handle.voxel_size.x)

    if voxel_size <= 0:
        raise RuntimeError(f"Seems like the MRC file: {mrc_fn} does not have a header.")

    c = mrc_file_handle.header["mapc"]
    r = mrc_file_handle.header["mapr"]
    s = mrc_file_handle.header["maps"]

    global_origin = mrc_file_handle.header["origin"]
    global_origin = np.array([global_origin.x, global_origin.y, global_origin.z])
    global_origin[0] += mrc_file_handle.header["nxstart"]
    global_origin[1] += mrc_file_handle.header["nystart"]
    global_origin[2] += mrc_file_handle.header["nzstart"]

    if multiply_global_origin:
        global_origin *= mrc_file_handle.voxel_size.x

    if c == 1 and r == 2 and s == 3:
        grid = mrc_file_handle.data
    elif c == 3 and r == 2 and s == 1:
        grid = np.moveaxis(mrc_file_handle.data, [0, 1, 2], [2, 1, 0])
    elif c == 2 and r == 1 and s == 3:
        grid = np.moveaxis(mrc_file_handle.data, [1, 2, 0], [2, 1, 0])
    else:
        raise RuntimeError("MRC file axis arrangement not supported!")

    return MRCObject(grid, voxel_size, global_origin)

