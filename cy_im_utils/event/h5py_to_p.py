#!/home/mcd4/miniforge3/envs/im_proc/bin/python
"""
Converting hdf5 files to serialized pickle files for reading on cluster...

    usage (im_proc environment...):
        $ python h5py_to_p.py --f_name <file_name>

"""
import argparse
from pathlib import Path
import pickle
from sys import path
path.append("/home/mcd4/cy_im_utils")
from cy_im_utils.event.read_hdf5 import __read_hdf5__


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--f_name",
                        type=str,
                        default = "",
                        help = "file name"
                        )

    return parser.parse_args()


if __name__ == "__main__":
    f_name = Path(parse().f_name)
    suffix = f_name.suffix
    name = f_name.name.split(suffix)[0]
    parent = f_name.parent
    f_name_out = parent / f"{name}.p"
    print(f"reading {f_name}")
    cd_data = __read_hdf5__(f_name, "CD")
    trigger_data = __read_hdf5__(f_name, "EXT_TRIGGER")
    print(f"writing {f_name_out}")
    with open(f_name_out, "wb") as f:
        pickle.dump({
                    'cd_data':cd_data,
                    'trigger_data':trigger_data
                    }, f)
