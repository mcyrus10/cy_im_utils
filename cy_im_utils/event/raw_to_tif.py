#!/home/mcd4/miniconda3/envs/openeb/bin/python3
from PIL import Image
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from multiprocessing import pool
from tifffile import imwrite
from tqdm import tqdm
from pickle import load
import cupy as cp
import argparse
import imageio
import numpy as np
import sys


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        "-i",
                        dest= "input",
                        type = str,
                        default = "",
                        help = "input raw file",
                        )
    parser.add_argument("--output",
                        "-o",
                        dest = "output",
                        type = str,
                        default = "raw_to_tif_output.tif",
                        help = "Output File Name",
                        )
    parser.add_argument("--delta_t",
                        "-dt",
                        dest="delta_t",
                        type = float,
                        default = 10_000,
                        help = "delta t (us)",
                        )
    parser.add_argument("--time_init",
                        "-t0",
                        dest="t0",
                        type = int,
                        default = 0,
                        help = "start time",
                        )

    parser.add_argument("--frame_max",
                        type = float,
                        default = np.inf,
                        help = "maximum frame to reduce..."
                        )
    parser.add_argument("--tif",
                        action = argparse.BooleanOptionalAction,
                        default = True,
                        help = "Boolean to SAVE the tif (Default = True)"
                        )
    parser.add_argument("--gif",
                        action = argparse.BooleanOptionalAction,
                        default = False,
                        help = "Create a gif with tif images (Default = False) "
                        )
    parser.add_argument("--gif_stride",
                        type = int,
                        default = 1,
                        help = "Down sampling to make gif run faster..."
                        )

    return parser.parse_args()


def raw_to_numpy(raw_file, delta_t, frame_max = np.inf, t0 = 0) -> np.array:
    """
    Converts .raw to numpy
    """
    mv_iterator = EventsIterator(raw_file,
            start_ts = t0,
            mode = 'delta_t',
            delta_t = delta_t
            )
    height, width = mv_iterator.get_size()
    print('hw = ',height,width)
    image = np.zeros([height, width, 3], dtype = np.uint8)
    print(image.shape)
    images = []
    desc = "raw -> numpy"
    if frame_max is np.nan:
        condition = lambda x: False
    else:
        condition = lambda x: x > frame_max

    for q, ev in tqdm(enumerate(mv_iterator), desc = desc):
        if condition(q):
            break
        BaseFrameGenerationAlgorithm.generate_frame(ev, image)
        images.append(image.copy())

    return np.stack(images)


def create_gif(f_name_gif, gif_stride) -> None:
    imageio.mimwrite(uri = f_name_gif,
                     ims = images[::gif_stride],
                     format = 'GIF', 
                     kwargs = {'duration':1})


def main() -> None:
    """
    Fetch Tif and conditionally write it and or write gif 
    """
    args = parse_args()

    images = raw_to_numpy(raw_file = args.input, 
                          delta_t = args.delta_t, 
                          frame_max = args.frame_max,
                          t0 = args.t0
                          )

    if args.tif:
        f_name_tif = args.output + ".tif"
        imwrite(f_name_tif, images)
        print(f"Done Writing {args.output}.tif")

    if args.gif:
        f_name_gif = args.output + ".gif"
        gif_stride = args.gif_stride
        create_gif()


if __name__ == "__main__":
    main()
