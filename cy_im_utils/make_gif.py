#!/home/mcd4/miniconda3/envs/openeb/bin/python3
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from os import getcwd
import argparse
import imageio
import numpy as np


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",
                        "-i",
                        dest= "input_path",
                        type = str,
                        default = "",
                        help = "input png files",
                        )
    parser.add_argument("--gif_stride",
                        type = int,
                        default = 1,
                        help = "Down sampling to make gif run faster..."
                        )
    parser.add_argument("--extension",
                        type = str,
                        default = "png",
                        help = "file extension to glob"
                        )


    return parser.parse_args()


def create_gif(images: list) -> None:
    imageio.mimwrite(uri = "__rename__.gif",
                    ims = images,
                     format = 'GIF', 
                     kwargs = {'duration':1})


def fetch_images(input_path, extension, gif_stride) -> list:
    image_files = sorted(list(Path(input_path).glob(f"*.{extension}")))[::gif_stride]
    n_images = len(image_files)
    print(np.asarray(Image.open(image_files[0])).shape)
    nx,ny,n_chan = np.asarray(Image.open(image_files[0])).shape
    #images = np.zeros([n_images, nx, ny, n_chan], dtype = np.float32)
    images = []
    desc = f"reading images from {input_path}.*{extension}"
    for i in tqdm(range(n_images), desc = desc):
        images.append(np.asarray(Image.open(image_files[i])))
    return images


def main() -> None:
    """
    Fetch Tif and conditionally write it and or write gif 
    """
    args = parse_args()

    gif_stride = args.gif_stride
    base_path = Path(getcwd()) / args.input_path
    print("--->", base_path)
    images = fetch_images(base_path, args.extension, gif_stride)
    create_gif(images)


if __name__ == "__main__":
    main()
