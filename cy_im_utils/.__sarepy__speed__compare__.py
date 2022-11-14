"""

Testing if using mulitple cores on the CPU can out-perform the GPU, the
multi-core can perform pretty well, but the cost of launching the processes is
a bit high. The pooled version is still a significant speedup over the serial
version though.. 

la_size = 131, sm_size = 11, n_sinograms = 23
    Serial : 111.6 s
    Pool : 22.5 s       (5  x speedup over serial)
    GPU : 5.9 s         (18 x speedup over serial)


"""
import numpy as np
import cupy as cp
from functools import partial
from time import time
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sys import path
path.append("C:\\Users\\mcd4\\Documents\\cy_im_utils")
from cy_im_utils.sarepy_cuda import remove_all_stripe_GPU
from cy_im_utils.prep import imstack_read

path.append("C:\\Users\\mcd4\\Documents\\vo_filter_source\\sarepy")
from sarepy.prep.stripe_removal_original import remove_all_stripe as remove_all_stripe_CPU


def sarepy_pool(index: int,
                sinograms = [],
                snr: float = 2.0,
                la_size: int = 1,
                sm_size: int = 1,
                ):
    return remove_all_stripe_CPU(  sinograms[:,index,:].copy(),
                                    la_size = la_size,
                                    snr = snr,
                                    sm_size = sm_size)
                        
def load_sinograms(truncate = 100):
    files = Path("E:/MIT_May_2022/May_2022/attenuation/MIT_Batteries_2022")
    image_files = sorted(list(files.glob("*.tif")))[::truncate]
    print(files.is_dir(),len(image_files))
    return np.transpose(imstack_read(image_files),(1,0,2))

def plot_diff(
        pool_imstack,
        imstack_serial,
        gpu_imstack,
        index = 5):
    fig,ax = plt.subplots(1,2, sharex = True, sharey = True)
    ax[0].imshow(imstack_serial[:,index,:]-gpu_imstack[:,index,:])
    ax[1].imshow(pool_imstack[:,index,:]-gpu_imstack[:,index,:])
    plt.show()

if __name__ == "__main__":
    sinograms = load_sinograms(truncate = 100)
    snr = 5.0
    la_size = 131
    sm_size = 11
    sarepy_wrapper = partial(
            sarepy_pool,
            sinograms = sinograms,
            la_size = 13,
            sm_size = 3)
    t1 = time()
    gpu_imstack = remove_all_stripe_GPU(
            cp.array(sinograms, dtype = cp.float32),
            snr = snr,
            la_size = la_size,
            sm_size = sm_size).get()

    print('gpu time = ',time()-t1)

    t0 = time()
    n_sino = sinograms.shape[1]
    with Pool() as pool:
        imstack = np.vstack([pool.map(sarepy_wrapper,tqdm(range(n_sino)))]).transpose(1,0,2)
    
    print("pool time = ",time()-t0)
    print(imstack.shape)
   
    t2 = time()
    imstack_serial = np.zeros_like(gpu_imstack, dtype = np.float32)
    for j in tqdm(range(n_sino)):
        imstack_serial[:,j,:] = remove_all_stripe_CPU(
                sinograms[:,j,:].copy(),
                snr = snr,
                la_size = la_size,
                sm_size = sm_size)
    print(imstack_serial.shape)
    print("seiral time = ",time()-t2)
    plot_diff(imstack,imstack_serial,gpu_imstack)

    
