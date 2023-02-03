# cy_im_utils (Cyrus' Image Utils)

KEEP IT SIMPLE STUPID:
**This should mostly just be functions, unless a class is really appropriate,
it will be simpler most of the time to just deal with numpy arrays.**

**Default dtype for everything: np.float32 (equivalent to cp.float32)**

The most natural way to represent projection image stack as 3D arrays:
- index 1 = Image index (in the stack)
- index 2 = Column
- index 3 = Row

Need to think about a consistent image representation!

cy_im_utils is the root

branch tier 1:
- Prep? -> All high performance
  - GPU data reduction pipeline? (Radiographs -> Attenuation?)
  - Field
  - Imread -> Dask?
  - extract norm patch
- Extract?
  - patch_slice
- Visualization
  - template for an interactive plot
  - patch plotting
  - nif_99to01contrast
    - modified this to quantiles
- statistical testing
- Patch_extract

## sarepy_cuda (Sarepy GPU functions)

My adaptation of Vo et. al's SAREPY. Here are a few considerations for possible
efficiency boosts:
- All sarepy functions make a copy (i.e. not in-place) 
    - should this be a conditional or just hard code in-place operations?
    - Vo's code always makes a copy
- *nd_interp2d* is a function to replace *scipy.interpolate.interp2d*
    - this function is a bit hacky, but doesn't seem to be terribly
      inefficient.
    - a vectorized version of this would make me feel better
- *detect_stripe_GPU* has a for loop over all the sinograms that can easily
  be turned into a kernel
- I haven't done much testing to see if my default *threads_per_block* (8,8,8)
  is optimized

![SAREPY GPU Array format](sarepy_gpu_array_format.png)

SAREPY GPU array format.

---

### CUDA notes

[link to cuda slides](https://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf)

[link to stack overflow](https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-gridhttps://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid)
> First of all, your thread block size should always be a multiple of 32, because kernels issue instructions in warps (32 threads).

[link to nvidia blog post](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

<<<a,b>>> $\rightarrow$ 
- a = number of thread blocks
- b = threads in a thread block

This will execute (add) once per thread rather than spreading hte computation across the parallel threads

    add<<<1,256>>>(N,x,y) 
    
> Together, the blocks of parallel threads make up what is known as the *grid*. Since I have N elements to process, and 256 threads per block, I just need to calculate the number of blocks to get at least N threads. I simply divide N by the block size (being careful to round up in case N is not a multiple of blockSize).

    int N = 1<<20;
    int blockSize = 256;
    int numBlocks = (N+blockSize-1) / blockSize;
    add<<<numBlocks, blockSize>>>(N,x,y);


