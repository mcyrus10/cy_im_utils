---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# CUDA Data Reduction Pipeline

[link to ASTRA docs](http://www.astra-toolbox.com/#) (Using version 2.0.0 now)


- ~~Rotate at image read~~
- ~~%Timeit on vo filter~~
- add logging to this
- Is GPU attenuation batch function ready to be moved over to cy_im_utils?

---

### Process Flow

0. Dictionary for data set parameters
1. (CPU)
    - Calculate COR and center the cropping window
2. (CPU) Read images $\rightarrow$ 
3. Batch (GPU)
    - -= df 
    - /= (ff-df)
    - normalize
    - spatial (3x3) median (helps with nans)
    - Lambert-Beer
    - crop
4. Batch (GPU)
    - Vo
5. ASTRA (GPU)


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

---

## Vo filter settings from Jake (9/21/2021):
- remove_all_stripe
- snr = 1.5
- sm_size = 3
- la_size = 85
- drop_ratio = default

Vo on projection images
<!-- #endregion -->

```python
%matplotlib widget
%load_ext autoreload
%autoreload 2
from sys import path
path.append("C:\\Users\\mcd4\\Documents\\cy_im_utils")
from cy_im_utils.prep import *
from cy_im_utils.post import *
from cy_im_utils.visualization import *
from cy_im_utils.sarepy_cuda import *
from cy_im_utils.recon_utils import *
path.append("C:\\Users\\mcd4\\Documents\\vo_filter_source\\sarepy")
from sarepy.prep.stripe_removal_original import remove_all_stripe as remove_all_stripe_CPU


from PIL import Image
from cupyx.scipy.ndimage import gaussian_filter,rotate as rotate_gpu, median_filter as median_gpu
from data_sets import *
from glob import glob
from ipywidgets import widgets
from matplotlib.gridspec import GridSpec
from numba import cuda
from scipy.ndimage import rotate as rotate_cpu
from tqdm import tqdm
import astra
import cupy as cp
import numpy as np
import pickle

#from scipy.ndimage import median_filter as median_cpu
```

```python
%matplotlib widget
import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.cmap'] = 'Spectral'
```

### Select Dataset

see *data_sets.py* for all current data sets

```python
data_sets = [AAA_bot,MIT,cliffs,granite,ni_cylinders,Pens]
data_set_select = widgets.Select(    options=[(d['Name'],d) for d in data_sets],
                                     description='Data Set:')
display(data_set_select)
```

```python
data_set = data_set_select.value

Transpose = data_set['Transpose']
dtype = data_set['dtype']
ext = data_set['extension']
if ext == 'tif':
    read_fcn = Image.open
elif ext == 'fit':
    read_fcn = imread_fit
print("\t\tData Set")
print("-"*80)
data_set
```

### Calculate center of rotation 

**(PRIOR TO IMPORTING ALL IMAGES)**

- $\checkmark$ ~~Convert this into an interactive function where you can control the cropping, normalization patch, COR coordinates and transpose~~
- $\checkmark$ ~~make this function load in ff and f itself so you can clean it up a bit~~
- $\checkmark$ ~~enforce COR y1 cannot be larger than crop y1 - crop y0~~

```python
interact = True
if interact:
    d_theta = 60
    angles = [j*d_theta for j in range(360//d_theta)]
    print(angles)
    COR_interact(data_set, angles = angles, figsize = (10,4), cmap = 'gist_ncar')
```

```python
def read_projections(data_dict : dict, dtype = np.float32) -> np.array:
    read_fcn = data_dict['imread function']
    proj_files = glob(data_dict['projection path'])
    n_proj = len(proj_files)
    height,width = np.asarray(read_fcn(proj_files[0])).shape
    projections = np.zeros([n_proj,height,width], dtype = dtype)
    for i in tqdm(range(n_proj)):
        projections[i,:,:] = np.asarray(read_fcn(proj_files[i]), dtype = dtype)
    return projections
```

### Read in projection images

```python
%%time
ff = field_gpu(glob(data_set['flat path']), dtype = dtype)
df = field_gpu(glob(data_set['dark path']), dtype = dtype)
n_projections = len(glob(data_set['projection path']))
Transpose = data_set['Transpose']
projections = pickle.load(open(data_set['serialized path'],'rb'))
#projections = read_projections(data_set)
if Transpose:
    projections = np.transpose(projections,(0,2,1))
    ff = ff.T
    df = df.T
print(projections.shape)
```

## GPU Attenuation Batch Loop

Testing this with batching numpy: 19.70 seconds per iteration

|  | Iterations per second (batch size = 20)| speedup |
| ---|--- |---|
numpy | .042 |  - |
cupy | 0.60 | 14.1 |




```python
def attenuation_gpu_batch(input_arr : cp.array, ff : cp.array, df : cp.array ,output_arr : cp.array ,id0 : int,id1 : int,batch_size : int,norm_patch : list,
                          crop_patch : list, theta : float, kernel :int = 3, dtype = cp.float32) -> None:
    """
    This is a monster (and probably will need some modifications)
    1) upload batch to GPU
    2) rotate
    3) transpose <------------ NOT NECESSARY SINCE YOU KNOW THE BLOCK STRUCTURE NOW
    4) convert image to transmission space
    5) extract normalization patches
    6) normalize transmission images
    7) spatial median (kernel x kernel) -> improves nans when you take -log
    8) lambert beer
    9) reverse the transpose from 3
    10) crop
    11) insert batch into output array
    Parameters:
    -----------
    input_arr: 3D numpy array 
        input volume array
    ff: 2D cupy array 
        flat field
    df: 2D cupy array 
        dark field
    output_arr: 3D numpy array 
        array to output into
    id0: int
        first index of batch
    id1: int
        final index of batch
    batch_size: int
        size of batch
    norm_patch: list
        list of coordinates of normalization patch (x0,x1,y0,y1)
    crop_patch: list
        list of coordinates of crop patch (x0,x1,y0,y1)
    theta: float
        angle to rotate the volume through
    kernel: int (odd number)
        size of median kernel
    dtype: numpy data type
        data type of all arrays
    """
    n_proj,height,width = input_arr.shape
    projection_gpu = cp.asarray(input_arr[id0:id1], dtype = dtype)
    projection_gpu = rotate_gpu(projection_gpu,theta, axes = (1,2), reshape = False)
    # uncomment line after for numpy
    #projection_gpu = rotate_cpu(projection_gpu,theta, axes = (1,2), reshape = False)
    projection_gpu -= df.reshape(1,height,width)
    projection_gpu /= (ff-df).reshape(1,height,width)
    patch = cp.mean(projection_gpu[:,norm_patch[0]:norm_patch[1],norm_patch[2]:norm_patch[3]], axis = (1,2), dtype = dtype)
    projection_gpu /= patch.reshape(batch_size,1,1)
    projection_gpu = median_gpu(projection_gpu, (1,kernel,kernel))
    # uncomment line after for numpy
    #projection_gpu = median_cpu(projection_gpu, (1,kernel,kernel))
    
    projection_gpu = -cp.log(projection_gpu)
    #-----------------------------------------------
    #---      make all non-finite values 0?      ---
    #-----------------------------------------------
    #-----------------------------------------------
    projection_gpu[~cp.isfinite(projection_gpu)] = 0
    #-----------------------------------------------
    #-----------------------------------------------
    output_arr[id0:id1] = cp.asnumpy(projection_gpu[:,crop_patch[0]:crop_patch[1],crop_patch[2]:crop_patch[3]])
    #output_arr[id0:id1] = projection_gpu[:,crop_patch[0]:crop_patch[1],crop_patch[2]:crop_patch[3]]
```

```python
#%%time
batch_size = 20
#batch_size = 1
#=====================================================================================
# PROJECTION LOOP (Projections -> Transmission -> Normalize -> Median -> Attenuation)
#=====================================================================================
# READ PARAMETERS FROM data_set DICT
theta = data_set['theta']
crop_patch = data_set['crop patch']
norm_patch = data_set['norm patch']
COR_rows = data_set['COR rows']

n_proj,height,width = projections.shape
nx = crop_patch[1]-crop_patch[0]
ny = crop_patch[3]-crop_patch[2]
attn = np.empty([n_proj,nx,ny], dtype = dtype)
kernel = 3

# For numpy speedup comparison
#ff = cp.asnumpy(ff)
#df = cp.asnumpy(df)
for j in tqdm(range(n_proj//batch_size)):
    id0,id1 = j*batch_size,(j+1)*batch_size
    attenuation_gpu_batch(projections,ff,df,attn,id0,id1,batch_size,norm_patch,crop_patch, theta , kernel = kernel, dtype = dtype)

remainder = n_proj%batch_size
attenuation_gpu_batch(projections,ff,df,attn,-remainder,n_proj,remainder,norm_patch,crop_patch, theta , kernel = kernel, dtype = dtype)

print("non finite = ",np.sum(~np.isfinite(attn)))
```

```python
# FREE UP MEMORY??
#del volume 
```

```python
print(theta,COR_rows[0], COR_rows[1])
sino_index = 1000
combined = attn[0]+attn[n_proj//2]
slice_ = combined.T[COR_rows[0]:COR_rows[1]]
if np.sum(~np.isfinite(slice_)) > 0:
    print(np.sum(~np.isfinite(combined)))
    print('nans in region of interest fitting -> can cause SVD to not converge')
fig,ax = plt.subplots(2,2, figsize = (10,10))
ax = ax.flatten()
ax[0].imshow(attn[0,:,:])
ax[0].plot([sino_index,sino_index],[0,nx-1],'k--')
ax[0].plot([0,ny-1],[nx//2,nx//2],'k--')
center_of_rotation(combined.T,COR_rows[0],COR_rows[1],ax[1])
ax[2].imshow(attn[:,nx//2,:])
ax[3].imshow(attn[:,:,sino_index])
fig.tight_layout()
```

```python
#_,ax = plt.subplots(2,1)
#ax[0].imshow(projections[:,:,1600])
#ax[1].imshow(attn[:,:,1600])
```

## GPU Sinogram Batch Filtering

```python
cpu_cpy = attn.copy()
```

```python
#attn = cpu_cpy.copy()
attn = np.transpose(attn,(0,2,1))
```

### Sarepy Filter Settings

```python
SAREPY_interact(data_set, attn, figsize = (12,5))
```

### SAREPY loop

Vo_batch is a function that moves batches of sinograms to the gpu, executes the vo filter and moves back to cpu

Can this be sped up with multi-threaded pipeline so the copy - process -write structure is handled by one thread each?

```python
def vo_batch(attenuation : cp.array, dim : int = 1, batch_size : int = 50, 
             snr : float = 1.5 ,la_size : int = 85, sm_size : int = 3, 
             in_place : bool = False):
    if not in_place:
        attenuation = attenuation.copy()
        
    if dim == 2:
        attenuation = np.transpose(attenuation,(0,2,1))
        
    n_proj,n_row,n_col = attenuation.shape
    for j in tqdm(range(n_row//batch_size)):
        id0,id1 = j*batch_size, (j+1)*batch_size
        vol_gpu = cp.asarray(attenuation[:,id0:id1,:])
        vol_gpu = remove_all_stripe_GPU(vol_gpu,snr,la_size,sm_size)
        attenuation[:,id0:id1,:] = cp.asnumpy(vol_gpu)

    remainder = n_row%batch_size
    if remainder > 0:
        vol_gpu = cp.asarray(attenuation[:,-remainder:,:])
        vol_gpu = remove_all_stripe_GPU(vol_gpu,snr,la_size,sm_size)
        attenuation[:,-remainder:,:] = cp.asnumpy(vol_gpu)
        
    if not in_place:
        if dim == 2:
            attenuation = np.transpose(attenuation,(0,2,1))
        return attenuation
    
    else:
        return None
```

### Vo GPU Speedup

```python
%%time
#===============================================================================
# SINOGRAM LOOP (Vo Filter)
#===============================================================================
n_proj,n_row,n_col = attn.shape
batch_size = 50
la_size = data_set['large filter']
sm_size = data_set['small filter']
snr = data_set['signal to noise ratio']
container = None

#attn_subset = attn.copy()#[:,::25,:]
#print(attn_subset.shape)
#vo_container = vo_batch(attn_subset, dim = 2, batch_size = batch_size, snr = snr, la_size = la_size, sm_size = sm_size)
vo_container = vo_batch(attn, dim = 1, batch_size = batch_size, snr = snr, la_size = la_size, sm_size = sm_size)
#vo_batch(attn, dim = 1, batch_size = batch_size, snr = snr, la_size = la_size, sm_size = sm_size)
#container = np.zeros_like(attn)
#for j in tqdm(range(n_row)):
#    container[:,j,:] = remove_all_stripe_CPU(attn[:,j,:], snr,la_size,sm_size)


print("-"*80)
print("-"*80)
print("COPY vo_container TO ATTN TO MOVE FORWARD")
print("-"*80)
print("-"*80)
```

```python
attn = vo_container.copy()
```

```python
"""
fig,ax = plt.subplots(3,1, figsize = (6,10))
#ax = [ax]
ax[0].imshow(temp[:,0,:])
ax[1].imshow(filter_cpu)
ax[2].imshow(filter_cpu-temp[:,0,:])
fig.tight_layout()
"""
```

```python
sinogram_index = 1600
l,h = np.quantile(attn[n_proj//2].flatten(),0.01),np.quantile(attn[n_proj//2].flatten(),0.99)
#print(l,h)
fig,ax = plt.subplots(3,3, figsize = (8,12))
ax[0,0].imshow(attn[n_proj//2,:,:], vmin = l, vmax = h)
ax[0,0].set_title("Projection")
ax[0,0].plot([nx//2,nx//2],[0,ny-1],'k--')
ax[0,0].plot([0,nx-1],[ny//2,ny//2],'k--')
ax[1,0].imshow(attn[:,sinogram_index,:].T, vmin = l, vmax = h)
ax[1,0].set_title("Sinogram (unfiltered)")
ax[2,0].imshow(attn[:,:,n_col//2].T, vmin = l, vmax = h)

ax[0,1].imshow(vo_container[n_proj//2,:,:], vmin = l, vmax = h)
ax[0,1].set_title("Projection")
ax[0,1].plot([nx//2,nx//2],[0,ny-1],'k--')
ax[0,1].plot([0,nx-1],[ny//2,ny//2],'k--')
ax[1,1].imshow(vo_container[:,sinogram_index,:].T, vmin = l, vmax = h)
ax[1,1].set_title("Sinogram (filtered)")
ax[2,1].imshow(vo_container[:,:,n_col//2].T, vmin = l, vmax = h)

ax[1,2].imshow(remove_all_stripe_CPU(attn[:,sinogram_index,:],snr,la_size,sm_size).T, vmin = l, vmax = h)
ax[1,2].set_title("Vo CPU (reference)")

ax[0,2].axis('off')
ax[2,2].axis('off')

fig.tight_layout()
```

# ASTRA

fbp_cuda_3d: this is very non-optimized but FBP_CUDA does not take 3d geometries

```python
# TRANSPOSE ATTENUATION ARRAY -> ASTRA takes the sinogram as the first index
attn = np.transpose(vo_container,(1,0,2))
```

```python
def fbp_cuda_3d(attn : cp.array, pixel_size : float) -> cp.array:
    """
    naiive implementation of FBP_CUDA on each sinogram individually; not sure why this 
    isn't in the regular ASTRA configuration for parallel3d.... 
    Parameters:
    ----------
    attn: 3d numpy array
        attenuation volume
        
    pixel_size: float
        pixel size in mm
        
    returns:
    --------
    recon: 3d numpy array
        reconstructed volume
        
    """
    detector_rows = attn.shape[0]
    detector_cols = attn.shape[2]
    n_projections = attn.shape[1]
    angles = np.linspace(0, 2 * np.pi, num = n_projections, endpoint=False)
    recon = np.zeros([detector_rows,detector_cols,detector_cols], dtype = np.float32)
    algorithm = 'FBP_CUDA'
    for row in tqdm(range(detector_rows)):
        proj_geom = astra.create_proj_geom('parallel', 1, detector_cols, angles)
        sino_id = astra.data2d.create('-sino', proj_geom, attn[row])
        vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols)
        reconstruction_id = astra.data2d.create('-vol', vol_geom)
        alg_cfg = astra.astra_dict(algorithm)
        alg_cfg['ProjectionDataId'] = sino_id
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        alg_cfg['option'] = {'FilterType': 'ram-lak'}
        algorithm_id = astra.algorithm.create(alg_cfg)
        #-------------------------------------------------
        astra.algorithm.run(algorithm_id)  # This is slow
        #-------------------------------------------------
        recon[row] = astra.data2d.get(reconstruction_id)
        # DELETE OBJECTS
        astra.algorithm.delete(algorithm_id)
        astra.data2d.delete([sino_id,reconstruction_id])
    return recon/pixel_size
```

### ASTRA_GENERIC

This is just function of convenience to hide some of the code

```python tags=[]
def ASTRA_GENERIC(attn : cp.array,geometry : str = 'cone', algorithm : str = 'FDK_CUDA', detector_pixel_size : float = 0.0087, 
                  source_origin : float = 5965.0, origin_detector : float = 35.0):
    """
    algorithm for cone -> FDK_CUDA
    algorithms for Parallel -> SIRT3D_CUDA, FP3D_CUDA, BP3D_CUDA
    """
    detector_rows,n_projections,detector_cols = attn.shape
    distance_source_origin = source_origin
    distance_origin_detector = origin_detector
    angles = np.linspace(0, 2 * np.pi, num = n_projections, endpoint=False)
    #  ---------    PARALLEL BEAM    --------------
    if geometry.lower() == 'parallel':
        proj_geom = astra.create_proj_geom('parallel3d', 1, 1, detector_rows, detector_cols, angles)
        projections_id = astra.data3d.create('-sino', proj_geom, attn)
    #  ---------    CONE BEAM    --------------
    elif geometry.lower() == 'cone':
        proj_geom = astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
                (distance_source_origin + distance_origin_detector) / detector_pixel_size, 0)
        projections_id = astra.data3d.create('-sino', proj_geom, attn)
        
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, detector_rows)
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
    alg_cfg = astra.astra_dict(algorithm)
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    alg_cfg['option'] = {'FilterType': 'ram-lak'}
    algorithm_id = astra.algorithm.create(alg_cfg)
    
    #-------------------------------------------------
    astra.algorithm.run(algorithm_id)  # This is slow
    #-------------------------------------------------

    reconstruction = astra.data3d.get(reconstruction_id)
    reconstruction /= detector_pixel_size

    # DELETE OBJECTS TO RELEASE MEMORY
    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete([projections_id,reconstruction_id])
    return reconstruction
```

```python
%%time
free = False
if free:
    del projections         
    projections = None
fbp = True
pixel_size = data_set['pixel size']
if fbp:
    reconstruction = fbp_cuda_3d(attn, pixel_size)
```

```python
n = 4
n0 = 840
slice_index_x = 932
slice_index_y = 500
handle = reconstruction[::-1,slice_index_x,:]
height,width = handle.shape
cmap = 'Spectral'
print(width)
print(slice_index_x)
l,h = nif_99to01contrast(handle[np.isfinite(handle)])
#l = -1
#h = 1
#h = 0.3
plt.figure(figsize = (10,20))
ax0 = plt.subplot(221)
ax0.imshow(handle, vmin = l, vmax = h, cmap = cmap)
ax1 = plt.subplot(223)
ax1.imshow(reconstruction[::-1,:,slice_index_y], vmin = l, vmax = h, cmap = cmap)
dz = height//(n+1)
linespec = {"color":'k',"linestyle":'--',"linewidth":1}
for j in range(n):
    ax0.plot([0,width-1],[dz*(j+1),dz*(j+1)],**linespec)
    ax1.plot([0,width-1],[dz*(j+1),dz*(j+1)],**linespec)
    a = plt.subplot(n,2,int((j+1)*2))
    temp = reconstruction[dz*(j+1),:,:]
    #l,h = nif_99to01contrast(temp)
    a.imshow(temp.astype(np.float32).T,cmap = cmap, vmin = l, vmax = h)
    #a.plot([0,width-1],[slice_index_x,slice_index_x],**linespec)
    a.plot([slice_index_x,slice_index_x],[0,width-1],**linespec)
    #a.plot([slice_index_y,slice_index_y],[0,width-1],**linespec)
    a.plot([0,width-1],[slice_index_y,slice_index_y],**linespec)
a.imshow(reconstruction[1051], vmin = l, vmax = h)
```

```python
%%time
to_disk = True
if to_disk:
    write_dir = f"D:\\Data\\Reconstructions\\EXPERIMENTATION\\{data_set['Name']}"
    write_volume(reconstruction,data_set,write_dir,data_set['Name'])
```

---
---
---


# Scratch Work

```python
print(crop_patch[1]-crop_patch[0])
print(crop_patch[3]-crop_patch[2])
```

```python
#print(reconstruction.shape)
data_set
```

```python
if False:
    dump_path = "D:\\Data\\serialized_data\\cone_beam_prep\\{}.p".format(data_set['Name'])
    pickle.dump(attn,open(dump_path,'wb'))
```

```python
del projections
```

```python
plt.figure()
plt.imshow(attn[:,100,:])
```

```python
data_name = data_set['Name']
file_path = "D:\\Data\\sinogram_binaries"
file_name = f'{file_path}\\{data_name}.npy'
log_file = f'{file_path}\\{data_name}.log'
np.save(file_name,attn)
with open(log_file,'w') as f:
    for key,val in data_set.items():
        f.write(f"{key} : {val}\n")
```

```python

```
