#------------------------------------------------------------------------------
#                   GPU Batching Functions
#------------------------------------------------------------------------------

from PIL import Image
from astropy.io import fits
from cupyx.scipy.ndimage import median_filter as median_gpu
from tqdm import tqdm
import cupy as cp
import matplotlib.pyplot as plt

def attenuation_gpu_batch(input_arr,ff,df,output_arr,id0,id1,batch_size,norm_patch,
                          crop_patch, theta, kernel = 3, dtype = cp.float32):
    
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
    projection_gpu -= df.reshape(1,height,width)
    projection_gpu /= (ff-df).reshape(1,height,width)
    patch = cp.mean(projection_gpu[:,norm_patch[0]:norm_patch[1],norm_patch[2]:norm_patch[3]], axis = (1,2), dtype = dtype)
    projection_gpu /= patch.reshape(batch_size,1,1)
    projection_gpu = median_gpu(projection_gpu, (1,kernel,kernel))
    projection_gpu = -cp.log(projection_gpu)
    #-----------------------------------------------
    #---      make all non-finite values 0?      ---
    projection_gpu[~cp.isfinite(projection_gpu)] = 0
    #-----------------------------------------------
    output_arr[id0:id1] = cp.asnumpy(projection_gpu[:,crop_patch[0]:crop_patch[1],crop_patch[2]:crop_patch[3]])
    
def fbp_cuda_3d(attn, pixel_size): 
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
    
def ASTRA_GENERIC(attn,geometry = 'cone', algorithm = 'FDK_CUDA', detector_pixel_size = 0.0087, source_origin = 5965, origin_detector = 35):
    
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
    
if __name__=="__main__":
    pass
