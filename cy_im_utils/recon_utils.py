import astra
import numpy as np
import pathlib
from tqdm import tqdm

def astra_2d_simple(sinogram : np.array,
                    algorithm : str = 'FBP_CUDA',
                    pixel_size : float = 0.0087,
                    angles = None
                    ) -> np.array:
    """
    basic for the AAA_bottom dataset with 0.0087 pixel size 
    """
    n_projections,detector_width = sinogram.shape
    vol_geom = astra.create_vol_geom(detector_width,detector_width)
    if not angles:
        angles = np.linspace(0,2*np.pi,n_projections, endpoint = False)
    proj_geom = astra.create_proj_geom('parallel',1.0, detector_width,angles)
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    reconstruction_id = astra.data2d.create('-vol',vol_geom)
    algorithm = algorithm
    cfg = astra.astra_dict(algorithm)
    cfg['ReconstructionDataId'] = reconstruction_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = {'FilterType':'ram-lak'}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    reconstruction = astra.data2d.get(reconstruction_id)
    reconstruction /= pixel_size
    astra.data2d.delete([sino_id,reconstruction_id])
    astra.algorithm.delete(alg_id)
    return reconstruction

def ASTRA_FDK_batch( attn,
                    data_dict : dict,
                    batch_size : int,
                    reconstruction = None,
                    to_disk : bool = False,
                    directory : pathlib.Path = pathlib.Path.cwd()):
    """
    This is For executing the cone beam reconstruction batch-wise so it doesn't
    overload the GPU memory

    Args:
    -----
        dat
    """
    algorithm = "FDK_CUDA"
    detector_rows,n_projections,detector_cols = attn.shape
    distance_source_origin = data_dict['source to origin distance']
    distance_origin_detector = data_dict['origin to detector distance']
    detector_pixel_size = data_dict['camera pixel size']*data_dict['reproduction ratio']
    center_row = detector_rows/2
    angles = np.linspace(0, 2 * np.pi, num = n_projections, endpoint=False)
    if not reconstruction:
        reconstruction = np.empty([detector_rows,detector_cols,detector_cols], dtype = np.float32)
    for j in tqdm(range(detector_rows//batch_size)):
        id0,id1 = j*batch_size, (j+1)*batch_size
        proj_geom = astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
                (distance_source_origin + distance_origin_detector) / detector_pixel_size, 0)
        projections_id = astra.data3d.create('-sino', proj_geom, attn)
        
        #vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, batch_size)
        #max_z = center_row-id0
        #min_z = center_row-id1
        #vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, batch_size, 
        #                      -detector_cols//2, detector_cols//2, -detector_cols//2, detector_cols//2, min_z, max_z)
        new_z = center_row-(id0+id1)/2
        vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, batch_size)
        vol_geom = astra.functions.move_vol_geom(vol_geom,(0,0,-new_z))
        reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
        alg_cfg = astra.astra_dict(algorithm)
        alg_cfg['ProjectionDataId'] = projections_id
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        alg_cfg['option'] = {'FilterType': 'ram-lak'}
        algorithm_id = astra.algorithm.create(alg_cfg)
    
        #-------------------------------------------------
        astra.algorithm.run(algorithm_id)  # This is slow
        #-------------------------------------------------

        reconstruction[id0:id1] = astra.data3d.get(reconstruction_id)

        # DELETE OBJECTS TO RELEASE MEMORY
        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete([projections_id,reconstruction_id])
        
    return reconstruction/detector_pixel_size

def ASTRA_General(  attn: np.array, 
                    data_dict: dict,
                    iterations: int = 1,
                    seed = 0,
                    ) -> np.array:
    """
    Hopefully this is sufficiently generic to handle arbitrariness...
    
    """
    detector_rows,n_projections,detector_cols = attn.shape
    distance_source_origin = data_dict['source to origin distance']
    distance_origin_detector = data_dict['origin to detector distance']
    detector_pixel_size = data_dict['camera pixel size']*data_dict['reproduction ratio']
    algorithm = data_dict['recon algorithm']
    geometry = data_dict['recon geometry']
    angles = np.linspace(0, 2 * np.pi, num = n_projections, endpoint=False)
    #  ---------    PARALLEL BEAM    --------------
    if geometry.lower() == 'parallel':
        proj_geom = astra.create_proj_geom( 'parallel3d',
                                            1,
                                            1,
                                            detector_rows,
                                            detector_cols,
                                            angles)
        projections_id = astra.data3d.create('-sino', proj_geom, attn)

    #  ---------    CONE BEAM    --------------
    elif geometry.lower() == 'cone':
        proj_geom = astra.create_proj_geom('cone',
                                            1,
                                            1,
                                            detector_rows,
                                            detector_cols,
                                            angles,
                                distance_source_origin/detector_pixel_size,
                                distance_origin_detector /detector_pixel_size)
        projections_id = astra.data3d.create('-sino', proj_geom, attn)
        
    vol_geom = astra.creators.create_vol_geom(  detector_cols,
                                                detector_cols,
                                                detector_rows)

    reconstruction_id = astra.data3d.create('-vol',
                                            vol_geom,
                                            data = seed)
    alg_cfg = astra.astra_dict(algorithm)
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    alg_cfg['option'] = {'FilterType': 'ram-lak'}
    algorithm_id = astra.algorithm.create(alg_cfg)
    
    #-------------------------------------------------
    astra.algorithm.run(algorithm_id, iterations = iterations)  # This is slow
    #-------------------------------------------------

    reconstruction = astra.data3d.get(reconstruction_id)
    reconstruction /= detector_pixel_size

    # DELETE OBJECTS TO RELEASE MEMORY
    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(projections_id)
    astra.data3d.delete(reconstruction_id)
    return reconstruction
