import astra
import numpy as np
import pathlib
from tqdm import tqdm
import logging
from .TV import TV_grad
from .prep import radial_zero

def astra_2d_simple(sinogram : np.array,
                    algorithm : str = 'FBP_CUDA',
                    pixel_size : float = 0.0087,
                    angles = None,
                    geometry: str = 'parallel',
                    seed: np.array = 0,
                    iterations: int = 1,
                    filters: dict = {'FilterType':'ram-lak'}
                    ) -> np.array:
    """
    basic for the AAA_bottom dataset with 0.0087 pixel size 

    Args:
    -----
        sinogram: np.array - input sinogram to back project (rows are
                             projections, columns are detector width)
        algorithm: str - not sure if this will work any other way than
                         'FBP_CUDA'
        pixel_size: float
        angles: np.array (optional) - angles of the projections
        geometry: str parallel or fanflat?
        seed: seed for iterative recon methods...
        iterations: number of iterations for iterative methods

    Returns:
    --------
        reconstructed image
    """
    assert geometry == 'parallel', "only parallel beam is implemented right now"
    n_projections,detector_width = sinogram.shape
    vol_geom = astra.create_vol_geom(detector_width,detector_width)
    if angles is None:
        angles = np.linspace(0,2*np.pi,n_projections, endpoint = False)
    proj_geom = astra.create_proj_geom(geometry,1.0, detector_width,angles)
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    reconstruction_id = astra.data2d.create('-vol',
                                            vol_geom,
                                            data = seed)
    algorithm = algorithm
    cfg = astra.astra_dict(algorithm)
    cfg['ReconstructionDataId'] = reconstruction_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = filters
    alg_id = astra.algorithm.create(cfg)
    # -----------------------------------------------------
    astra.algorithm.run(alg_id, iterations = iterations)
    # -----------------------------------------------------
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
    detector_pixel_size = data_dict['camera pixel size'] \
                          * data_dict['reproduction ratio']
    center_row = detector_rows/2
    angles = np.linspace(0, 2 * np.pi, num = n_projections, endpoint=False)
    if not reconstruction:
        reconstruction = np.empty([detector_rows,detector_cols,detector_cols], 
                                    dtype = np.float32)
    for j in tqdm(range(detector_rows//batch_size)):
        id0,id1 = j*batch_size, (j+1)*batch_size
        proj_geom = astra.create_proj_geom('cone',
                                            1,
                                            1,
                                            detector_rows,
                                            detector_cols,
                                            angles,
                        (distance_source_origin + distance_origin_detector) \
                                            / detector_pixel_size,
                                            0)
        projections_id = astra.data3d.create('-sino', proj_geom, attn)
        
        new_z = center_row-(id0+id1)/2
        vol_geom = astra.creators.create_vol_geom(  detector_cols,
                                                    detector_cols,
                                                    batch_size)
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
                    angles: np.array = None,
                    seed = 0,
                    ) -> np.array:
    """
    Hopefully this is sufficiently generic to handle arbitrariness...
    
    """
    known_algos = ['FDK_CUDA','SIRT3D_CUDA','CGLS3D_CUDA','FP3D_CUDA','FP3D_CUDA']
    warning = "unknown ASTRA 3D algorithm"
    assert data_dict['recon algorithm'] in known_algos, warning

    detector_rows,n_projections,detector_cols = attn.shape
    distance_source_origin = data_dict['source to origin distance']
    distance_origin_detector = data_dict['origin to detector distance']
    detector_pixel_size = data_dict['camera pixel size'] \
                          * data_dict['reproduction ratio']
    algorithm = data_dict['recon algorithm']
    geometry = data_dict['recon geometry']
    if angles is None:
        angles = np.linspace(0, 2 * np.pi, num = n_projections, endpoint = False)
    #  ---------    PARALLEL BEAM    --------------
    if geometry.lower() == 'parallel':
        logging.info("Using Parallel Beam Geometry")
        proj_geom = astra.create_proj_geom( 'parallel3d',
                                            1,
                                            1,
                                            detector_rows,
                                            detector_cols,
                                            angles)
        projections_id = astra.data3d.create('-sino', proj_geom, attn)

    #  ---------    CONE BEAM    --------------
    elif geometry.lower() == 'cone':
        logging.info("Using Cone Beam Geometry")
        proj_geom = astra.create_proj_geom('cone',
                                            1,
                                            1,
                                            detector_rows,
                                            detector_cols,
                                            angles,
        (distance_source_origin + distance_origin_detector )/detector_pixel_size,
                                            0)
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

def ASTRA_forward_project_2D(   recon_image : np.array,
                                angles = None,
                                geom = 'parallel',
                                ) -> np.array:
    """
    Args:
    -----
        recon_image: np.array
            reconstructed image base
        algorithm: str
            not sure if this will work any other way than 'FBP_CUDA'
        angles: np.array (optional)
            angles of the projections

    Returns:
    --------
        sinogram image
    """
    detector_width = recon_image.shape[0]
    vol_geom = astra.create_vol_geom(detector_width, detector_width)
    proj_geom = astra.create_proj_geom(geom, 1.0, detector_width, angles)
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    _,sinogram = astra.create_sino(recon_image,proj_id)
    astra.data2d.delete(proj_id)
    return sinogram

def astra_back_projec_local_function(   sinogram : np.array,
                                        algorithm : str = 'BP_CUDA',
                                        pixel_size : float = 0.0087,
                                        angles = None,
                                        geom = 'parallel',
                                        iterations = 1,
                                        seed = 0
                                        ) -> np.array:
    """
    This is being used by the iterative reconstruction method(s)

    Args:
    -----
        sinogram: np.array
            input sinogram to back project (rows are projections, columns are
            detector width)
        algorithm: str
            not sure if this will work any other way than 'FBP_CUDA'
        pixel_size: float
        angles: np.array (optional)
            angles of the projections

    Returns:
    --------
        reconstructed image
    """
    n_projections,detector_width = sinogram.shape
    vol_geom = astra.create_vol_geom(detector_width,detector_width)
    if angles is None:
        angles = np.linspace(0,2*np.pi,n_projections, endpoint = False)
    proj_geom = astra.create_proj_geom(geom, 1.0, detector_width, angles)
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    reconstruction_id = astra.data2d.create('-vol',vol_geom, data = seed)
    algorithm = algorithm
    cfg = astra.astra_dict(algorithm)
    cfg['ReconstructionDataId'] = reconstruction_id
    cfg['ProjectionDataId'] = sino_id
    #cfg['option'] = {'FilterType':'ram-lak'}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations = iterations)
    reconstruction = astra.data2d.get(reconstruction_id)
    reconstruction /= pixel_size
    astra.data2d.delete([sino_id,reconstruction_id])
    astra.algorithm.delete(alg_id)
    return reconstruction

def astra_forward_project_local_function(  recon_image : np.array,
                            angles = None,
                            geom = 'parallel',
                            pixel_size = 0.0087
                            ) -> np.array:
    """
    This is being used by the iterative method(s)

    Args:
    -----
        recon_image: np.array
            reconstructed image base
        algorithm: str
            not sure if this will work any other way than 'FBP_CUDA'
        angles: np.array (optional)
            angles of the projections

    Returns:
    --------
        sinogram image
    """
    detector_width = recon_image.shape[0]
    vol_geom = astra.create_vol_geom(detector_width, detector_width)
    proj_geom = astra.create_proj_geom(geom, 1.0, detector_width, angles)
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    _,sinogram = astra.create_sino(recon_image,proj_id)
    astra.data2d.delete(proj_id)
    return sinogram

def TV_POCS(sinogram: np.array,
            algorithm: str = 'SIRT_CUDA',
            beta: float = 1.0,
            beta_red: float = 0.995,
            ng: float = 5,
            alpha: float = 0.2,
            r_max: float = 0.95,
            alpha_red: float = 0.95,
            eps: float = 2.5,
            num_iter: int = 25,
            recon_iter: int = 1,
            enforce_positivity: bool = True,
            iter_thresh: float = 0.005 ,
            pixel_size: float = 0.0087,
            debug: bool = False,
            seed: np.array = None
            ) -> np.array:
    #logging.warning("Remove debugging conditionals for better performance")
    g = sinogram.copy()
    g0 = sinogram.copy()
    n_proj,detector_width = g.shape
    f = np.zeros([detector_width,detector_width], dtype = np.float32)
    if debug: print(f"n_proj = {n_proj}")
    angles = np.linspace(0, 2*np.pi, n_proj, endpoint = False)
    if seed is not None:
        recon_ds = astra_back_projec_local_function(sinogram,
                                                    algorithm = algorithm,
                                                    angles = angles,
                                                    pixel_size = 1,
                                                    seed = seed)
    else:
        recon_ds = astra_back_projec_local_function(sinogram,
                                                    algorithm = 'FBP_CUDA',
                                                    angles = angles,
                                                    pixel_size = 1)
    f = recon_ds.copy()
    recon_downsampled = recon_ds.copy()
    radial_zero(recon_downsampled)
    
    if debug:
        print("starting sum = ",np.sum(f))
        print("starting sino sum = ",np.sum(g))
        print("g0 sum = ",np.sum(g0))
        
    for j in range(num_iter):
    #for j in tqdm(range(num_iter)):
        if debug: print(f"iteration {j} -----------------------------")
        f0 = f.copy()
        f += beta*astra_back_projec_local_function(g0-g, algorithm = algorithm, 
                            iterations = recon_iter, pixel_size = 1, seed = 0)
        radial_zero(f)

        # Positivity Constraint
        if enforce_positivity:
            neg_idx = f < 0
            f[neg_idx] = 0   
        f_res = f.copy()

        # Loop exit condition (page 122 from Sidky et al. 2006)
        condition = np.linalg.norm(f_res.ravel()-f0.ravel(), ord = 1) / np.sum(f0)
        if debug: print("residual = ",condition)
        if condition < iter_thresh:
            print(f"breaking main loop residual = {condition}")
            break

        g_ = astra_forward_project_local_function(f, angles = angles)
        if debug: print("---> g_",np.sum(g_))
        dd_vec = g_.ravel()-g0.ravel()
        dd = np.linalg.norm(dd_vec, ord = 1)
        if debug: print("dd = ",dd)
        dp_vec = f.ravel()-f0.ravel()
        dp = np.linalg.norm(dp_vec, ord = 1)
        if debug: print("dp = ",dp)

        # Gradient Descent
        if j == 0:
            dtvg = alpha * dp

        if debug: print("dtvg = ",dtvg)
        f0 = f.copy()
        for _ in range(ng):
            df = TV_grad(f)
            df /= np.linalg.norm(df.ravel(), ord = 1)
            f -= dtvg*np.pad(df,((1,1),(1,1)))
        dg = np.sum(np.abs(f-f0))
        if debug: print("dg = ",dg, type(dg))

        # Making gradient step size smaller
        if dg > r_max*dp and dd > eps:
            dtvg *= alpha_red
            if debug: print(f"iter {j}: stepping down gradient descent parameter" )
        beta *= beta_red
        if debug: print("beta = ",beta)
        #g = (M*f).reshape(n_proj,detector_width)
        g = astra_forward_project_local_function(f, angles = angles)
        if debug: print("g sum = ",np.sum(g))

    return f_res / pixel_size, g
