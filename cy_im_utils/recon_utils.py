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
                    angles = None
                    ) -> np.array:
    """
    basic for the AAA_bottom dataset with 0.0087 pixel size 

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
                                pixel_size = 0.0087
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

class iterative_recon_base:
    """
    base class for iterative methods to inherit from with the forward and back
    projections
    """
    def __init__(   self):
        pass

    def ASTRA_back_project_2D(  self,
                                sinogram : np.array,
                                algorithm: str = 'SIRT_CUDA',
                                angles = None,
                                geom = 'parallel',
                                iterations = 1,
                                pixel_size: float = 1.0,
                                seed = 0
                                ) -> np.array:
        """

        Args:
        -----
            sinogram: np.array
                input sinogram to back project (rows are projections, columns are
                detector width)
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
        cfg = astra.astra_dict(algorithm)
        cfg['ReconstructionDataId'] = reconstruction_id
        cfg['ProjectionDataId'] = sino_id
        cfg['option'] = {'FilterType':'ram-lak'}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations = iterations)
        reconstruction = astra.data2d.get(reconstruction_id)
        reconstruction /= pixel_size
        astra.data2d.delete([sino_id,reconstruction_id])
        astra.algorithm.delete(alg_id)
        return reconstruction

    def ASTRA_forward_project_2D(   self,
                                    recon_image : np.array,
                                    angles = None,
                                    geom = 'parallel'
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

class TV_POCS_2D(iterative_recon_base):
    def __init__(self):
        super().__init__()

    def recon(  self, 
                sinogram: np.array,
                num_iter: int = 10,
                ng: int = 5,
                recon_iter: int = 10,
                algorithm: str = 'SIRT_CUDA',
                alpha: float = 0.2,
                alpha_red: float = 0.95,
                beta: float = 1.0,
                beta_red: float = 0.995,
                r_max: float = 0.95,
                eps: float = 2.5,
                pixel_size: float = 0.0087,
                enforce_positivity: bool = True
                ) -> np.array:
        """
        see page 4788 from Sidky et al. 2008 for this algorithm which is
        modified ASD-POCS

        NOTE: during the inner loops the pixel_size needs to be 1 so the
        forward and back projections can modify the signal without changing its
        magnitude

        Args:
        -----
            sinogram: np.array - sinogram to be reconstructed
            num_iter: int - number of iterations of main (outer loop)
            ng: int - number of gradient descents to execute per iteration of
                    main loop
            recon_iter: int - number of iterations of recon method (if
                            iterative)
            algorithm: str - which reconstruction algorithm to use
            alpha: float - this factor controls the strength of the gradient
                            descent steps 
            r_max: float ?
            alpha_red: float - this factor [0-1] controls the decay rate of
                                gradient descent strength
            eps: float ?
            pixel_size: float - detector pixel size
            enforce_positivity: bool - turn negatives into 0s

        Returns:
        --------
            TV regularized reconstruction

        """
        logging.warning("----> DEPRECATED??")
        g = sinogram.copy()
        g0 = sinogram.copy()
        n_proj, detector_width = sinogram.shape
        angles = np.linspace(0,2*np.pi, n_proj, endpoint = False)
        f = self.ASTRA_back_project_2D(g, algorithm = "FBP_CUDA", pixel_size = 1)
        
        for j in range(num_iter):
            f0 = f.copy()
            f += beta*self.ASTRA_back_project_2D(g0-g, algorithm = algorithm, 
                    iterations = recon_iter, pixel_size = 1, seed = 0)
            #f = self.ASTRA_back_project_2D(g, algorithm = algorithm, 
            #        iterations = recon_iter, pixel_size = 1, seed = f)

            # Enforce Positvity
            if enforce_positivity:
                f[f < 0] = 0
            f_res = f.copy()
            g = self.ASTRA_forward_project_2D(f,angles = angles)
            dd = np.linalg.norm((g-g0).ravel(), ord = 1)
            dp = np.linalg.norm((f-f0).ravel(), ord = 1)

            # Gradient Descent
            if j == 0:
                dtvg = alpha * dp
            for _ in range(ng):
                df = TV_grad(f)
                #df /= np.linalg.norm(df.ravel(), ord = 1)
                df /= np.sum(np.abs(df))
                f -= dtvg*np.pad(df,((1,1),(1,1)))
            dg = np.linalg.norm((f-f0).ravel())

            # Modify gradient descent step
            if dg > r_max*dp and dd > eps:
                dtvg *= alpha_red
            beta *= beta_red
            g = self.ASTRA_forward_project_2D(f, angles = angles)

        #---------------------
        # Delete This
        logging.warning("Sidky et al. return f_res not f")
        f_res = f.copy()
        #---------------------
        radial_zero(f_res)
        f_res /= pixel_size
        return f_res,g

    def TV_qmark(  self, 
                sinogram: np.array,
                num_iter: int = 10,
                ng: int = 5,
                recon_iter: int = 10,
                algorithm: str = 'SIRT_CUDA',
                alpha: float = 0.2,
                alpha_red: float = 0.95,
                beta: float = 1.0,
                beta_red: float = 0.995,
                r_max: float = 0.95,
                eps: float = 2.5,
                pixel_size: float = 0.0087,
                enforce_positivity: bool = True,
                debug: bool = False
                ) -> np.array:

        g = sinogram.copy()
        g0 = sinogram.copy()
        n_proj = g.shape[0]
        angles = np.linspace(0, 2*np.pi, n_proj, endpoint = False)
        f = self.ASTRA_back_project_2D(g, algorithm = 'FBP_CUDA', pixel_size = 1)

        for j in tqdm(range(num_iter)):
            f0 = f.copy()
            f += beta*self.ASTRA_back_project_2D(g0-g, algorithm = 'SIRT_CUDA',
                    iterations = recon_iter, pixel_size = 1, seed = 0)
            #------------
            g0 = g.copy()
            #------------
            radial_zero(f)
            
            # Positivity Constraint
            if enforce_positivity:
                neg_idx = f < 0
                f[neg_idx] = 0   
            f_res = f.copy()
            #g = M*f
            g = self.ASTRA_forward_project_2D(f, angles = angles)
            dd = np.linalg.norm(g.ravel()-g0.ravel(), ord = 1)
            dp = np.linalg.norm(f.ravel()-f0.ravel(), ord = 1)
            
            # Gradient Descent
            if j == 0:
                dtvg = alpha * dp
            for _ in range(ng):
                df = TV_grad(f)
                df /= np.sum(np.abs(df))
                f -= dtvg*np.pad(df,((1,1),(1,1)))
            dg = np.sum(np.abs(f-f0))

            # Making gradient step size smaller
            if dg > r_max*dp and dd > eps:
                dtvg *= alpha_red
            beta *= beta_red
            
            g = self.ASTRA_forward_project_2D(f, angles = angles)

        #---------------------
        if debug:
            logging.warning("DEBUGGING -> Sidky et al. return f_res not f")
            f_res = f.copy()
        #---------------------
        radial_zero(f_res)
        f_res /= pixel_size
        return f_res,g

