import astra
import numpy as np
import pathlib
from tqdm import tqdm
from functools import partial
import logging
try:
    from .TV import TV_grad
except:
    print("ERROR LOADING TV ")
from .prep import radial_zero

class astra_tomo_handler:
    """ 
    This is for the napari gui so that it can handle a single 2D reconstruction
    or the full 3D reconstruction for previewing the state
    """
    def __init__(self, settings_dict: dict):
        self.settings = settings_dict
        self.algorithm = self.settings['algorithm']
        self.geometry = self.settings['geometry']
        self.proxy_methods = {
                                'FDK_CUDA':'FBP_CUDA',
                                'SIRT3D_CUDA':'SIRT_CUDA',
                                'CGLS3D_CUDA':"CLGS_CUDA"
                                }

        self.geometries_2d = ['parallel','fanflat']
        self.algorithms_2d = ['FBP_CUDA','SIRT_CUDA','CGLS_CUDA','EM_CUDA','SART_CUDA']

        self.geometries_3d = ['parallel3d','cone']
        self.algorithms_3d = ['FDK_CUDA','SIRT3D_CUDA','CGLS3D_CUDA']

        if 'iterations' in self.settings:
            self.iterations = self.settings['iterations']
        else:
            self.iterations = 1
        self.pixel_size = self.settings['pixel pitch']

        options = {}
        if 'options' in self.settings:
            options = self.settings['options']
        self.options = options

    def _gen_fbp_fdk_seed(self, 
                sinogram,
                angles,
                ndim: int = 2
                ) -> np.array:
        """
        """
        logging.info("Generating FBP Seed")
        seed_settings = self.settings.copy()
        if ndim == 2:
            seed_settings['algorithm'] = 'FBP_CUDA' 
            seed_settings['geometry'] = 'parallel' 
        elif ndim == 3:
            seed_settings['algorithm'] = 'FDP_CUDA' 
            seed_settings['geometry'] = 'parallel3d' 

        seed_settings['options']['FilterType'] = 'ram-lak'
        opt_handle = self.settings['options']
        gpuindex = 0 if 'GPUindex' not in opt_handle else opt_handle['GPUindex']
        seed_settings['options']['GPUindex'] = gpuindex
        seed_settings['fbp_fdk_seed'] = False
        return self.astra_reconstruct_2D(   sinogram,
                                            angles,
                                            settings = seed_settings)

    def _parse_settings_2D( self,
                            sinogram,
                            settings,
                            detector_width,
                            angles):
        """ This function parses the settings dict so that the call to
        reconstruct is more streamlined
        """

        # Re-Assign Algorithm (if needed)
        algorithm = settings['algorithm']
        if algorithm in self.proxy_methods:
            algorithm = self.proxy_methods[algorithm]
            logging.info(f"Using {algorithm} for 2D Parallel")
        
        # create proj_args  (for 2D scenario)
        geometry = settings['geometry']
        if geometry in ['parallel','parallel3d']:
            proj_args = ['parallel',1.0, detector_width,angles]
        elif geometry in ['cone','fanflat']:
            source_detector = settings['source to origin distance']
            origin_detector = settings['origin to detector distance']
            proj_args = ['fanflat',1.0, detector_width,angles,
                (source_detector + origin_detector)/self.pixel_size, 0]
        else:
            assert False, f"Unkonwn geometry {geometry}"

        # Seed operations
        seed = 0
        if 'fbp_fdk_seed' in settings:
            if settings['fbp_fdk_seed']:
                seed = self._gen_fbp_fdk_seed(sinogram, angles, ndim = 2)

        return algorithm, geometry, proj_args, seed

    def _parse_settings_3D(    self,
                            sinogram,
                            settings,
                            detector_rows,
                            detector_cols,
                            angles):
        """ This function parses the settings dict so that the call to
        reconstruct is more streamlined
        """
        # create proj_args  (for 2D scenario)
        geometry = settings['geometry']
        if geometry == 'parallel3d':
            proj_args = [   geometry,
                            1.0,
                            1.0,
                            detector_rows,
                            detector_cols,
                            angles]
        elif geometry == 'cone':
            source_detector = settings['source to origin distance']
            origin_detector = settings['origin to detector distance']
            proj_args = [   geometry,
                            1.0,
                            1.0,
                            detector_rows,
                            detector_cols,
                            angles,
                            (source_detector + origin_detector)/self.pixel_size,
                            0
                            ]
        else:
            assert False, "Only cone and parallel3D have been implemented"


        # Seed operations
        seed = 0
        if 'fbp_fdk_seed' in settings:
            if settings['fbp_fdk_seed']:
                seed = self._gen_fbp_fdk_seed(sinogram, angles, ndim = 3)

        return proj_args,  seed

    def astra_reconstruct_2D(self,
                            sinogram: np.array,
                            angles: np.array,
                            settings = None
                            ) -> np.array:
        """
        """
        if settings is None:
            settings = self.settings

        n_projections, detector_width = sinogram.shape
        vol_geom = astra.create_vol_geom(detector_width,detector_width)

        algorithm,geometry,proj_args,seed = self._parse_settings_2D(
                                                            sinogram,
                                                            settings,
                                                            detector_width,
                                                            angles
                                                            )
        proj_geom = astra.create_proj_geom(*proj_args)
        sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
        reconstruction_id = astra.data2d.create('-vol',
                                                vol_geom,
                                                data = seed
                                                )
        cfg = astra.astra_dict(algorithm)
        cfg['ReconstructionDataId'] = reconstruction_id
        cfg['ProjectionDataId'] = sino_id
        cfg['option'] = {} if 'options' not in settings else settings['options']
        alg_id = astra.algorithm.create(cfg)
        # -----------------------------------------------------
        astra.algorithm.run(alg_id, iterations = self.iterations)
        # -----------------------------------------------------
        reconstruction = astra.data2d.get(reconstruction_id)
        reconstruction /= self.pixel_size
        astra.data2d.delete([sino_id,reconstruction_id])
        astra.algorithm.delete(alg_id)
        return reconstruction

    def astra_reconstruct_3D(self,
                            sinogram: np.array,
                            angles: np.array,
                            settings = None
                            ) -> np.array:
        """
        """
        if settings is None:
            settings = self.settings
        n_sino, n_projections, detector_width = sinogram.shape
        detector_rows,detector_cols = n_sino,detector_width
        vol_geom = astra.creators.create_vol_geom(  detector_cols,
                                                    detector_cols,
                                                    detector_rows)

        algorithm = self.algorithm
        geometry = self.geometry
        proj_args,seed = self._parse_settings_3D(   sinogram,
                                                    settings,
                                                    detector_rows,
                                                    detector_width,
                                                    angles
                                                    )
        proj_geom = astra.create_proj_geom(*proj_args)
        sino_id = astra.data3d.create('-sino', proj_geom, sinogram)
        reconstruction_id = astra.data3d.create('-vol',
                                                vol_geom,
                                                data = seed
                                                )
        cfg = astra.astra_dict(algorithm)
        cfg['ReconstructionDataId'] = reconstruction_id
        cfg['ProjectionDataId'] = sino_id
        cfg['option'] = self.options
        alg_id = astra.algorithm.create(cfg)
        # -----------------------------------------------------
        astra.algorithm.run(alg_id, iterations = self.iterations)
        # -----------------------------------------------------
        reconstruction = astra.data3d.get(reconstruction_id)
        reconstruction /= self.pixel_size
        astra.data3d.delete([sino_id,reconstruction_id])
        astra.algorithm.delete(alg_id)
        return reconstruction

    def reconstruct_volume( self,
                            sinogram_volume,
                            angles
                            ) -> np.array:
        nsino,n_proj,detector_width = sinogram_volume.shape
        if self.geometry in self.geometries_2d and \
                self.algorithm in self.algorithms_2d:
            recon = np.zeros([nsino,detector_width,detector_width], 
                                                        dtype = np.float32)
            for j in tqdm(range(nsino), desc = 'reconstrucing volume'):
                recon[j] = self.astra_reconstruct_2D(sinogram_volume[j],
                                                angles = angles,
                                                settings = self.settings)
        elif self.geometry in self.geometries_3d and \
                self.algorithm in self.algorithms_3d:
            recon = self.astra_reconstruct_3D(  sinogram_volume, 
                                                angles,
                                                settings = self.settings)
        else:
            assert False, f"{self.geometry} incompatible wiht {self.algorithm}"

        return recon

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
                                            data = seed
                                            )
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
                    cfg_options: dict = {'FilterType':'ram-lak'},
                    seed = 0,
                    ) -> np.array:
    """
    Hopefully this is sufficiently generic to handle arbitrariness...
    
    """
    logging.warning("This funciton is deprecated, use astra_tomo_handler")
    known_algos = ['FDK_CUDA','SIRT3D_CUDA','CGLS3D_CUDA','FP3D_CUDA','FP3D_CUDA']
    warning = "unknown ASTRA 3D algorithm"
    assert data_dict['recon algorithm'] in known_algos, warning

    detector_rows,n_projections,detector_cols = attn.shape
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
        distance_source_origin = data_dict['source to origin distance']
        distance_origin_detector = data_dict['origin to detector distance']
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
    alg_cfg['option'] = cfg_options
    algorithm_id = astra.algorithm.create(alg_cfg)
    
    #----------------------------------------------------------------------
    astra.algorithm.run(algorithm_id, iterations = iterations)
    #----------------------------------------------------------------------

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
    astra.projector.delete(proj_id)
    return sinogram

def astra_back_project_local_function(sinogram : np.array,
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
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations = iterations)
    reconstruction = astra.data2d.get(reconstruction_id)
    reconstruction /= pixel_size
    astra.data2d.delete([sino_id,reconstruction_id])
    astra.algorithm.delete(alg_id)
    return reconstruction

def astra_forward_project_local_function(recon_image: np.array,
                                         angles=None,
                                         geom='parallel',
                                         pixel_size=0.0087
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
    astra.projector.delete(proj_id)
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
            enforce_positivity: bool = False,
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
        recon_ds = astra_back_project_local_function(sinogram,
                                                    algorithm = algorithm,
                                                    angles = angles,
                                                    pixel_size = 1,
                                                    seed = seed)
    #else:
    #    recon_ds = astra_back_project_local_function(sinogram,
    #                                                algorithm = 'FBP_CUDA',
    #                                                angles = angles,
    #                                                pixel_size = 1)
    else: 
        recon_ds = f.copy()

    f = recon_ds.copy()
    recon_downsampled = recon_ds.copy()
    radial_zero(recon_downsampled)
    g = astra_forward_project_local_function(f, angles = angles)
    
    if debug:
        print("starting sum = ",np.sum(f))
        print("starting sino sum = ",np.sum(g))
        print("g0 sum = ",np.sum(g0))
        
    for j in range(num_iter):
    #for j in tqdm(range(num_iter)):
        if debug: print(f"iteration {j} -----------------------------")
        f0 = f.copy()
        f += beta*astra_back_project_local_function(g0-g, algorithm = algorithm, 
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
    del f0,g_,g
    return f_res / pixel_size
