import astra
import numpy as np

def astra_2d_simple(sinogram, algorithm = 'FBP_CUDA', pixel_size = 0.0087):
    n_projections,detector_width = sinogram.shape
    vol_geom = astra.create_vol_geom(detector_width,detector_width)
    angles = np.linspace(0,2*np.pi,n_projections, endpoint = False)
    proj_geom = astra.create_proj_geom('parallel',1.0, detector_width,angles)
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    reconstruction_id = astra.data2d.create('-vol',vol_geom)
    algorithm = 'FBP_CUDA'
    cfg = astra.astra_dict(algorithm)
    cfg['ReconstructionDataId'] = reconstruction_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = {'filtertype':'Ram-Lak'}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    reconstruction = astra.data2d.get(reconstruction_id)
    reconstruction/=pixel_size
    astra.data2d.delete([sino_id,reconstruction_id])
    astra.algorithm.delete(alg_id)
    return reconstruction