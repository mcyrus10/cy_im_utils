import h5py
import numpy as np
from tqdm import tqdm
from cy_im_utils.event.integrate_intensity import _integrate_events_wrapper_


def __read_hdf5__(file_name, field_name) -> np.array:
    """
    wrapper for hdf5 reading call field name can be "CD" (which stands for
    contrast detector) or "EXT_TRIGGER"
    """
    with h5py.File(file_name, "r") as f:
        data = f[field_name]['events'][()]
    return data


def __hdf5_to_numpy__(
                    trigger_indices,
                    cd_data, 
                    num_images: int,
                    width: int = 720,
                    height: int = 1280,
                    dtype= np.int16,
                    omit_neg: bool = False,
                    ) -> (np.array, np.array):
    """
    This is for loading in an hdf5 file so that the exposure time of the
    frame camera can be matched to the event signal directly.....
    
    Just use the regular raw -> numpy function if you want a finer frame
    rate sampling since this will load the data with gaps!!!
    (discontinuous event data)
    """
    if num_images == -1:
        print("sampling all triggers")
        n_im = trigger_indices.shape[0]
    else:
        print(f"only taking first {num_images} images (subset of total triggers)")
        n_im = num_images
    image_stack = np.zeros([n_im, width, height], dtype = dtype)
    image_buffer = np.zeros([width, height], dtype = dtype)
    integrator_fun = _integrate_events_wrapper_(dtype)

    for j in tqdm(range(n_im), desc = "reading hdf5"):
        id_0, id_1 = trigger_indices[j]
        image_buffer[:] = 0
        slice_ = slice(id_0, id_1, 1)
        integrator_fun(image_buffer, 
                       cd_data['x'][slice_],
                       cd_data['y'][slice_], 
                       cd_data['p'][slice_].astype(bool),
                       omit_neg
                       )
        image_stack[j] = image_buffer.copy()

    return image_stack
