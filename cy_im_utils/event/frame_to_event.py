from cupyx.scipy import ndimage
from numba import njit
from tqdm import tqdm
import cupy as cp
import numpy as np
import os
import subprocess 
from multiprocessing import Pool

@njit
def csv_ops(colorized: np.array,
             csv_rep: np.array,
             time_0: float,
             dt: float) -> None:
    """
    Converts frame based image into a 4-column array that can be converted to
    csv and encoded for .raw file format that the event camera can read

    It randomly changes the timestamp inside the time0-time0+dt window so the
    temporal indices can be "sub-sampled"

    Parameters:
    -----------
        - colorized: np.array [nx,ny,3] - 
        - csv_rep: np.array [nx*ny,4] - in place array that is filled with
              events; Note that this requires another operation to remove all
              the zeros!
        - time_0: float
        - dt: float

    Returns:
    --------
        None
    """
    nx,ny = colorized.shape[:2]
    zero_arr = np.array([255,255,255]).astype(np.uint8)
    for i in range(nx):
        for j in range(ny):
            temp = colorized[i,j] == zero_arr
            #timestamp = time_0
            timestamp = np.random.uniform(time_0, time_0+dt)
            if temp.sum() != 3:
                lindex = i*nx+j
                if temp[0] and not temp[-1]:
                    csv_rep[lindex] = [i,j,1,timestamp]
                elif not temp[0] and temp[-1]:
                    csv_rep[lindex] = [i,j,0,timestamp]


class im_stack_to_raw:
    def __init__(self, image_stack: np.array) -> None:
        self.im_stack = image_stack
        self.nz, self.nx, self.ny = image_stack.shape

    def _colorize_emulate_(self, 
                         im1: np.array,
                         im0: np.array,
                         thresh: float,
                         blur_bool: bool = False,
                         blur_size: int = 5,
                         speckle_noise: int = 200,
                         negative_polarity_arr = cp.array([200,126,64], dtype = np.uint8),
                         positive_polarity_arr = cp.array([255,255,255], dtype = np.uint8),
                         void_arr = cp.array([52,37,30], dtype = np.uint8),
                         ) -> np.array:
        """
        This basically does what the event emulation library does on GPU 

        Parameters:
        -----------
            im1: np.array - image at time + dt
            im0: np.array - image at time
            thresh: float - this bounds the +/- change that triggers a positive /
                neutral / negative polarity event
            blur_bool: bool - applies a blur to the images to reduce differential
                noise
            blur_size: int - size of blurring kernel

        Returns:
        --------
            np.array - emulated frame  (uint8)

        """
        nx,ny = self.nx, self.ny
        im_0, im_1 = cp.array(im0, dtype = cp.int32), cp.array(im1, dtype = cp.int32)
        # This mirrors the cv2.blur function
        if blur_bool:
            blur_kern = cp.ones([blur_size, blur_size]) / (blur_size**2)
            im_0 = ndimage.convolve(im_0, blur_kern)
            im_1 = ndimage.convolve(im_1, blur_kern)
        diff = im_1-im_0
        pos = (diff - thresh) > 0 
        neg = (diff + thresh) < 0
        output = cp.full([nx,ny,3], void_arr, dtype = cp.uint8)
        #output[:,:,:] = void_arr
        print(neg.shape, output.shape)
        output[neg] = negative_polarity_arr[None,None,:]
        output[pos] = positive_polarity_arr[None,None,:]
        speckle_noise_indices_x = cp.random.uniform(0,nx,speckle_noise).astype(int)
        speckle_noise_indices_y = cp.random.uniform(0,ny,speckle_noise).astype(int)
        speckle_slice = (speckle_noise_indices_x, speckle_noise_indices_y,None)
        output[speckle_slice] = positive_polarity_arr
        return output.get()

    def write_csv(self,
                  file_name: str,
                  fps: int,
                  thresh: float,
                  lambda_poisson: float = 10_000_000,
                  blur_bool: bool = False,
                  blur_size = 5,
                  speckle_noise: int = 0
                  ) -> np.array:
        """
        Converts image stack to csv file representation
        """
        images = self.im_stack
        nz = self.nz
        freq = np.round((1/fps)*1e6).astype(np.uint16)
        csv_tot = []
        desc = 'converting images to csv events'
        for j in tqdm(range(nz-1), desc = desc):
            im1 = self.im_stack[j+1]
            im0 = self.im_stack[j]
            colorized = self._colorize_emulate_(
                    im1,
                    im0,
                    thresh,
                    blur_bool,
                    blur_size = blur_size,
                    speckle_noise = speckle_noise
                    )
            csv_rep = np.zeros([colorized.size, 4], dtype = np.int64)
            csv_ops(colorized, csv_rep, j*freq, freq)
            nonzero_indices = csv_rep.sum(axis = 1) != 0
            csv_rep = csv_rep[nonzero_indices]
            csv_rep[:,3] = np.round((csv_rep[:,3]*lambda_poisson)/lambda_poisson).astype(int)
            index_array = np.argsort(csv_rep[:,3])
            csv_tot.append(csv_rep[index_array])
        csv_tot = np.vstack(csv_tot)
        np.savetxt(file_name, csv_tot, fmt='%d', delimiter = ',')
        print(f"Saved CSV to {file_name}")

    def csv_to_raw(self, output_name, csv_name) -> None:
        """
        This calls the csv to raw encoder on the 
        """
        converter_path = "/home/mcd4/Documents/Experimentation/metavision_testing/raw_to_filter_to_csv_to_raw/metavision_evt2_raw_file_encoder/metavision_evt2_raw_file_encoder"
        print(f"csv_name: {csv_name}; output_name: {output_name}")
        output_path_full = " " + os.getcwd() + "/" + output_name
        csv_path_full = " " + os.getcwd() + "/" + csv_name
        args = [converter_path, output_path_full, csv_path_full]
        subprocess.run(" ".join(args), shell = True)


def test_raw_converter():
    np.random.seed(42)
    #             nz  nx  ny
    image_dims = (100,200,100)
    images = np.random.rand(np.prod(image_dims)).reshape(image_dims)*10_000
    inst = im_stack_to_raw(images)
    csv_name =  "_test_.csv"
    raw_name = "_test_.raw"
    inst.write_csv(csv_name, 10, 0.000001, True)
    inst.csv_to_raw(raw_name, csv_name)
    print("FYI: test image stack is not 1280 x 720")
    subprocess.run([f"metavision_viewer -i {raw_name}"], shell = True)


if __name__ == "__main__":
    test_raw_converter()
