#!/home/mcd4/miniconda3/envs/im_proc/bin/python
from cupyx.scipy.ndimage import map_coordinates
from magicgui import magicgui
from skimage.transform import ThinPlateSplineTransform, warp_coords
from tqdm import tqdm
from pickle import dump
from pathlib import Path
import cupy as cp
import napari
import numpy as np


def cp_free_mem() -> None:
    """
    Frees cupy's memory pool...
    """
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()


def fetch_dst_coords() -> np.array:
    """
    Hard coding the dst spline transform coords for the 1280 x 720 checkerboard

    probably will be useful elsewhere..
    """
    y_coords = np.hstack([np.arange(0,800,100),  [720]])
    x_coords = np.hstack([np.arange(0,1300,100) , [1280]])
    mgrid = np.vstack(np.meshgrid(y_coords,x_coords))
    return mgrid.reshape(2,-1).T[:,::-1]


class barrel_dist_corr_gui:
    """
    This is for taking the checkerboard gif and calibrating the pixels for the
    event camera to be 1280 x 720......

    Could mabye automate this with corner finding algorithm....
    """
    def __init__(self):
        self.viewer = napari.Viewer()
        self.viewer.title = "Event-Frame Registration GUI"
        self.pull_affine_matrix = None
        self.affine_matrix = None
        self.frame_0 = 0

        dock_widgets = {
                'Operations': [
                    self._estimate_checkerbaord_(),
                    self._sort_tps_points_(),
                    self._fit_warp_(),
                    self._apply_warp_(),
                    self._write_points_(),
                              ]
                        }
        for key,val in dock_widgets.items():
            self.viewer.window.add_dock_widget(val,
                                               name = key,
                                               add_vertical_stretch = False,
                                               area = 'right'
                                               )
        self.total_shift = 0

    def _apply_warp_(self):
        @magicgui(call_button="Apply Warp To Layer")
        def inner(layer_name: str):
            layer_handle = self.__fetch_layer__(layer_name).data
            if layer_handle.ndim != 4:
                print("image must have 4 dimensions...? (z,x,y,color)")
            for elem in tqdm(layer_handle):
                for j in range(3):
                    nx,ny = elem[:,:,j].shape
                    elem[:,:,j] = map_coordinates(
                        cp.array(elem[:,:,j], dtype = elem.dtype),
                        self.warp_coords,
                        order = 0).reshape(nx,ny).get()
            
        return inner

    def _fit_warp_(self):
        @magicgui(call_button="Fit Warp")
        def inner():
            points_handle = self.__fetch_layer__("Points").data
            if points_handle.ndim == 3:
                src = points_handle[:,1:][:,::-1]
            elif points_handle.ndim == 2:
                src = points_handle[:,::-1]

            im_handle = self.viewer.layers[0].data
            if im_handle.ndim == 4:
                image_shape = self.viewer.layers[0].data.shape[1:3]
            elif im_handle.ndim == 3:
                image_shape = self.viewer.layers[0].data.shape[:2]

            dst = fetch_dst_coords()
            tps = ThinPlateSplineTransform()
            tps.estimate(dst, src)
            self.warp_coords = cp.array(
                    warp_coords(tps, image_shape).reshape(2,-1)
                    )
            print("Finished Fitting Warp Coordinates")
            
        return inner

    def _estimate_checkerbaord_(self):
        @magicgui(call_button="Find Checkerboard")
        def inner(image_layers: str,
                  pattern_x: int,
                  pattern_y: int
                ):
            from cv2 import findChessboardCorners
            image_layers = [int(elem) for elem in image_layers.split(",")]
            image_handle = self.viewer.layers[0].data[image_layers].copy()
            image_handle = np.median(image_handle, axis = 0)
            self.viewer.add_image(image_handle)
            pattern_size = (pattern_x, pattern_y)
            retval, corners = findChessboardCorners(
                    image = image_handle.astype(np.uint8),
                    patternSize = pattern_size)
            self.viewer.add_points(np.squeeze(corners)[:,::-1], name = "Points")
        return inner

    def _sort_tps_points_(self):
        @magicgui(call_button="sort spline points")
        def inner():
            stride = 9
            print(f"---> HARD CODED STRIDE = {stride} <---")
            pts_handle = self.viewer.layers[-1].data.copy()
            second_ax = np.argsort(pts_handle[:,1])
            pts_handle = pts_handle[second_ax]
            n_steps = pts_handle.shape[0] // stride
            for j in range(n_steps):
                slice_ = slice(j*stride, (j+1)*stride)
                sort_arg = np.argsort(pts_handle[slice_,0])
                pts_handle[slice_] = pts_handle[slice_][sort_arg]
            self.viewer.layers[-1].data = pts_handle
            print("Done Sorting Points for Fit")
        return inner


    def _write_points_(self):
        @magicgui(call_button="write transform points")
        def inner(file_name: Path):
            points_handle = self.__fetch_layer__("Points").data
            if points_handle.ndim == 3:
                src = points_handle[:,1:][:,::-1]
            elif points_handle.ndim == 2:
                src = points_handle[:,::-1]
            
            with open(file_name,'wb') as f:
                dump(src, f)
            print(f"finished writing transform to {str(file_name)}")
        return inner
    
    def __fetch_layer__(self, layer_name: str):
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                return layer
        else:
            print(f"Layer {layer_name} not found")
            return None


if __name__ == "__main__":
    inst = barrel_dist_corr_gui()
    napari.run()
