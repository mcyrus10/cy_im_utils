from matplotlib.patches import Rectangle
from tomo_dataset import tomo_dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import unittest

from sys import path
path.append("C:\\Users\\mcd4\\Documents\\cy_im_utils")
from cy_im_utils.recon_utils import astra_tomo_handler


class test_tomo_dataset(unittest.TestCase):
    """ This tests the different methods of tomo_dataset to make sure that it
    will still work correctly inside a script, etc.
    Note that the "inst_1" referred to in this class exists in the scope of
    __main__ where it is inst_1antiated.
    also note that the transmission images are heavily down-sampled to make
    this run more quickly inst_1ead of having it 
    """
    def test_1_init(self):
        inst_1.load_transmission_sample()

    def test_2_fetch_files(self):
        for p in inst_1.settings['paths']:
            inst_1.fetch_files(Path(p))

    def test_3_fetch_combined_image(self):
        inst_1.fetch_combined_image()

    def test_4_load_transmission_sample(self):
        inst_1.load_transmission_sample()

    def test_5_cropping_norm(self):
        inst_1.load_transmission_sample()
        im = inst_1.transmission_sample
        vmin,vmax = np.quantile(im.flatten(),0.1), np.quantile(im.flatten(),0.9)
        fig,ax = plt.subplots()
        ax.imshow(im, vmin = vmin, vmax = vmax, origin = 'upper')
        for i,key in enumerate(['crop','norm']):
            handle = inst_1.settings[key]
            xy = handle['y0'],handle['x0']
            dx = handle['y1']-handle['y0']
            dy = handle['x1']-handle['x0']
            color = 'r' if i == 0 else 'b'
            rectangle = Rectangle(xy, dx, dy, fill = False, color = color)
            ax.add_artist(rectangle)
        fig.tight_layout()
        print("---> Close Image To Continue <---")
        plt.show()

    def test_6_load_projections(self):
        inst_1.load_projections(truncate_dataset = 25)
        inst_1.transmission = inst_1.transmission[::25]

    def test_7_remove_all_stripe(self):
        inst_1.remove_all_stripe(batch_size = 10)

    def test_8_recon(self):
        inst_1._reconstructor_ = astra_tomo_handler(inst_1.settings['recon'])
        inst_1.reconstruct()

    def test_9_volumetric_median(self):
        med_kernel = (3,3,3)
        print(inst_1.reconstruction.shape)
        inst_1.apply_volumetric_median()


if __name__ == "__main__":
    config_file = ".__test__config__pens__.yml"
    inst_1 = tomo_dataset(config_file)
    unittest.main()
