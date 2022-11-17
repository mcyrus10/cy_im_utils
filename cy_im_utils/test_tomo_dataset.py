from matplotlib.patches import Rectangle
from napari_tomo_gui import tomo_dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import unittest


class test_tomo_dataset(unittest.TestCase):
    
    def test_init(self):
        inst.load_transmission_sample()

    def test_fetch_files(self):
        for p in inst.settings['paths']:
            inst.fetch_files(Path(p))

    def test_fetch_combined_image(self):
        inst.fetch_combined_image()

    def test_load_transmission_sample(self):
        inst.load_transmission_sample()

    def test_cropping_norm(self):
        inst.load_transmission_sample()
        im = inst.transmission_sample
        vmin,vmax = np.quantile(im.flatten(),0.1), np.quantile(im.flatten(),0.9)
        fig,ax = plt.subplots()
        ax.imshow(im, vmin = vmin, vmax = vmax, origin = 'lower')
        for i,key in enumerate(['crop','norm']):
            handle = inst.settings[key]
            xy = handle['x0'],handle['y0']
            dx = handle['x1']-handle['x0']
            dy = handle['y1']-handle['y0']
            color = 'r' if i == 0 else 'b'
            rectangle = Rectangle(xy, dx, dy, fill = False, color = color)
            ax.add_artist(rectangle)
        fig.tight_layout()
        print("---> Close Image To Continue <---")
        plt.show()

    def test_load_projections(self):
        inst.load_projections()

if __name__ == "__main__":
    # ------------------------------------------------------------------------
    #   This Instance Will Be Used By test_tomo_dataset
    # ------------------------------------------------------------------------
    config_file = ".__test__config__script__.yml"
    inst = tomo_dataset(config_file)
    unittest.main()
