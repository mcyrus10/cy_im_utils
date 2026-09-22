import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy import ndimage

def add_points_for_affine(napari_instance, matches, handle_1: str, handle_2: str):
    """
    This is a bit hacky with some bad time complexity, but it is for building
    the points layer for the affine matrix to fit. once you run this you can add
    it to the viewer and use "fit affine" to transform from the original
    coordinate system to the new one....

    I haven't tested if this works differently from using "refine" but it seems
    like sometimes when I use "refine" and it is already close, then some
    outliers tend to dominate since the transform doesn't need to shift very far
    to accomodate most of the data....

    after you use "estimate matched pairs" you should have the matches attribute
    in a napari instance...
    """
    pts = []
    for a_, b_ in tqdm(matches):
        for a in a_:
            for part_1, df_1 in napari_instance.track_holder[handle_1].groupby("particle"):
                if part_1 != a:
                    continue
                for b in b_:
                    for part_2, df_2 in napari_instance.track_holder[handle_2].groupby("particle"):
                        if part_2 != b:
                            continue
                        mutual = np.intersect1d(df_1['frame'].values,
                                                df_2['frame'].values,
                                                return_indices = True)
                        if len(mutual) == 0:
                            print(f"error with {a},{b}")
                            continue
                        pts_1 = df_1[['frame','y','x']].values[mutual[1]]
                        pts_2 = df_2[['frame','y','x']].values[mutual[2]]
                        # This does the same as  iterating over a zip of these two arrays
                        zipped = np.stack([pts_1, pts_2], axis = 1).reshape(-1,3)
                        if len(zipped) == 0:
                            continue
                        pts.append(zipped)
    pts = np.vstack(pts)
    return pts


def form_modalities_image_array(x0: int, y0: int, slice_size: int, napari_inst, px_size: float, micron_scale_bar: float, fontsize: float = 16):
    """
    Helper function for making the subplots of all the modalities. Might just be
    useful as a template since all the modalities might not be able to use the
    same tracking settings...

    Parameters
    ==========
        x0, y0, slice_size: define the edges and width of the square region to plot
        px_size: micron per pixel
        micron_scale_bar: how many microns the scale bar should show

    """
    size_pixels = micron_scale_bar / px_size
    fontprops = fm.FontProperties(size = fontsize)

    fig,ax = plt.subplots(1,2, sharex = True, sharey = True, figsize = (8,4))
    for j, elem in enumerate(napari_inst.viewer.layers):
        print(elem.name)
        name = elem.name
        elem.visible = True
        if name == "neural_network":
            napari_inst._preview_track_centroids_crocker_grier_()(name, minmass = 1, maxsize = -1, percentile = -1, separation = -1, diameter = 21, threshold = 0.1, preprocess = True)
        else:
            napari_inst._preview_track_centroids_crocker_grier_()(name, minmass = 0.1, maxsize = -1, percentile = -1, separation = -1, diameter = 21, threshold = 0.001, preprocess = True)
        napari_inst.__remove_layer_string_match__("Diameter Mask")

        pts = napari_inst.viewer.layers[-1].data
        im_0 = napari_inst.viewer.export_figure()
        elem.visible = False
        napari_inst.__fetch_layer__("event").visible = True
        im_1 = napari_inst.viewer.export_figure()
        napari_inst.__fetch_layer__("event").visible = False
        for a in ax:
            a.cla()
            a.axis(False)
        ax[0].imshow(im_0)
        ax[1].imshow(im_1)
        colors = 'wk'
        for q, a in enumerate(ax):
            a.plot(*pts[:,::-1].T, marker = "X", markerfacecolor = 'w', color = 'k', linestyle = '', markersize = 10)
            asb = AnchoredSizeBar(a.transData,
                                  size_pixels, f"{micron_scale_bar} $\mu$m", 
                                  loc = "lower left", pad = 0.1, borderpad = 0.1, 
                                  sep = 2, frameon = False, size_vertical = 1,
                                  color = colors[q], fontproperties= fontprops)
            a.add_artist(asb)
            a.axis(False)
        ax[0].set(xlim = (y0, y0+slice_size), ylim = (x0, x0+slice_size))
        if j == 0:
            fig.tight_layout()
        fig.savefig(f"{name}_modality.png", dpi = 150)



class pair_tracks_by_correlation:
    def __init__(self):
        pass

    def calc_global_pairs(self, tracks_1: pd.DataFrame, tracks_2: pd.DataFrame, 
                          spatial_thresh: float = 1.0, corr_thresh: float = 200, 
                          offset: int = 0, plot: bool = False, 
                          min_mutual: int = 15):
        """
        This is a helper funciton for registering two datasets with zero a
        priori information about the data, scale and rotation invariant

        note -> it assumes the data are temporally synchronized +/- offset

        it takes two sets of tracks and calculates the correlation between each displacement of each particle 

        Steps:
            - convert dataframes into dictionaries for faster read
            - iterate over event particles:
                - calculate displacement
                - iterate over frame particles
                    - calculate mutual frames
                    - if no mutual frames -> continue
                    - calculate displacement for frame camera on mutual frames
                    - calculate correlation
                    - if correlation > previous maximum -> estimate affine matrix and calculate avg error
                - if avg error of max correlation < thresh -> add to global matching points

        parameters:
        -----------
            - tracks_{1,2} dataframes with tracks for event and frame respectively
            - spatial_thresh: float - average spatial error must be below this to be considered as a "true" match
            - corr_thresh: float - correlation must exceed this value to be considered a potential "true" match
            - offset: int - triggers for PCO are often offset by 1...? not sure hwat that means...?
            - plot: bool - plot all the valid correlated tracks?
            - min_mutual - minimum number of mutual tracks to be considered valid...

        """
        trx_0 = {pid:{'x':df['x'].values, 'y':df['y'].values, 'frame':df['frame'].values} for pid, df in tracks_1.groupby("particle")}
        trx_1 = {pid:{'x':df['x'].values, 'y':df['y'].values, 'frame':df['frame'].values+offset} for pid, df in tracks_2.groupby("particle")}

        global_pairs = []

        for key,val in tqdm(trx_0.items()):
            dx = np.diff(val['x'])
            dy = np.diff(val['y'])
            dr = np.sqrt(dx**2+dy**2)
            frames_ev = val['frame'][:-1]
            corr = [0]
            min_err = np.inf
            if plot:
                _,ax = plt.subplots(1,2)
            for key2, val2 in trx_1.items():
                frames_fr = val2['frame'][:-1]
                overlap = np.intersect1d(frames_ev, frames_fr, return_indices = True)
                if len(overlap[0]) <= min_mutual:
                    continue
                dx_fr = np.diff(val2['x'])
                dy_fr = np.diff(val2['y'])
                dr_fr = np.sqrt(dx_fr**2 + dy_fr**2)
                corr_local = np.correlate(dr[overlap[1]], dr_fr[overlap[2]])[0]
                if corr_local > np.max(corr) and corr_local > corr_thresh:
                    corr.append(corr_local)
                    x_ev = val['x'][overlap[1]]
                    y_ev = val['y'][overlap[1]]
                    x_fr = val2['x'][overlap[2]]
                    y_fr = val2['y'][overlap[2]]

                    X = np.vstack([x_ev, y_ev, np.ones_like(x_ev)]).T
                    Y = np.vstack([x_fr, y_fr, np.ones_like(y_ev)]).T

                    tform_mat = np.linalg.lstsq(X,Y)[0]
                    tform_coords = np.dot(X,tform_mat).T
                    tform_x, tform_y = tform_coords[:2]
                    dx_tform = tform_x - x_fr
                    dy_tform = tform_y - y_fr
                    dr_tform = np.sqrt(dx_tform**2+dy_tform**2)
                    avg_err = np.mean(dr_tform)
                    if avg_err < min_err:
                        min_err = avg_err.copy()
                        pts = np.vstack([x_ev, y_ev, x_fr, y_fr]).T
                        if plot:
                            ax[0].cla()
                            ax[1].cla()
                            ax[0].plot(dr[overlap[1]], dr_fr[overlap[2]], marker = '.', linestyle = '')
                            ax[1].plot(x_fr-x_fr[0], y_fr- y_fr[0])
                            ax[1].plot(tform_x-tform_x[0], tform_y-tform_y[0])
                            title = f"correlation = {corr_local}\n{key} - {key2}\n{overlap[0][0]}-{overlap[0][-1]}"
                            ax[0].set_title(title)
                            ax[1].set_title(f"mean err: {avg_err:0.2f}")
            if min_err < spatial_thresh:
                print(f"key: {min_err:0.2f}, {thresh}")
                global_pairs.append(pts)
        return np.vstack(global_pairs)

    def calc_affine(self, global_pairs):
        """
        global pairs is an n x 4 array where the first two columns are the event
        coordinate system and the second two are the frame. This function
        estimates the homogeneous affine transform 
        """
        X, Y = np.vstack(global_pairs)[:,:2], np.vstack(global_pairs)[:,2:]
        X = np.hstack([X, np.ones(len(X))[:,None]])
        Y = np.hstack([Y, np.ones(len(Y))[:,None]])
        tform_mat = np.linalg.lstsq(X,Y)[0]
        affine_mat = self.remap_affine(tform_mat.T)
        print("[INFO] returning push affine matrix (apply inv for pull)")
        self.affine_mat = np.linalg.inv(affine_mat)
        return self.affine_mat

    def remap_affine(self, affine): 
        """
        the output from the least squares has the wrong order for the napari
        representation....which requires flipping the diagonal and off diagonal
        for the first 2 rows/cols. This also enforces the homogenous bottom row
        [0,0,1], which can turn to floats from the least squares calculation...
        """
        return np.array([[affine[1,1],affine[1,0],affine[1,2]], 
                         [affine[0,1],affine[0,0],affine[0,2]], 
                         [0, 0, 1]]) 

    def demo_tform(self, image, ref_im_shape): 
        """
        wrapper to call affine transform on an image
        """
        return ndimage.affine_transform(
            image,
            self.affine_mat,
            output_shape = ref_im_shape)
