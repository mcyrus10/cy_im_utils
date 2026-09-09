import numpy as np
import pandas as pd
from tqdm import tqdm

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