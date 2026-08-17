import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


def composite_centroid(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    threshold: float,
    weight: float,
    ) -> pd.DataFrame:
    """
    Match particles across two DataFrames (e.g., positive/negative event-camera polarities)
    using mutual nearest-neighbor pairing per frame, then return their weighted centroid.

    Parameters
    ----------
    df1, df2  : DataFrames with at least columns ["frame", "x", "y"]
    threshold : max Euclidean distance for two points to be considered the same particle
    weight    : 0-1 blend toward df1; composite = weight*df1 + (1-weight)*df2

    Returns
    -------
    DataFrame with columns ["frame", "x", "y"] — one row per matched pair per frame.
    """
    records = []

    # Only process frames that exist in both streams
    shared_frames = set(df1["frame"].unique()) & set(df2["frame"].unique())

    for frame_id in tqdm(sorted(shared_frames)):
        g1 = df1[df1["frame"] == frame_id][["x", "y"]].reset_index(drop=True)
        g2 = df2[df2["frame"] == frame_id][["x", "y"]].reset_index(drop=True)

        if g1.empty or g2.empty:
            continue

        pts1 = g1.to_numpy()  # (n1, 2)
        pts2 = g2.to_numpy()  # (n2, 2)

        D = cdist(pts1, pts2)  # pairwise Euclidean distances, shape (n1, n2)

        nn1_to_2 = D.argmin(axis=1)  # nearest df2 index for each df1 point
        nn2_to_1 = D.argmin(axis=0)  # nearest df1 index for each df2 point

        for i, j in enumerate(nn1_to_2):
            # Accept only mutual nearest neighbors within threshold
            if nn2_to_1[j] == i and D[i, j] <= threshold:
                cx = weight * pts1[i, 0] + (1 - weight) * pts2[j, 0]
                cy = weight * pts1[i, 1] + (1 - weight) * pts2[j, 1]
                records.append({"frame": frame_id, "x": cx, "y": cy})

    return pd.DataFrame(records, columns=["frame", "x", "y"])
