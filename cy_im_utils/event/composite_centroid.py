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


def composite_centroid_var(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Like composite_centroid, but weights are derived from the 'var' column in each
    DataFrame using inverse-variance weighting: the tighter (lower-variance) detection
    pulls the composite centroid toward itself.

    For a matched pair (i from df1, j from df2):
        w1 = var2_j / (var1_i + var2_j)   — weight on df1 point
        w2 = var1_i / (var1_i + var2_j)   — weight on df2 point
        cx = w1 * x1 + w2 * x2

    Parameters
    ----------
    df1, df2  : DataFrames with columns ["frame", "x", "y", "var"]
    threshold : max Euclidean distance for two points to be considered the same particle

    Returns
    -------
    DataFrame with columns ["frame", "x", "y", "var"] — one row per matched pair per frame.
    The output variance is the harmonic-mean-like combined variance: 1/(1/var1 + 1/var2).
    """
    records = []

    shared_frames = set(df1["frame"].unique()) & set(df2["frame"].unique())

    for frame_id in tqdm(sorted(shared_frames)):
        g1 = df1[df1["frame"] == frame_id][["x", "y", "var"]].reset_index(drop=True)
        g2 = df2[df2["frame"] == frame_id][["x", "y", "var"]].reset_index(drop=True)

        if g1.empty or g2.empty:
            continue

        pts1 = g1[["x", "y"]].to_numpy()
        pts2 = g2[["x", "y"]].to_numpy()
        var1 = g1["var"].to_numpy()
        var2 = g2["var"].to_numpy()

        D = cdist(pts1, pts2)

        nn1_to_2 = D.argmin(axis=1)
        nn2_to_1 = D.argmin(axis=0)

        for i, j in enumerate(nn1_to_2):
            if nn2_to_1[j] == i and D[i, j] <= threshold:
                v1, v2 = var1[i], var2[j]
                total = v1 + v2
                w1 = v2 / total  # lower variance → higher weight
                w2 = v1 / total
                cx = w1 * pts1[i, 0] + w2 * pts2[j, 0]
                cy = w1 * pts1[i, 1] + w2 * pts2[j, 1]
                combined_var = 1.0 / (1.0 / v1 + 1.0 / v2)
                records.append({"frame": frame_id, "x": cx, "y": cy, "var": combined_var})

    return pd.DataFrame(records, columns=["frame", "x", "y", "var"])
