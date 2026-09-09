"""Post‑processing utilities for the synthetic event‑camera pipeline.

This module implements a thin wrapper around the :mod:`tonic` library to load
event files, convert them into image frames, and visualise or save the result.
It deliberately imports heavy optional dependencies (``tonic`` and ``opencv``)
only when the corresponding functions are called, allowing the core pipeline
to run on systems without these packages.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import tonic


def load_events(
    f_name: str | os.PathLike,
    n_images: int = -1,
    delta_t: float | None = None,
) -> np.ndarray:
    """
    Read the hdf5 output from a v2e simulation and convert it into structured
    array.

    When *n_images* > 0 and *delta_t* is provided, only the events within the
    first ``n_images * delta_t`` time-units are loaded.  This avoids reading the
    entire dataset into memory when only a temporal subset is needed.
    """
    import h5py

    if n_images > 0 and delta_t is None:
        raise ValueError("delta_t is required when n_images > 0")

    with h5py.File(f_name, "r") as f:
        if n_images > 0:
            timestamps = f["events"][:, 0]
            t_max = timestamps[0] + delta_t * n_images
            idx_max = int(np.searchsorted(timestamps, t_max))
            events = f["events"][:idx_max]
        else:
            events = f["events"][:]

    if events[:,0].max() != events[-1,0]:
        print("[WARNING] final timestamp is not highest value; potential casting error")

    dtype = [
            ('t', np.int64),
            ('x', np.uint16),
            ('y', np.uint16),
            ('p', np.int16),
            ]
    structured = np.empty(events.shape[0], dtype=dtype)
    structured['t'] = events[:, 0]
    structured['x'] = events[:, 1]
    structured['y'] = events[:, 2]
    structured['p'] = events[:, 3]
    return structured


def events_to_frames(
    events: Any,
    height: int,
    width: int,
    delta_t: float,
) -> np.ndarray:
    """Convert events to a stack of frames using ``tonic.transform.ToFrame``.

    Parameters
    ----------
    events:
        A ``tonic.EventStore`` object.
    height, width:
        Desired spatial resolution of the output frames.
    delta_t:
        Temporal window for each frame in micro seconds. 
    """
    # ``tonic.transforms.ToFrame`` can operate on either a tonic.EventStore or a
    # NumPy structured array with fields ``x``, ``y``, ``t`` and ``p``. We therefore
    # skip the ``hasattr`` check and pass the object directly.
    transform = tonic.transforms.ToFrame([height, width, 2], time_window=delta_t)
    frames = transform(events)
    # Collapse polarity dimension into a single signed intensity image for
    # downstream visualisation (positive minus negative).
    signed = frames[:,1] - frames[:, 0]
    return signed.astype(np.float32)
