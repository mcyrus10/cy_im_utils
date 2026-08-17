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


def load_events(f_name: str | os.PathLike, field: str ) -> np.ndarray:
    """
    Read the hdf5 output from a v2e simulation and convert it into structured
    array
    """
    import h5py
    with h5py.File(f_name, "r") as f:
        events_raw = f[field][:]
        events = np.vstack(events_raw)

    dtype = [
            ('t', np.int64), 
            ('x', np.uint16),
            ('y', np.uint16),
            ('p', np.int16),
            ]
    events = np.array([tuple(events[j]) for j in range(events.shape[0])], dtype=dtype)
    return events


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
