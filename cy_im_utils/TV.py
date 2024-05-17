"""

    Total Variation functions for calculating the gradient for gradient descent

"""
from numba import njit, cuda, prange
import numpy as np
import cupy as cp



@njit(parallel=True)
def TV_grad(img: np.array, eps: float = 1e-8) -> np.array:
    """
    
    Equation 7 from Accurate image reconstruction from few-views and
    limited-angle data in divergent...
    
    Note: if you don't want to pad the array when doing the gradient descent
    you can just return the entirety of out instead of a slice
    
    Args:
    -----
        img: np.array
            2d array (image)
        eps: float
            this pads the denominator so you don't have division by zero in the
            gradients

    Returns:
    --------
        np.array of the tv gradient
    
    """
    nx,ny = img.shape
    out = np.zeros_like(img)
    for i in prange(1,nx-1):
        for j in prange(1,ny-1):
            n1 = (img[i,j]-img[i-1,j]) + (img[i,j]-img[i,j-1])
            d1 = (eps + (img[i,j] - img[i-1,j])**2 + (img[i,j] - img[i,j-1])**2)**(0.5)
            
            n2 = img[i+1,j]-img[i,j]
            d2 = (eps + (img[i+1,j]-img[i,j])**2 + (img[i+1,j] - img[i+1,j-1])**2)**(0.5)
            
            n3 = img[i,j+1] - img[i,j]
            d3 = (eps + (img[i,j+1] - img[i,j])**2 + (img[i,j+1]-img[i-1,j+1])**2)**(0.5)
            out[i,j] = n1/d1 - n2/d2 - n3/d3
    return out[1:-1,1:-1]


def TV_grad_vectorized(img, eps=1e-8):
    """
    Equation 7 from Sidky et al.

    This is numpy / cupy agnostic

    Args:
    -----
        img: array
            (numpy or cupy; 2D or 3D)
        eps: float
            this pads the denominator so you don't have division by zero in the
            gradients

    """
    nx, ny = img.shape[-2:]

    c_slice = (..., slice(1, nx-1), slice(1, ny-1))       # Center Slice

    l_slice = (..., slice(0, nx-2), slice(1, ny-1))       # Left Slice
    r_slice = (..., slice(2, nx), slice(1, ny-1))         # Right Slice

    b_slice = (..., slice(1, nx-1), slice(2, ny))         # Below Slice
    a_slice = (..., slice(1, nx-1), slice(0, ny-2))       # Above Slice

    ar_slice = (..., slice(2, nx), slice(0, ny-2))        # Above Right Slice
    lb_slice = (..., slice(0, nx-2), slice(2, ny))        # Left Below Slice


    return ((img[c_slice] - img[l_slice] ) + (img[c_slice]-img[a_slice])) / \
           (eps + (img[c_slice] - img[l_slice])**2 + (img[c_slice] - img[a_slice])**2)**(0.5) \
           - (img[r_slice] - img[c_slice]) / \
           (eps + (img[r_slice] - img[c_slice])**2 + (img[r_slice] - img[ar_slice])**2)**(0.5) \
           - (img[b_slice] - img[c_slice]) / \
           (eps + (img[b_slice] - img[c_slice])**2 + (img[b_slice] - img[lb_slice])**2)**(0.5)


@cuda.jit('void(float32[:,:,:],float32[:,:,:], float32)')
def TV_grad_cuda(   img: cp.array,
                    out: cp.array,
                    eps: np.float32 = np.float32(1e-8)
                    ) -> None:
    """

    Equation 7 from Accurate image reconstruction from few-views and
    limited-angle data in divergent...

    Note: if you don't want to pad the array when doing the gradient descent
    you can just return the entirety of out instead of a slice

    Args:
    -----
        img: cp.array
            3d array (image)
        out: cp.array
            3d array that gets populated with the TV gradient
        eps: float
            this pads the denominator so you don't have division by zero in the
            gradients

    Returns:
    --------
        None
    """
    n_im, nx, ny = img.shape
    k, i, j = cuda.grid(3)
    if k < n_im and i < nx and j < ny:
        out[k, i, j] = ((img[k, i, j] - img[k, i-1, j]) + (img[k, i, j] - img[k, i, j-1])) / \
                (eps + (img[k, i, j] - img[k, i-1, j])**2 + (img[k, i, j] - img[k, i, j-1])**2)**(0.5) \
                - (img[k, i+1, j] - img[k, i, j]) / \
                (eps + (img[k, i+1, j] - img[k, i, j])**2 + (img[k, i+1, j] - img[k, i+1, j-1])**2)**(0.5) \
                - (img[k, i, j+1] - img[k, i, j]) / \
                (eps + (img[k, i, j+1] - img[k, i, j])**2 + (img[k, i, j+1] - img[k, i-1, j+1])**2)**(0.5)
