# WRITTEN BY AARON GOLDFAIN...?

import trackpy as tp # conda install -c conda-forge trackpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile
from math import floor
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)
from scipy.optimize import curve_fit
from os.path import join, basename, dirname, isdir, exists
from os import mkdir

# import bayesian diffusion estimation package
# installation instructions
# add git to anaconda: <conda install git>
# download diffusive_distinguishability repository <pip install git+https://github.com/AllenCellModeling/diffusive_distinguishability.git>
# find file Anaconda3\envs\DFTIRMenv\Lib\site-packages\diffusive_distinguishability\ndim_homogeneous_distinguishability.py
# change line 8 of this file to <import diffusive_distinguishability.fbm_analysis as fa>
import diffusive_distinguishability.ndim_homogeneous_distinguishability as hd


def A_to_r(A, eta, T):
    '''
    converts MSD vs. lagtime slope to hydrodynamic radius
    A = MSD vs. lagtime slope (m^2/s)
    eta = viscosity (Pa*s)
    T = temperature (deg C)
    '''
    kB = 1.38e-23
    D = A/4 #for n-dimensional tracks, D=A/(2*n)
    return kB*(T+273.15)/(D*6*np.pi*eta)


def A_unc_to_r_unc(A, eta, T, A_unc):
    '''
    converts MSD vs. lagtime slope to hydrodynamic radius
    A = MSD vs. lagtime slope (m^2/s)
    eta = viscosity (Pa*s)
    T = temperature (deg C)
    A_unc = uncertainty in A
    '''
    kB = 1.38e-23
    D = A/4 #for n-dimensional tracks, D=A/(2*n)
    D_unc = A_unc/4 
    return kB*(T+273.15)/(6*np.pi*eta)*D_unc/D**2


def D_to_r(D, eta, T):
    '''
    converts diffusion coefficient to hydrodynamic radius
    D = diffusion coefficient (m^2/s)
    eta = viscosity (Pa*s)
    T = temperature (deg C)
    '''
    kB = 1.38e-23
    return kB*(T+273.15)/(D*6*np.pi*eta)


def D_unc_to_r_unc(D, eta, T, D_unc):
    '''
    converts uncertainty in diffusion  coefficient to uncertainty in hydrodynamic radius
    D = diffusion coefficient (m^2/s)
    eta = viscosity (Pa*s)
    T = temperature (deg C)
    D_unc = uncertainty in D
    '''
    kB = 1.38e-23
    return kB*(T+273.15)/(6*np.pi*eta)*D_unc/D**2


def keep_MSD_fraction(im_in, f_track) -> np.array:
    """
    keeps first fraction of MSD vs lagtime 
    f_track fraction of track to keep
    """
    im_out = im_in.copy()
    for column in im_out:
        track_length = len(im_out[column].dropna())
        im_out[column].values[int(f_track*track_length):] = np.nan
    return im_out


def linear_func(x, m, b):
    """

    """
    return m*x+b


def water_viscosity(T, unit = 'C'):
    """
    returns viscosity of water as a function of temperature
    Input temperature unit can be 'C' or 'K'
    Output is Dynamics viscosity in units of Pa*s

    Source for formula:
    https://en.wikipedia.org/wiki/Temperature_dependence_of_viscosity, which in
    turn cites: Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E. (1987),
    The Properties of Gases and Liquids, McGraw-Hill Book Company, ISBN
    0-07-051799-1
    """
    if unit == 'C':
        T += 273.15
    elif unit == 'K':
        T = T
    else:
        print ('Temperature unit unknown. C and K are supported')
        return
    A = 1.856e-14 #Pa*s
    B = 4209 # K
    C = 0.04527 # K^-1
    D = -3.376e-5 # K^-2
    return A*np.exp(B/T + C*T +D*T**2)


def nextPerfectSquare(N):
    '''returns next perfect square larger than N
    '''
    nextN = floor(np.sqrt(N)) + 1
    return nextN * nextN


def make_quilt_from_image_list(ims, sep_px = 1, sep_val = 1, n_ims_max = None):
    '''makes a quilt of smaller images
    ims is a list of images, each image is a square 2D numpy array with the same shape
    sep_px (int) is the number of pixels as a separation between each image
    set_val (float) is the value to use for the separation pixels
    n_ims_max can be used to nake the quilt on a bigger grid as if there were this many images'''
    
    if n_ims_max is None:
        n_ims = len(ims)
    else:
        n_ims = max( n_ims_max, len(ims) )
    n_rows = int(np.sqrt(nextPerfectSquare(n_ims))) #number of rows in quilt
    im_px = ims[0].shape[0]
    
    #create blank quilt
    quilt_px = int(n_rows*im_px + sep_px*(n_rows-1))    
    quilt = sep_val*np.ones((quilt_px, quilt_px))
    
    #fill in quilt image by image
    for row in range(n_rows):
        for col in range(n_rows):
            im_index = col+row*n_rows
            if im_index>=len(ims):
                break
            image = ims[im_index]
            
            left_px = col*(im_px+sep_px)
            top_px = row*(im_px+sep_px)
            quilt[top_px : top_px + im_px, left_px : left_px+im_px] = image
    return quilt


def pad_image(data, pad_size, pad_val):
    '''
    pads an image or stack of images with a border around each edge
    data (np.ndarry): 2d or 3d: if 3d, first axis should be stack axis, other two axis will be padded
    pad_size (int): number of pixels to pad on each side
    pad_val (float): value to fill padded pixels with
    '''
    #pad for images near edge of FOV
    if data.ndim ==3:
        row_to_concat = pad_val*np.ones((data.shape[0],data.shape[1], pad_size))
        data = np.concatenate( (row_to_concat, data, row_to_concat ), axis = 2)
        row_to_concat = None
        col_to_concat = pad_val*np.ones((data.shape[0], pad_size, data.shape[2]))
        data = np.concatenate( (col_to_concat, data, col_to_concat ), axis = 1)
        col_to_concat = None
    if data.ndim ==2:
        row_to_concat = pad_val*np.ones((data.shape[0], pad_size))
        data = np.concatenate( (row_to_concat, data, row_to_concat ), axis = 1)
        row_to_concat = None
        col_to_concat = pad_val*np.ones((pad_size, data.shape[1]))
        data = np.concatenate( (col_to_concat, data, col_to_concat ), axis = 0)
        col_to_concat = None
    return data


def overwrite_check(out_file_base, file_extension):# Get output file mame for saving data
    """
    create output folder
    """
    output_folder = dirname(out_file_base)
    if not isdir(output_folder):
        mkdir(output_folder)
    
    # Get output file mame for saving data
    out_file_num = 0
    out_file = out_file_base+str(out_file_num).zfill(2)+file_extension
    while exists(out_file):
        out_file_num += 1
        out_file = out_file_base+str(out_file_num).zfill(2)+file_extension
    if out_file_num>0:
        out_file_num = out_file_num-1
        out_file = out_file_base+str(out_file_num).zfill(2)+file_extension
        print('Previous Output File Found')
        print(out_file)
        print('Type \'0\' to overwrite, \'1\' to save a new file, or \'x\' to not save any output.')
        overwrite = input()
        if overwrite == '1':#new file
            out_file_num +=1
        elif overwrite == '0': #overwrite
            out_file_num = out_file_num
        else: #don't save
            out_file_num = None
    if out_file_num is not None:
        out_file = out_file_base+str(out_file_num).zfill(2)+file_extension
        print('Saving as:', out_file)
    else:
        out_file = None
        print('Not Saving Results!' )

    return out_file
