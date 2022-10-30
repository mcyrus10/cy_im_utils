import numpy as np
from scipy import ndimage
import cupyx.scipy.ndimage as ndimage_GPU
import logging
import cupy as cp

def crete_blur_metric(F: np.array, h: int = 9) -> float:
    """
    Crete, F., Dolmiere, T., Ladret, P., & Nicolas, M. (2007). The blur effect:
    perception and estimation with a new no-reference perceptual blur metric.
    Human Vision and Electronic Imaging XII, 6492, 64920I.
    https://doi.org/10.1117/12.702790

    THE MANUSCRIPT IS A LITTLE BIT SCREWY ON ITS DESIGNATION OF m x n (rows x
    columns) SYNTAX!!!!!!!!!!!!!! (SEE EQUATION 2)

    """

    h_v = (1/h)*np.ones(h).reshape(1,h)                         # equation 1
    h_h = h_v.T                                                 # equation 1

    B_ver = ndimage.convolve(F, h_v)                            # equation 1
    B_hor = ndimage.convolve(F, h_h)                            # equation 1

    D_F_ver = np.abs(np.diff(F,1,1))                            # equation 2
    D_F_hor = np.abs(np.diff(F,1,0))                            # equation 2
                                                                             
    D_B_ver = np.abs(np.diff(B_ver,1,1))                        # equation 2
    D_B_hor = np.abs(np.diff(B_hor,1,0))                        # equation 2


    V_ver = np.maximum(0,D_F_ver[1:-1,1:-1]-D_B_ver[1:-1,1:-1]) # equation 3
    V_hor = np.maximum(0,D_F_hor[1:-1,1:-1]-D_B_hor[1:-1,1:-1]) # equation 3

    s_F_ver = np.sum(D_F_ver[1:-1,1:-1])                        # equation 4
    s_F_hor = np.sum(D_F_hor[1:-1,1:-1])                        # equation 4

    s_V_ver = np.sum(V_ver[1:-1,1:-1])                          # equation 4
    s_V_hor = np.sum(V_hor[1:-1,1:-1])                          # equation 4

    b_F_ver = (s_F_ver-s_V_ver)/s_F_ver                         # equation 5
    b_F_hor = (s_F_hor-s_V_hor)/s_F_hor                         # equation 5

    for v in [s_F_ver,s_F_hor,s_V_ver,s_V_hor]:
        logging.debug(v)

    for v in [b_F_ver,b_F_hor]:
        logging.debug(v)

    blur = np.max([b_F_ver,b_F_hor])                            # equation 6
    logging.debug(f'Crete blur metric = {blur}')
    logging.debug(f'Crete blur metric (normalized) = {1.0 - blur}')
    return blur

def crete_blur_metric_GPU(F: cp.array, h: int = 9) -> np.array:
    """

    this is intended to operate on 3D arrays (stack of 2D images) instead of
    single images with 
        index 0 : image 
        index 1 : rows 
        index 2 : columns

    """

    nx,ny,nz = F.shape

    h_v = (1/h)*cp.ones(h).reshape(1,1,h)                       # equation 1
    h_h = cp.transpose(h_v,(0,2,1))                             # equation 1

    B_ver = ndimage_GPU.convolve(F, h_v)                        # equation 1
    B_hor = ndimage_GPU.convolve(F, h_h)                        # equation 1

    D_F_ver = cp.abs(cp.diff(F,1,2))                            # equation 2
    D_F_hor = cp.abs(cp.diff(F,1,1))                            # equation 2
                                                                             
    D_B_ver = cp.abs(cp.diff(B_ver,1,2))                        # equation 2
    D_B_hor = cp.abs(cp.diff(B_hor,1,1))                        # equation 2

    blur = np.zeros([nx], dtype = np.float32)
    for j in range(nx):
        V_ver = cp.maximum(0,D_F_ver[j,1:-1,1:-1]-D_B_ver[j,1:-1,1:-1])
        V_hor = cp.maximum(0,D_F_hor[j,1:-1,1:-1]-D_B_hor[j,1:-1,1:-1])

        s_F_ver = cp.sum(D_F_ver[j,1:-1,1:-1])                  # equation 4
        s_F_hor = cp.sum(D_F_hor[j,1:-1,1:-1])                  # equation 4

        s_V_ver = cp.sum(V_ver[1:-1,1:-1])                      # equation 4
        s_V_hor = cp.sum(V_hor[1:-1,1:-1])                      # equation 4

        b_F_ver = (s_F_ver-s_V_ver)/s_F_ver                     # equation 5
        b_F_hor = (s_F_hor-s_V_hor)/s_F_hor                     # equation 5

        for v in [s_F_ver,s_F_hor,s_V_ver,s_V_hor]:
            logging.debug(v)

        for v in [b_F_ver,b_F_hor]:
            logging.debug(v)

        blur[j] = np.max([b_F_ver.get(),b_F_hor.get()]) # equation 6
    #logging.debug(f'Crete blur metric = {blur}')
    #logging.debug(f'Crete blur metric (normalized) = {1.0 - blur}')
    return np.array(blur)

def helm_GPU(im: cp.array, window_size: int = 15) -> np.array:
    """
    This is taken from the implementation in the infer gitlab, with the
    modification being that it takes a convolution not a 'fspecial'
    """
    conv_kernel = cp.ones([1,window_size,window_size]) / (window_size**2)
    U = ndimage_GPU.convolve(im,conv_kernel)
    R1 = U/im
    R1[im == 0] = 1
    index = U > im
    FM = 1/R1
    FM[index] = R1[index]
    return cp.mean(FM, axis = (1,2)).get()

def diagonal_laplacian(F : np.array, s : int = 1):
    """
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    edge = s*2+1
    kernel_1 = np.zeros(edge).reshape(1,edge)
    kernel_1[0,0],kernel_1[0,s],kernel_1[0,-1] = 1,-2,1

    kernel_2 = np.zeros([edge,edge])
    kernel_2[0,0],kernel_2[s,s],kernel_2[-1,-1] = 1,-2,1
    sqrt_2 = 2**(-1/2)
    kernels = [
                kernel_1,
                kernel_1.T,
                kernel_2*sqrt_2,
                kernel_2[::-1]*sqrt_2
                ]

    FM = np.array([np.abs(ndimage.convolve(F,k)) for k in kernels])
    FM = np.sum(FM, axis = 0)
    return np.mean(FM)

def diagonal_laplacian_GPU(F : cp.array, s : int = 1):
    """
    """
    assert s == 1, "I dont think this works for s > 1...?"
    edge = s*2+1
    kernel_1 = cp.zeros(edge).reshape(1,edge)
    kernel_1[0,0],kernel_1[0,s],kernel_1[0,-1] = 1,-2,1
    kernel_1 = kernel_1[None,:,:]

    kernel_2 = cp.zeros([edge,edge])
    kernel_2[0,0],kernel_2[s,s],kernel_2[-1,-1] = 1,-2,1
    kernel_2 = kernel_2[None,:,:]
    const = 2**(-1/2)
    kernels = [
            kernel_1,
            cp.transpose(kernel_1,(0,2,1)),
            kernel_2*const,
            kernel_2[:,::-1,:]*const
                ]

    FM = [cp.abs(ndimage_GPU.convolve(F,k)) for k in kernels]
    FM = cp.transpose(cp.stack(FM),(1,0,2,3))
    FM = cp.sum(FM,axis = 1)
    return cp.mean(FM, axis = (1,2)).get()
 
def variance_of_laplacian_GPU(im: cp.array) -> np.array:
    """
    Variance of laplacian as a blurring metric
    """
    laplacian_kernel = cp.array([   [0,1,0],
                                    [1,-4,1],
                                    [0,1,0]], dtype = cp.float32)[None,:,:]
    conv = ndimage_GPU.convolve(im,laplacian_kernel)
    return cp.asnumpy(cp.std(conv, axis = (1,2))**2)

if __name__=="__main__":
    """
    """
    plt.rcParams['image.cmap'] = 'gray'
    file_path = "/mnt/d/Misc Images/C0372413-Bone_tissue,_SEM.jpg"
    #file_path = "/mnt/d/Misc Images/test_image.png"
    file_path = "/mnt/d/Misc Images/sem3.jpg"
    F = np.array(Image.open(file_path), dtype = np.float32)[:,:,0]
    if len(F.shape) == 3:
        F = F[:,:,0]

    blur_im = False
    if blur_im:
        sigma = 1.0
        F = ndimage.gaussian_filter(F,sigma)

    h = 9
    h_v = (1/h)*np.ones(h).reshape(1,h)                         # equation 1
    h_h = h_v.T                                                 # equation 1

    B_ver = ndimage.convolve(F, h_v)                            # equation 1
    B_hor = ndimage.convolve(F, h_h)                            # equation 1

    D_F_ver = np.abs(np.diff(F,1,1))                            # equation 2
    D_F_hor = np.abs(np.diff(F,1,0))                            # equation 2
                                                                             
    D_B_ver = np.abs(np.diff(B_ver,1,1))                        # equation 2
    D_B_hor = np.abs(np.diff(B_hor,1,0))                        # equation 2


    V_ver = np.maximum(0,D_F_ver[1:-1,1:-1]-D_B_ver[1:-1,1:-1]) # equation 3
    V_hor = np.maximum(0,D_F_hor[1:-1,1:-1]-D_B_hor[1:-1,1:-1]) # equation 3

    s_F_ver = np.sum(D_F_ver[1:-1,1:-1])                        # equation 4
    s_F_hor = np.sum(D_F_hor[1:-1,1:-1])                        # equation 4

    s_V_ver = np.sum(V_ver[1:-1,1:-1])                        # equation 4
    s_V_hor = np.sum(V_hor[1:-1,1:-1])                        # equation 4

    b_F_ver = (s_F_ver-s_V_ver)/s_F_ver                         # equation 5
    b_F_hor = (s_F_hor-s_V_hor)/s_F_hor                         # equation 5

    for v in [s_F_ver,s_F_hor,s_V_ver,s_V_hor]:
        print(v)

    for v in [b_F_ver,b_F_hor]:
        print(v)

    blur = np.max([b_F_ver,b_F_hor])                            # equation 6

    print(f'CRETE blur metric = {blur}')
    print(f'CRETE blur metric (normalized) = {1.0 - blur}')
    # PLOTS
    _,ax = plt.subplots(1,3, sharex = True, sharey = True)
    ax[0].imshow(F)
    ax[1].imshow(B_ver)
    ax[2].imshow(B_hor)
    ax[0].set_title("F")
    ax[1].set_title("B_ver (transpose)")
    ax[2].set_title("B_hor transpose)")
    # D_
    _,ax = plt.subplots(2,2, sharex = True, sharey = True)
    ax = ax.flatten()
    strings = ['D_F_ver','D_F_hor','D_B_ver','D_B_hor']
    for i,im in enumerate([D_F_ver,D_F_hor,D_B_ver,D_B_hor]):
        ax[i].imshow(im, vmin = 0, vmax = 1)
        ax[i].set_title(strings[i])
    # V_
    _,ax = plt.subplots(1,2, sharex = True, sharey = True)
    ax[0].imshow(V_ver)
    ax[1].imshow(V_hor)

    plt.show()
