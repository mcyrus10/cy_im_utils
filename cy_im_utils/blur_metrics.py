import numpy as np
from scipy import ndimage
import logging

def crete_blur_metric(F : str,h : int = 9):
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

def diagonal_laplacian(F : str, s : int = 1):
    """
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    edge = s*2+1
    kernel_1 = np.zeros(edge).reshape(1,edge)
    kernel_1[0,0],kernel_1[0,s],kernel_1[0,-1] = 1,-2,1

    kernel_2 = np.zeros([edge,edge])
    kernel_2[0,0],kernel_2[s,s],kernel_2[-1,-1] = 1,-2,1
    kernels = [
                kernel_1,
                kernel_1.T,
                kernel_2,
                kernel_2[::-1]
                ]

    FM = [np.abs(ndimage.convolve(F,k)) for k in kernels]
    FM = np.sum(np.dstack(FM), axis = 2)
    return np.mean(FM)
 
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