import cupy as cp
from tqdm import tqdm

def calc_auto_correlation(image_stack, down_sampling: int = 1):
    """
    idea:
        Calculate auto correlation for |gradient| of first image -> |gradient|
        of 1 period image will have high auto correlation to 1st image's
        gradient (gradients work better than image intensities...)
    
    down sampling speeds this up
    """
    im_handle = image_stack[:,::down_sampling,::down_sampling]
    n_im, nx, ny = im_handle.shape

    im_0 = cp.array(im_handle[0])
    dx, dy = cp.gradient(im_0)
    grad_ref = cp.array((dx**2+dy**2)**(1/2))
    auto_correlation = cp.zeros(n_im)
    for j in tqdm(range(1,n_im), desc = 'correlating'):
        im_cp = cp.array(im_handle[j])
        dx, dy = cp.gradient(im_cp)
        grad_mag = (dx**2+dy**2)**(1/2)
        correlation = cp.correlate(grad_ref.flatten(), grad_mag.flatten())
        auto_correlation[j-1] = correlation
    return auto_correlation.get()


