"""

    GEMINI WRITTEN CODE

"""
import cupy as cp
import cupyx.scipy.ndimage as ndimage
from cupy.lib.stride_tricks import as_strided


# CuPy compiles this raw C++/CUDA code automatically the first time you run it.
# It computes mean, variance, and covariance in a single memory-safe pass.
symmetry_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_symmetry(const float* img, float* out, int width, int height, int radius) {
    // Calculate the thread's (x, y) pixel coordinates
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    // Keep threads within image bounds
    if (x >= width || y >= height) return;

    float sum_xy = 0.0f;
    float sum_x = 0.0f;
    float sum_x2 = 0.0f;
    float count = 0.0f;

    // Loop over the local window
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int px1 = x + dx;
            int py1 = y + dy;
            
            // For the 180-degree flipped coordinate
            int px2 = x - dx;
            int py2 = y - dy;

            // Border handling (clamps to edge if looking outside image)
            px1 = max(0, min(width - 1, px1));
            py1 = max(0, min(height - 1, py1));
            px2 = max(0, min(width - 1, px2));
            py2 = max(0, min(height - 1, py2));

            float val1 = img[py1 * width + px1];
            float val2 = img[py2 * width + px2];

            // Accumulate stats
            sum_xy += val1 * val2;
            sum_x += val1;
            sum_x2 += val1 * val1;
            count += 1.0f;
        }
    }

    // Calculate variance and covariance
    float mu = sum_x / count;
    float var = (sum_x2 / count) - (mu * mu);
    float cov = (sum_xy / count) - (mu * mu);

    // Calculate final symmetry and clamp between -1.0 and 1.0
    float sym = cov / (var + 1e-6f);
    sym = max(-1.0f, min(1.0f, sym));

    out[y * width + x] = sym;
}
''', 'compute_symmetry')


def compute_local_point_symmetry_gpu_kernel(image, radius=25):
    """
    Robust GPU point-symmetry using a compiled CUDA kernel.
    Safely handles large radii and prevents VRAM overflow.
    """
    # 1. Ensure input is float32 and contiguous on the GPU
    img_cp = cp.asarray(image, dtype=cp.float32)
    if not img_cp.flags.c_contiguous:
        img_cp = cp.ascontiguousarray(img_cp)
        
    h, w = img_cp.shape
    
    # 2. Prepare output array on GPU
    out_cp = cp.empty_like(img_cp)
    
    # 3. Define block and grid dimensions for the GPU threads
    threads_per_block = (16, 16)
    blocks_per_grid_x = (w + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (h + threads_per_block[1] - 1) // threads_per_block[1]
    
    # 4. Execute the CUDA kernel
    symmetry_kernel(
        (blocks_per_grid_x, blocks_per_grid_y), 
        threads_per_block, 
        (img_cp, out_cp, w, h, radius)
    )
    
    return out_cp


def compute_local_point_symmetry_gpu(image, radius=5):
    """
    Computes local 180-degree rotational correlation using CuPy.
    Fully vectorized with no Python for-loops.

    symmetry map = cov(X,Y) - mean^2 / (sigma^2 + epsilon) where X is a local image patch and Y is the same patch rotate 180 degrees
    
    Parameters:
    - image: 2D cupy array (or numpy array, will be converted)
    - radius: Half the window size. Window is (2*radius + 1) square.
    
    Returns:
    - symmetry_map: 2D cupy array [-1.0, 1.0]
    """
    # 1. Move to GPU (if not already) and cast to float32
    img_cp = cp.asarray(image, dtype=cp.float32)
    window_size = 2 * radius + 1
    
    # 2. Compute local mean (mu) and local variance (sigma^2)
    # cupyx uniform_filter is the GPU equivalent of cv2.boxFilter(normalize=True)
    mu = ndimage.uniform_filter(img_cp, size=window_size, mode='reflect')
    
    img_sq = img_cp ** 2
    mu_sq = ndimage.uniform_filter(img_sq, size=window_size, mode='reflect')
    
    variance = mu_sq - (mu ** 2)
    
    # 3. Vectorized shift-and-multiply using sliding_window_view
    # Pad the image so the output shape matches the input shape
    padded = cp.pad(img_cp, radius, mode='reflect')
    
    # Extract overlapping local windows. Shape: (h, w, window_size, window_size)
    # This is a zero-copy memory view!
    #windows = sliding_window_view(padded, (window_size, window_size))
    new_shape = (img_cp.shape[0], img_cp.shape[1], window_size, window_size)
    new_strides = (padded.strides[0], padded.strides[1], padded.strides[0],padded.strides[1])
    windows = as_strided(padded, shape = new_shape, strides = new_strides)
    
    # Flip the windows upside down and left-to-right (180 degree rotation)
    windows_flipped = windows[:, :, ::-1, ::-1]
    
    # Multiply and compute the mean over the window axes (-2 and -1)
    # This replaces the nested for-loops entirely.
    S_mean = cp.mean(windows * windows_flipped, axis=(-2, -1))
    
    # 4. Calculate final symmetry score
    epsilon = 1e-6
    covariance = S_mean - (mu ** 2)
    symmetry_map = covariance / (variance + epsilon)
    
    # Clip to [-1.0, 1.0] to handle floating point noise
    cp.clip(symmetry_map, -1.0, 1.0, out=symmetry_map)
    
    return symmetry_map


# Compiles the raw C++ GST algorithm for the GPU
gst_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_gst(const float* mag, const float* theta, float* out, 
                 int width, int height, int radius, float sigma) {
                 
    // Calculate the thread's (x, y) center pixel coordinates
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float symmetry_sum = 0.0f;

    // Slide window to find pairs (p1, p2) mirrored across the center
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx == 0 && dy == 0) continue; // Skip the exact center

            int px1 = x + dx;
            int py1 = y + dy;
            int px2 = x - dx;
            int py2 = y - dy;

            // Strict bounds check - skip pair if either hangs off the image edge
            if (px1 < 0 || px1 >= width || py1 < 0 || py1 >= height ||
                px2 < 0 || px2 >= width || py2 < 0 || py2 >= height) {
                continue;
            }

            float m1 = mag[py1 * width + px1];
            float m2 = mag[py2 * width + px2];

            // Optimization: If there are no edges here, skip the expensive trig math
            if (m1 == 0.0f || m2 == 0.0f) continue;

            float t1 = theta[py1 * width + px1];
            float t2 = theta[py2 * width + px2];

            // Distance weight: exp(-d / sigma)
            float dist = sqrtf((float)(dx * dx + dy * dy));
            float distance_weight = expf(-dist / sigma);

            // Angle of the line connecting the two pixels
            float alpha = atan2f((float)dy, (float)dx);

            // Reisfeld's Phase Weight formula:
            // P = (1 - cos(t1 + t2 - 2*alpha)) * (1 - cos(t1 - t2))
            float phase1 = 1.0f - cosf(t1 + t2 - 2.0f * alpha);
            float phase2 = 1.0f - cosf(t1 - t2);
            float phase_weight = phase1 * phase2;

            // Accumulate score (Log magnitudes * distance * phase)
            symmetry_sum += m1 * m2 * distance_weight * phase_weight;
        }
    }

    // Divide by 2 because the loop double-counts every pair (d and -d)
    out[y * width + x] = symmetry_sum / 2.0f;
}
''', 'compute_gst')


def compute_generalized_symmetry_gpu(image, radius=25, sigma=None):
    """
    Computes the Generalized Symmetry Transform (GST) using CuPy.
    
    Parameters:
    - image: 2D numpy/cupy array (event histogram or grayscale image)
    - radius: Max distance to look for symmetric edge pairs
    - sigma: Gaussian falloff for distance. Defaults to radius / 2.0.
    
    Returns:
    - gst_map: 2D cupy array containing symmetry magnitudes.
    """
    if sigma is None:
        sigma = radius / 2.0

    img_cp = cp.asarray(image, dtype=cp.float32)
    h, w = img_cp.shape
    
    # 1. Compute Image Gradients (Sobel filters)
    # axis=0 is dy (vertical), axis=1 is dx (horizontal)
    dy = ndimage.sobel(img_cp, axis=0)
    dx = ndimage.sobel(img_cp, axis=1)
    
    # 2. Compute Magnitude and Theta (Angles)
    mag = cp.hypot(dx, dy)
    theta = cp.arctan2(dy, dx)
    
    # 3. Log-compress magnitudes (Reisfeld's specification)
    # cp.log1p computes log(1 + x) safely
    mag_log = cp.log1p(mag)
    
    # Ensure memory is contiguous for the C++ pointer math
    mag_log = cp.ascontiguousarray(mag_log)
    theta = cp.ascontiguousarray(theta)
    
    # 4. Prepare output array
    out_cp = cp.zeros_like(img_cp)
    
    # 5. Launch CUDA Kernel
    threads_per_block = (16, 16)
    blocks_per_grid_x = (w + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (h + threads_per_block[1] - 1) // threads_per_block[1]
    
    gst_kernel(
        (blocks_per_grid_x, blocks_per_grid_y), 
        threads_per_block, 
        (mag_log, theta, out_cp, w, h, radius, cp.float32(sigma))
    )
    
    return out_cp


# We update the kernel to output TWO arrays: out_sym and out_var
symmetry_variance_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_symmetry_and_variance(const float* img, float* out_sym, float* out_var, 
                                   int width, int height, int radius) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum_xy = 0.0f;
    float sum_x = 0.0f;
    float sum_x2 = 0.0f;
    float count = 0.0f;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int px1 = max(0, min(width - 1, x + dx));
            int py1 = max(0, min(height - 1, y + dy));
            int px2 = max(0, min(width - 1, x - dx));
            int py2 = max(0, min(height - 1, y - dy));

            float val1 = img[py1 * width + px1];
            float val2 = img[py2 * width + px2];

            sum_xy += val1 * val2;
            sum_x += val1;
            sum_x2 += val1 * val1;
            count += 1.0f;
        }
    }

    float mu = sum_x / count;
    float var = (sum_x2 / count) - (mu * mu);
    float cov = (sum_xy / count) - (mu * mu);

    float sym = cov / (var + 1e-6f);
    
    // Output both symmetry and variance
    out_sym[y * width + x] = max(-1.0f, min(1.0f, sym));
    out_var[y * width + x] = var;
}
''', 'compute_symmetry_and_variance')


def evaluate_bias_quality(event_histogram, radius=25, noise_threshold=0.01):
    """
    Calculates a global, variance-weighted anti-symmetry score to evaluate
    event camera bias configurations. Higher score = better contrast/bandwidth.
    
    Parameters:
    - event_histogram: 2D array of events (ON = positive, OFF = negative)
    - radius: Window size for symmetry detection
    - noise_threshold: Minimum local variance required to consider a region 
                       "active". Prevents flat background noise from diluting the score.
                       
    Returns:
    - global_score (float): A scalar value where higher means better bias tuning.
    """
    img_cp = cp.asarray(event_histogram, dtype=cp.float32)
    if not img_cp.flags.c_contiguous:
        img_cp = cp.ascontiguousarray(img_cp)
        
    h, w = img_cp.shape
    
    # Prepare dual output arrays
    out_sym = cp.empty_like(img_cp)
    out_var = cp.empty_like(img_cp)
    
    # Launch Kernel
    threads_per_block = (16, 16)
    blocks_per_grid_x = (w + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (h + threads_per_block[1] - 1) // threads_per_block[1]
    
    symmetry_variance_kernel(
        (blocks_per_grid_x, blocks_per_grid_y), 
        threads_per_block, 
        (img_cp, out_sym, out_var, w, h, radius)
    )
    
    # 1. Get the Anti-Symmetry map (since ON/OFF mirror each other)
    anti_sym = -out_sym 
    
    # 2. Mask out the empty background / dead pixels
    # Event data is sparse; we only want to score areas with actual structural activity
    valid_mask = out_var > noise_threshold
    
    # Safety check: if the biases are completely dead and produce no events
    if not cp.any(valid_mask):
        return 0.0
        
    valid_sym = anti_sym[valid_mask]
    valid_var = out_var[valid_mask]
    
    # 3. Calculate Variance-Weighted Mean
    # Formula: Sum(Symmetry * Variance) / Sum(Variance)
    weighted_sum = cp.sum(valid_sym * valid_var)
    total_variance = cp.sum(valid_var)
    
    global_score = weighted_sum / total_variance
    
    return float(global_score)
