"""

workflow for barrel distortion correction -> square corners reprojection:

1) find centroids of checkerboard?
2) sort centroids (staggered checkerboard...)
    - FLIP X AND Y COORDINATES! <---------
    - USE visualization to make usre centroids are in right order!!
3) fit parameters
4) transform point set from fisheye to perspective (barrel_dist_correct_points)
5) square projection


Notes:
    the x,y format for fit_distortion should be (based on napari's coordinate
    system with horizontal = x, vertical = y):
        sorted_centroids = [[y1,x1],[y2,x2], ....]
    

    if you are getting weird results use visualize_sorting_matploltlib. it
    expects the sorted_centroids to be in the correct shape that should be
    input to fit_distortion...


"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_checkerboard_centroids(image_float32):
    # 1. Normalize and convert to uint8 grayscale
    img_norm = cv2.normalize(image_float32, None, 0, 255, cv2.NORM_MINMAX)
    gray = np.uint8(img_norm)
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # 2. Threshold the image to isolate the lighter squares
    # Using Otsu's method to automatically find the best threshold between the black and gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Optional: Apply morphological operations to smooth the jagged edges
    # kernel = np.ones((3,3), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 4. Find contours of the isolated squares
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []

    # 5. Calculate the centroid (center of mass) for each square
    for cnt in contours:
        # Filter out tiny noise blobs
        if cv2.contourArea(cnt) > 20:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                centroids.append((cX, cY))

    return centroids, binary


def sort_staggered_centroids(centroids, angle_tolerance=30):
     """
     Sorts unstructured, staggered checkerboard centroids into a strict 
     left-to-right, top-to-bottom order, automatically tracking barrel distortion.
     """
     # Convert input to a list of tuples for easy removal during tracking
     if isinstance(centroids, np.ndarray):
         centroids = centroids.reshape(-1, 2)
     remaining_pts = [tuple(pt) for pt in centroids]
 
     rows = []
 
     while remaining_pts:
         # 1. Start a new row at the absolute leftmost available point.
         # This guarantees we always start at the extreme left edge of SOME row, 
         # completely bypassing issues caused by the curved edges of the distortion.
         start_pt = min(remaining_pts, key=lambda p: p[0])
 
         row = [start_pt]
         remaining_pts.remove(start_pt)
 
         curr_pt = start_pt
         current_angle = 0.0  # Assume a generally horizontal starting trajectory
 
         while True:
             candidates = []
 
             # 2. Search for the next point to the right
             for p in remaining_pts:
                 dx = p[0] - curr_pt[0]
                 dy = p[1] - curr_pt[1]
 
                 if dx > 0: # Ensure we are moving right
                     angle = np.degrees(np.arctan2(dy, dx))
                     angle_diff = abs(angle - current_angle)
 
                     # 3. Angle Constraint: Ignore points outside the horizontal cone.
                     # This completely filters out the staggered diagonal points.
                     if angle_diff <= angle_tolerance:
                         dist = np.hypot(dx, dy)
                         candidates.append((dist, angle, p))
 
             if not candidates:
                 break  # No more points fit this row
 
             # 4. Pick the closest point that satisfied the angle constraint
             candidates.sort(key=lambda x: x[0])
             best_dist, best_angle, next_pt = candidates[0]
 
             row.append(next_pt)
             remaining_pts.remove(next_pt)
 
             # 5. Smoothly update the trajectory angle to follow the barrel curve
             current_angle = 0.5 * current_angle + 0.5 * best_angle
             curr_pt = next_pt
 
         rows.append(row)
 
     # 6. Sort the completed rows top-to-bottom based on their average Y coordinate
     rows.sort(key=lambda r: np.mean([pt[1] for pt in r]))
 
     # 7. Flatten into a single ordered list
     sorted_centroids = []
     for r in rows:
         sorted_centroids.extend(r)
 
     print(f"Sorted into {len(rows)} distinct rows.")
     return sorted_centroids


def fit_distortion(sorted_centroids: np.ndarray, board_width: int, 
                   board_height: int, image_width: int = 1280, 
                   image_height: int = 720, first_square_offset = 0):
    """
    This function calculates the fisheye distortion correction...

    subsequently use this to undistort an image...(or point set):
        undistorted_image = cv2.fisheye.undistortImage(im, K, D, Knew = new_K)

     first square offset:
        Determine if the very first square (top-left, row 0, col 0) is bright or
        black.  Set this to 0 if (0,0) is bright, or 1 if (0,0) is black.

    """
    obj_pts_list = []


    print("flipped these back.. again")
    for j in range(board_width):
        for i in range(board_height):
            # Checkerboard logic: only create a point if it corresponds to a bright square
            if (i + j) % 2 == first_square_offset:
                # Add the physical X, Y grid coordinates; Z is always 0
                obj_pts_list.append([j, i, 0])

    # Convert to the shape OpenCV expects: (N, 1, 3) float32
    expected_points = len(obj_pts_list)
    obj_pts = np.array(obj_pts_list, dtype=np.float32).reshape(expected_points, 1, 3)


    # Now, ensure your sorted_centroids matches this exact length
    num_centroids = len(sorted_centroids)
    if num_centroids != expected_points:
        raise ValueError(
            f"Point mismatch! Found {num_centroids} centroids in the image, "
            f"but expected {expected_points} bright squares."
        )

    imagePoints = [np.array(sorted_centroids, dtype=np.float32).reshape(-1, 1, 2)]
    objectPoints = [obj_pts]



    # Create a basic initial guess for the camera matrix K
    # Assume optical center is exactly in the middle of the image
    K = np.zeros((3, 3), dtype=np.float64)
    K[0, 2] = image_width / 2.0   # cx
    K[1, 2] = image_height / 2.0  # cy
    K[2, 2] = 1.0
    # Give a rough guess for focal length (can be tuned if it still fails)
    K[0, 0] = image_width         # fx
    K[1, 1] = image_width         # fy

    D = np.zeros((4, 1), dtype=np.float64)

    # Remove CALIB_CHECK_COND. 
    # Add USE_INTRINSIC_GUESS and FIX_PRINCIPAL_POINT to stabilize the math.
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
        cv2.fisheye.CALIB_FIX_SKEW |
        cv2.fisheye.CALIB_USE_INTRINSIC_GUESS |
        cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
    )

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objectPoints,
        imagePoints,
        (image_width, image_height),
        K,
        D,
        flags=calibration_flags
    )
    balance = 1
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (image_width, image_height), np.eye(3), balance = balance)

    #undistorted_image = cv2.fisheye.undistortImage(im, K, D, Knew = new_K)
    return rms, K, D, rvecs, tvecs, new_K


def visualize_sorting_matplotlib(image, sorted_points):
    """
    Visualizes the sorted checkerboard points using Matplotlib.
    The points are colored in a gradient from Red (start) to Green (end),
    with a line tracing the exact sorting path.
    """
    # Convert points to a NumPy array for easy column slicing
    pts = np.array(sorted_points)
    x = pts[:, 0]
    y = pts[:, 1]

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Handle image color channels (OpenCV loads as BGR, Matplotlib expects RGB)
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 1. Draw a faint line connecting all points to show the trace path
    plt.plot(x, y, color='white', linewidth=1, alpha=0.5, zorder=1)

    # 2. Draw the points with a colormap to represent the sorting sequence
    # 'RdYlGn' maps the lowest indices (0) to Red, and highest indices to Green
    indices = np.arange(len(x))
    scatter = plt.scatter(x, y, c=indices, cmap='RdYlGn', s=20, edgecolors='black', linewidth=0.5, zorder=2)

    # 3. Add a colorbar to act as a legend for the sequence
    cbar = plt.colorbar(scatter)
    cbar.set_label('Point Index (Sorting Order)')

    # 4. Explicitly label the START and END points for quick debugging
    plt.annotate('START', (x[0], y[0]), textcoords="offset points", xytext=(0,10),
                 ha='center', color='red', fontweight='bold', fontsize=12)
    plt.annotate('END', (x[-1], y[-1]), textcoords="offset points", xytext=(0,-15),
                 ha='center', color='lime', fontweight='bold', fontsize=12)

    plt.title(f"Sorted Sequence ({len(x)} total points)")
    plt.axis('off') # Hide axes for a cleaner image view
    plt.tight_layout()
    f_name = "/tmp/visualize_sorting.png"
    print(f"[INFO] saving sorted points visualization to {f_name}")
    plt.gcf().savefig(f_name)


def barrel_dist_correct_points(pts, K, D, K_new):
    """
    correct barrel distortion of points...
    """
    pts = sort_staggered_centroids(pts)
    pts = np.ascontiguousarray(pts).reshape(-1,1,2)
    undistorted_centroids = cv2.fisheye.undistortPoints(pts, K, D, P = K_new)
    return undistorted_centroids


def square_reprojection(undistorted_centroids, board_width: int, board_height: int, square_pixel_size: int = 25, first_square_offset = 0):
    """

    after calling this function to calculate H, output_width, output_height warp image with:

        undistorted_img = cv2.fisheye.undistortImage(....)
        squared_up_image = cv2.warpPerspective(undistorted_img,H,
                                               (output_width, output_height)
        )

    first square offset:
        Determine if the very first square (top-left, row 0, col 0) is bright or
        black.  Set this to 0 if (0,0) is bright, or 1 if (0,0) is black.

    """
    # 2. Build the perfectly squared destination grid
    dst_pts_list = []
    for i in range(board_height):
        for j in range(board_width):
            # Using the exact same staggering logic as before
            if (i + j) % 2 == first_square_offset:
                # Multiply grid coordinates by the desired pixel size
                dst_pts_list.append([j * square_pixel_size, i * square_pixel_size])

    dst_pts = np.array(dst_pts_list, dtype=np.float32).reshape(-1, 1, 2)

    # Assume 'undistorted_centroids' are your 572 points after running cv2.fisheye.undistortPoints
    src_pts = undistorted_centroids.reshape(-1, 1, 2)

    # 3. Calculate the Homography Matrix (H)
    # We use RANSAC to make the math robust against any tiny remaining inaccuracies
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    #undistorted_img = cv2.fisheye.undistortImage(inst.__fetch_layer__("im").data.copy(), K, D, Knew = K_new)

    # Calculate the required dimensions of the final cropped image
    output_width = board_width * square_pixel_size
    output_height = board_height * square_pixel_size

    ## Warp the undistorted image to the flat perspective
    return H, output_width, output_height

