"""

                                TRACKPY UTILS

"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import trackpy as tp
#import trackpy.predict
import numba as nb
import cupy as cp


def imsd_powerlaw_fit(imsd_dict,
        start_index: int = 0, 
        end_index: int = None, 
        stride: int = 1,
        verbose = True,
        ) -> tuple:
    """
    This performs the log-log fit on all the imsd curves
    """
    if end_index is None:
        end_index = len(imsd_dict.index.values)
    fit_slice = slice(start_index, end_index, stride)
    time = imsd_dict.index.values[fit_slice]
    log_t = np.log(time)
    ones = np.ones_like(log_t)
    A_mat = np.vstack([ones, log_t]).T
    # Taking slice along first dimension so it can handle single particle or
    # imsd with multiple particles
    imsd_handle = imsd_dict.values[fit_slice]
    b_mat = np.log(imsd_handle, where = imsd_handle > 0)
    A, n = np.linalg.lstsq(A_mat, b_mat, rcond = -1)[0]

    ndim = imsd_dict.values.ndim
    if ndim == 2:
        fits = np.exp(A)[None,:] * time[:,None] ** n[None,:]
        if np.isnan(fits[0]).sum() > 0 and verbose:
            print("warning -> nans in fit")
    elif ndim == 1:
        fits = np.exp(A) * time ** n
    return A, n, fits


def imsd_linear_fit(imsd_dict: pd.DataFrame,
                    start_index: int = 0,
                    end_index: int = None,
                    stride: int = 1,
                    verbose = True,
                    ) -> tuple:
    """
    This performs the linear fit on all the imsd curves

    Parameters:
    -----------
        imsd_dict : pandas dataframe - output of imsd individual msd curve for
                                       each particle
        start index : int - for slicing irregular parts of MSD curve
        end index: int 
    """
    if end_index is None:
        end_index = len(imsd_dict.index.values)
    fit_slice = slice(start_index, end_index, stride)
    time = imsd_dict.index.values[fit_slice]
    ones = np.ones_like(time)
    A_mat = np.vstack([ones, time]).T
    ndim = imsd_dict.values.ndim
    if verbose: print("ndim = ",ndim)
    if ndim == 1:
        b_mat = imsd_dict.values[fit_slice]
        b, m = np.linalg.lstsq(A_mat, b_mat, rcond = -1)[0]
        fits = m * time + b
    elif ndim == 2:
        b_mat = imsd_dict.values[fit_slice, :]
        b, m = np.linalg.lstsq(A_mat, b_mat, rcond = -1)[0]
        fits = m[None,:] * time[:,None] + b[None,:]
    else:
        return None

    if np.isnan(fits[0]).sum() > 0 and verbose:
        print("warning -> nans in fit")
    return m, b, fits


def imsd_linear_fit_weighted(imsd_dict: pd.DataFrame,
                    start_index: int = 0,
                    end_index: int = None,
                    stride: int = 1,
                    verbose = True,
                    mode = 'weighted',
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Weighted fits .... high variance points get less weight...

    This can handle a single series or a set of msd curves...

    """
    # IF a 1D series is passed this adds a dimension converting to df
    if isinstance(imsd_dict, pd.core.series.Series):
        print("[INFO] Series -> DataFrame")
        imsd_dict_local = pd.DataFrame(imsd_dict)
    else:
        imsd_dict_local = imsd_dict

    ndim = imsd_dict.values.ndim

    if mode == 'weighted':
        std = np.nanstd(imsd_dict_local.values, axis = 1)
    else:
        std = np.ones(imsd_dict_local.values.shape[0])

    fit_slice = slice(start_index, end_index, stride)
    x_fit = imsd_dict_local.index.values[fit_slice]
    print("->",x_fit)
    weights = np.ones_like(x_fit)
    np.divide(1, std[fit_slice], where = std[fit_slice] != 0, out = weights)
    params = []
    fits = []
    for key, elem in tqdm(imsd_dict_local.items(), desc = "fitting weighted imsd curves") :
        y_fit = elem.values[fit_slice]
        m, b = np.polyfit(x_fit, y_fit, 1, w = weights)
        params.append([m, b])
        lin_fit_local = m * x_fit + b
        fits.append(lin_fit_local)
    print(m, b, x_fit, y_fit)
    params = np.vstack(params)
    m, b = params.T
    fits = np.vstack(fits).T
    if ndim == 1:
        return m[0], b[0], fits.flatten()
    else:
        return m, b, fits


@nb.njit
def isin(a,b) -> np.array:
    """
    this does the same thing as np.isin, but is implemented for numba since np.isin is not
    """
    out = np.empty(a.shape[0], dtype = nb.boolean)
    b = set(b)
    for i in nb.prange(a.shape[0]):
        if a[i] in b:
            out[i] = True
        else:
            out[i] = False
    return out
                    

@nb.jit(nb.void(nb.float64[:,:], nb.float64[:,:], nb.boolean[:,:], nb.float64, nb.int64), parallel = True)
def _fetch_particle_pairs_(tr_1, tr_2, matches, thresh, super_sampling = 1) -> None:
    """
    This matches particle pairs whose median displacement is within a given
    threshold distance

    Note if an array is super-sampled it should be the second argument
    """
    tr_1_particles = np.unique(tr_1[:,3])
    tr_2_particles = np.unique(tr_2[:,3])
    for i in nb.prange(tr_1_particles.shape[0]):
        _particle_1_ = tr_1_particles[i]
        tr_1_slice = np.where(tr_1[:,3] == _particle_1_)[0]
        tr_1_fr = tr_1[tr_1_slice,2]
        for j in nb.prange(tr_2_particles.shape[0]):
        #for j in range(tr_2_particles.shape[0]):
            _particle_2_  = tr_2_particles[j]
            tr_2_slice = np.where(tr_2[:,3] == _particle_2_)[0]
            tr_2_fr = tr_2[tr_2_slice,2]
            # Align frames (if super sampling != 1)
            aligned = np.where(tr_2_fr % super_sampling == 0)[0]
            intersect_frames = np.intersect1d(
                                          tr_1_fr, 
                                          tr_2_fr[aligned] // super_sampling
                                          )
            if len(intersect_frames) == 0:
                continue
            tr_1_fr_slice = isin(tr_1_fr, intersect_frames)
            tr_2_fr_slice = isin(tr_2_fr[aligned] // super_sampling, intersect_frames)
            y_1 = tr_1[tr_1_slice[tr_1_fr_slice],0]
            x_1 = tr_1[tr_1_slice[tr_1_fr_slice],1]
            y_2 = tr_2[tr_2_slice[aligned][tr_2_fr_slice],0]
            x_2 = tr_2[tr_2_slice[aligned][tr_2_fr_slice],1]
            dx = x_1-x_2
            dy = y_1-y_2
            dr = (dx**2+dy**2)**(1/2)
            if np.median(dr) < thresh:
                matches[i,j] = True


def associate_particles_recursive(
                                  n,
                                  idx, 
                                  rows = np.array([]),
                                  cols = np.array([]),
                                  match_matrix = []
                                  ) -> (np.array, np.array):
    """
    Recursive particle re-association fetches the entire set of particles from
    two datasets that are the same particle. This avoids nesting while loops,
    etc. 

    description: 
        n controls which dimension of the match matrix is being sliced
            n == 0 --> looking at a row, finding all column matches 
            n == 1 --> looking at a column, finding all row matches 
            
        if the current index is not in the aggregated rows/columns it is added
        for all the newly detected rows/columns iterate over them by self call

        once there are no more new values it breaks the recursion and returns

    There's probably a syntactax with fewer lines to express this, but I am
    trying to keep it readable

    Parameters:
    -----------
        n: int - 0 or 1
        idx: int - row or column index depending on n
        rows: np.array - aggregator of all the rows that gets passed through the recursion
        cols: np.array - aggregator of all the columns that gets passed through the recursion
        match_matrix: np.array (n,m) - array of all the match indices

    Returns:
    --------
        (np.array, np.array) - tuple of rows and columns that are matched 
        
    """
    if n == 0:
        n_p1 = 1 
        _cols_ = np.where(match_matrix[idx])[0]
        new_cols = _cols_[~np.isin(_cols_, cols)]
        cols = np.hstack([cols,new_cols])
        if ~np.isin(idx, rows):
            rows = np.hstack([rows, idx])
        for elem in new_cols:
            rows, cols = associate_particles_recursive(1,
                                                       elem,
                                                       rows = rows,
                                                       cols = cols,
                                                       match_matrix = match_matrix)
        return rows, cols
    elif n == 1:
        n_p1 = 0
        _rows_ = np.where(match_matrix[:,idx])[0]
        new_rows = _rows_[~np.isin(_rows_,rows)]
        rows = np.hstack([rows, new_rows])
        if ~np.isin(idx, cols):
            cols = np.hstack([cols, idx])
        for elem in new_rows:
            rows,cols = associate_particles_recursive(0,
                                                      elem,
                                                      rows = rows,
                                                      cols = cols,
                                                      match_matrix = match_matrix)
        return rows, cols
    else:
        assert False, f"invalid array dimension?: {n}"


def fetch_particle_pairs(
                         df_1: pd.DataFrame, 
                         df_2: pd.DataFrame,
                         thresh: float,
                         super_sampling: int
                         ) -> np.array:
    """
    wrapper for the call to _fetch_particle_pairs_ that preps the arrays and
    strips the particles out that did not have any matches.

    note -1 is the flag for unmatched so the entire array is seeded with -1
    which are expected to be overwritten by _fetch_particle_pairs_. if there are
    -1's in the matched array then something went wrong
    """
    tr_1 = df_1[['y','x','frame','particle']].values
    tr_2 = df_2[['y','x','frame','particle']].values
    tr_1_particles = np.unique(tr_1[:,3])
    tr_2_particles = np.unique(tr_2[:,3])
    matches = np.zeros([len(tr_1_particles),len(tr_2_particles)], dtype = bool)
    _fetch_particle_pairs_(tr_1, tr_2, matches, thresh, super_sampling = super_sampling)

    # Fetch all the associations and extrac the particle names that are
    # associated with them
    associations = []
    row_assoc, col_assoc = np.array([]), np.array([])
    for j in tqdm(range(matches.shape[0]), "rec associating"):
        row, col = associate_particles_recursive(0, j, match_matrix = matches)
        if len(row) > 0 and len(col) > 0:
            intersect_rows = np.intersect1d(row, row_assoc)
            intersect_cols = np.intersect1d(col, col_assoc)
            if len(intersect_rows) > 0 and len(intersect_cols) > 0:
                print("already found")
                print(row, intersect_rows, row_assoc)
                print(col, intersect_cols, col_assoc)
                continue
            tr_1_assoc = tr_1_particles[row.astype(int)]
            tr_2_assoc = tr_2_particles[col.astype(int)]
            associations.append([tr_1_assoc, tr_2_assoc])
            row_assoc = np.hstack([row, row_assoc])
            col_assoc = np.hstack([col, col_assoc])
    
    return associations


def re_link(tracks_1, tracks_2, matches, mode: str = 'frame match') -> tuple:
    """
    This takes two sets of matched tracks and turns them into two synchronized
    track dataframes where each particle index matches 1:1

    The first track dataframe is expected to be the more stable so only its
    unique values are used. This means that if for instance one of its
    particles becomes lost and re-associated with another particle that the
    latter track dataframe tracked more stably, it will not become confused by
    this edge case

    Parameters:
    -----------
        - tracks_1: pd.DataFrame - stable(r) particle track dataframe (output
                    of trackpy locate) 
        - tracks_2: pd.DataFrame - less stable particle track dataframe
        - matches: np.array [n,2] - array of matched particle indices where
                    first column is associated with tracks_1 and second column
                    is associated with tracks_2
    Returns:
    --------
        - (pd.DataFrame, pd.Dataframe): tracks_1 "matched", tracks_2 "matched"

    """
    matched_1 = []
    matched_2 = []
    for q, (idx_1, idx_2) in tqdm(enumerate(matches)):
        iterable = zip(
                       [idx_1,idx_2],
                       [tracks_1, tracks_2],
                       [matched_1, matched_2]
                       )
        for indices, track_dict, match in iterable:
            local_tracks = []
            for elem in indices:
                slice_ = track_dict['particle'].values == elem
                tracks_local = track_dict[slice_].copy()
                tracks_local['particle'] = q*np.ones(len(tracks_local), dtype = int)
                local_tracks.append(tracks_local.copy())
            local_tracks = pd.concat(local_tracks)
            u,c = np.unique(local_tracks['frame'].values, return_counts = True)
            duplicates = u[c > 1]
            if len(duplicates) > 0:
                print(f"found duplicates {duplicates}")
            local_tracks.drop_duplicates('frame', keep = 'first', inplace = True)
            match.append(local_tracks)

    # This section makes only tracks that are mutually shared ON FRAMES. 
    if mode == 'frame match':
        print("Asserting only passing particles that are in the same frame")
        desc = "iterating over particles"
        for q, (m1, m2) in tqdm(enumerate(zip(matched_1, matched_2)), desc = desc):
            t1 = m1['frame'].values
            t2 = m2['frame'].values
            intersect = np.intersect1d(t1,t2)
            m1_bool = np.isin(t1, intersect)
            m2_bool = np.isin(t2, intersect)
            matched_1[q] = m1[m1_bool].copy()
            matched_2[q] = m2[m2_bool].copy()
    else:
        print("Warning -> tracks that are not frame synced are shared here")

    return pd.concat(matched_1), pd.concat(matched_2)


def tracks_to_mask(input_tracks: pd.DataFrame, 
                   nz: int, 
                   nx: int,
                   ny: int,
                   size_multiple: float = 1.0
                   ) -> np.array:
    """
    Converts trackpy tracks into a boolean mask
    """
    #nz,nx,ny = input_array.shape
    mask = cp.zeros([nx,ny], dtype = bool)
    xx, yy = cp.meshgrid(cp.arange(ny), cp.arange(nx))
    xx = xx.astype(cp.float32)
    yy = yy.astype(cp.float32)
    mask = np.zeros([nz,nx,ny], dtype = bool)
    desc = "tracks -> mask"
    batch_size = 50
    for idx in tqdm(range(nz), desc = desc):
        fr_slice = input_tracks['frame'].values == idx
        y,x,size = cp.array(input_tracks[fr_slice][['y','x','size']].values.T)
        size *= size_multiple
        numel = len(x)
        n_batch = int(np.ceil(numel/batch_size))
        for q in range(n_batch):
            batch_slice = slice(q*batch_size,(q+1)*batch_size)
            radial_arr = cp.sqrt((xx[None,:,:] - x[batch_slice,None,None])**2 + (yy[None,:,:] - y[batch_slice,None,None])**2)
            bool_arr = radial_arr <= size[batch_slice,None,None]*2
            mask_local = bool_arr.sum(axis = 0)
            mask[idx] += mask_local.astype(bool).get()
    if nz > 1:
        return mask
    elif nz == 1:
        return mask[0]
    else:
        assert False, "mask generation failed"
        


def locate_hot_region(located_x, located_y, nx, ny, bin_size, threshold) -> tuple:
    """
    This is for locating image regions that are corrupted by stuck particles or
    structured noise particles... 

    inputs:
        located_x: trackpy x coordinates (np.array)
        located_y: trackpy y coordinates (np.array)
        nx,ny : image extents
        threshold: this function returns the coordinates where the histogram
                   exceeds the threshold
    """
    bins_x = np.arange(0, nx+1, bin_size)
    bins_y = np.arange(0, ny+1, bin_size)
    hist = np.histogram2d(located_x, located_y, bins = (bins_x, bins_y))
    return hist[0].T, np.vstack(np.where(hist[0].T > threshold)).T


class event_tracks:
    """
    This class is for handling the metavision tracks...
    """
    def __init__(self, input_data):
        if isinstance(input_data, str):
            self.data = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):
            self.data = input_data
        else:
            print("???")

    def _fetch_valid_indices_(self, min_length: int) -> np.array:
        """
        captures only the "valid" particles that meet the criteria greater than
        length...
        """
        data = self.data
        unique_elements = data['obj id'].unique()
        valid_indices = np.zeros(len(data['obj id'].values), dtype = bool)
        
        for elem in tqdm(unique_elements):
            indices = data['obj id'] == elem
            nnonzero = indices.sum()
            condition_0 = nnonzero < min_length
            condition_1 = data['x_double'].nunique() != nnonzero
            condition_2 = data['x_double'].nunique() != nnonzero
            # Remove shorter than minimum length
            if condition_0:# or condition_1 or condition_2:
                continue

            # The end of the recording can end on a float that rounds down to
            # the previous value so you need to remove the last value to avoid
            # a rounding error double instance of one time step.......
            remove_last_val = np.ones(nnonzero).astype(bool)
            remove_last_val[-1] = False
            valid_indices[indices] = remove_last_val
        return valid_indices

    def format_data_for_tp(self,
                           min_length: int,
                           update_frequency: float,
                           height: int = 720,
                           affine_matrix: np.array = None
                           ) -> pd.DataFrame:
        """
        This formats the data for processing by trackpy
        """
        valid_indices = self._fetch_valid_indices_(min_length)
        data = self.data
        #valid_indices = np.ones_like(self.data['obj id'].values).astype(bool)

        # Trackpy wants the "frames" to be sequential...
        normalized_frames = data['timestamp'].values - data['timestamp'].values[0]
        dt = 1e6 / update_frequency
        event_frames = np.round(normalized_frames / dt)


        if affine_matrix is not None:
            ones = np.ones_like(data['x_double'].values)
            new_coords = np.dot(affine_matrix, 
                                np.vstack([data['x_double'].values,
                                           data['y_double'].values,
                                           ones]))
        else:
            new_coords = np.vstack([data['x_double'].values,
                                    data['y_double'].values])

        df = pd.DataFrame.from_dict({
                'frame':event_frames[valid_indices].astype(int),
                'particle':data['obj id'][valid_indices].astype(int),
                'timestamp':data['timestamp'][valid_indices],
                'x':new_coords[0,valid_indices],
                'y':new_coords[1,valid_indices],
               }).drop_duplicates(subset = ['frame','particle'],
                                  keep = 'last')
    
        return df


def predict_curry(U, V, grid_size, default_x):
    """
    curried function to pass a quantized velocity field into trackpy to improve
    the linking

    usage:
        U,V = ....<use a function to calculate the velcotiy field>
        predictor = predict_curry(U ,V, grid_size, 0)
        tracks = tp.link(<dataframe>, search_range = <range>, memory = <mem>, predictor = predictor)
    """
    @tp.predict.predictor
    def predict(t1, particle):
        # This is supposed to be close to the velocity in fakeframe().
        # Note that the default order for coordinates in trackpy is (y, x).
        # You will rarely have to know this unless you are doing something
        # nerdy like making your own predictor.
        grid_y, grid_x = np.round((np.array(particle.pos) // grid_size)).astype(int)
        spatial_args = (grid_y, grid_x)
        v_y = V[spatial_args]
        v_x = U[spatial_args]
        if (v_x == 0 and v_y == 0):
            velocity = np.array([0, default_x])
        else:
            velocity = np.array([v_y, v_x])  # (v_y, v_x)
        return particle.pos + velocity * (t1 - particle.t)

    return predict


def generate_velocity_field(df_input, grid_size=10, img_width=None, img_height=None):
    """
    Converts tracked particle coordinates into a quantized 2D velocity field.
    """
    # 2. Sort to ensure chronological order for each particle
    df = df_input.copy()
    df.index.name = None
    df = df.sort_values(by=['particle', 'frame'])
    
    # 3. Calculate position and time differences
    # group by particle so we don't calculate differences between two distinct particles
    df['dx'] = df.groupby('particle')['x'].diff()
    df['dy'] = df.groupby('particle')['y'].diff()
    df['dt'] = df.groupby('particle')['frame'].diff()
    
    # Drop the first frame of each particle (where diffs are NaN) and zero-time steps
    df = df.dropna(subset=['dx', 'dy', 'dt'])
    df = df[df['dt'] > 0]
    
    # 4. Calculate instantaneous velocities (vx, vy)
    df['vx'] = df['dx'] / df['dt']
    df['vy'] = df['dy'] / df['dt']
    
    # Determine the bounding box if image dimensions aren't provided
    if img_width is None:
        img_width = int(np.ceil(df['x'].max()))
    if img_height is None:
        img_height = int(np.ceil(df['y'].max()))
        
    # 5. Quantize positions into grid cells
    # We use the starting 'x' and 'y' of the step to assign the velocity to a cell
    df['grid_x'] = (df['x'] // grid_size).astype(int)
    df['grid_y'] = (df['y'] // grid_size).astype(int)
    
    # 6. Aggregate velocities by grid cell
    # Calculate the mean velocity for all particles that passed through each cell
    cell_velocities = df.groupby(['grid_y', 'grid_x'])[['vx', 'vy']].median().reset_index()
    
    # Define grid dimensions
    max_grid_x = (img_width // grid_size) + 1
    max_grid_y = (img_height // grid_size) + 1
    
    # 7. Create dense matrices for U (x-velocity) and V (y-velocity)
    U = np.zeros((max_grid_y, max_grid_x))
    V = np.zeros((max_grid_y, max_grid_x))
    
    # Populate the matrices with our aggregated data
    U[cell_velocities['grid_y'], cell_velocities['grid_x']] = cell_velocities['vx']
    V[cell_velocities['grid_y'], cell_velocities['grid_x']] = cell_velocities['vy']
    
    # Create X, Y coordinate grids representing the center of each bin
    X, Y = np.meshgrid(
        np.arange(max_grid_x) * grid_size + (grid_size / 3), 
        np.arange(max_grid_y) * grid_size + (grid_size / 3)
    )
    
    return X, Y, U, V
