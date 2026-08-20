import numpy as np
import matplotlib.pyplot as plt


def sweep_update_factor(scale, filter_length, napari_instance, numel: int = 20, plot: bool = True, idx_0: int = 0):
    """
    Sweep over the update factor while keeping the scale and filter length fixed
    to find the "elbow"
    """
    n_ev_tot = np.sum(napari_instance.__fetch_layer__("event").data[idx_0:-1] != 0)
    pct = []
    update_factors = np.logspace(np.log10(0.01),np.log10(0.99),numel)
    for update_factor in update_factors:
        napari_instance._preview_event_noise_filter_()(
                scale = scale,
                filter_length = filter_length,
                update_factor = update_factor,
                interpolation_method = 3,
                n_images = napari_instance.__fetch_layer__("event").data.shape[0]-1,
                )
        im_handle = np.abs(napari_instance.__fetch_layer__("event filter preview").data[idx_0:])
        n_ev_filtered = np.sum(im_handle != 0)
        pct.append([update_factor, n_ev_filtered/n_ev_tot])

    if plot:
        print(["[INFO] Plotting Elbow Sweep"])
        plot_pct(pct)
    
    return pct


def plot_pct(pct) -> None:
    """
    helper function to visualize elbow, etc.
    """
    pct = np.vstack(pct)
    probe_idx = 0
    fig,ax = plt.subplots(1,2)
    x_ft = pct[np.array([0,-1]),probe_idx]/pct[-1, probe_idx]
    y_ft = pct[np.array([0,-1]),-1]
    x1,x2 = x_ft
    x0 = pct[:,probe_idx]/pct[-1,probe_idx]
    y1,y2 = y_ft
    y0 = pct[:,-1]
    ft = np.polyfit(x_ft, y_ft, 1)

    dy = (pct[:,-1] - pct[0,-1])
    dx = (pct[:,probe_idx] - pct[0,probe_idx]) / pct[-1,probe_idx]
    dr = (dy**2+dx**2)**(1/2)
    elbow_dist = np.sin(np.arctan2(dy,dx) - np.arctan(ft[0]))*dr

    elbow_dist_2 = np.abs(((x2-x1)*(y1-y0) - (x1 - x0)*(y2-y1))) / (((x2-x1)**2+(y2-y1)**2)**(1/2))
    label = "update factor"

    ax[0].scatter(pct[:,0]/pct[-1,0], pct[:,-1])
    ax[0].plot(x_ft, np.polyval(ft, x_ft), 'k--')
    ax[0].set_xlabel(f"{label} normalized")
    twiny = ax[0].twiny()
    twiny.plot(pct[:,probe_idx], pct[:,-1])
    twiny.set_xlabel(label)
    ax[1].plot(pct[:,probe_idx],elbow_dist, marker = '+')
    ax[1].plot(pct[:,probe_idx],elbow_dist_2, marker = 'x')
    twiny = ax[1].twiny()
    twiny.plot(elbow_dist, marker = '.')
    ax[1].set_xlabel(label)
    ax[0].set_ylabel("n filtered/n unfiltered")