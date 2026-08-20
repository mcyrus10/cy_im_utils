#!/home/mcd4/miniforge3/envs/im_proc/bin/python
import numpy as np
from argparse import ArgumentParser
import trackpy as tp
from tqdm import tqdm


def water_viscosity(T, unit = 'C') -> float:
    """
    returns viscosity of water as a function of temperature
    Input temperature unit can be 'C' or 'K'
    Output is Dynamics viscosity in units of Pa*s

    Source for formula
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


def mean_displacement(diameter: float, 
                      micron_per_pixel: float, 
                      FPS: float,
                      T: float = 295) -> float:
    """
    What is the mean displacement of a particle of a given size?

    Parameters:
    ------------
        particle_size: float - nm
    """
    kb = 1.38e-23
    eta = water_viscosity(T, unit = 'K')
    diffusivity = kb*T/(3*np.pi*eta*diameter*1e-9)*1e12
    msd = 4*diffusivity/FPS
    sigma = np.sqrt(msd)/micron_per_pixel
    print(f"\tmicron per pixel: {micron_per_pixel} um px-1"
        + f"\n\tDiameter: {diameter} nm"
        + f"\n\tDiffusivity: {diffusivity} micron^2/s"
        + f"\n\tmsd: {msd} micron^2/s"
        + f"\n\tsigma: {sigma} px/frame"
        + f"\n\t3 sigma: {3*sigma} px/frame"
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--diameter",
                        type = float,
                        default = None,
                        help = "particle diameter in nm",
                        )
    parser.add_argument("--temperature",
                        dest = "T",
                        type = float,
                        default = 295,
                        help = "Temperature",
                        )
    parser.add_argument("--mpp",
                        type = float,
                        default = None,
                        help = "micron per pixel",
                        )
    parser.add_argument("--fps",
                        type = float,
                        default = None,
                        help = "Frames per second",
                        )

    return parser.parse_args()


def msd_with_velo(tracks, mpp, fps, velocity_label: str = 'y', fit_0: int = 0, 
                  fit_1: int = 5, T = 295) -> np.ndarray:
    """
    This subtracts the velocity from the tracks
    """
    kb,eta = 1.38e-23,water_viscosity(T, unit = "K")
    ft_slice = slice(fit_0,fit_1)
    D = []
    for part, df_ in tqdm(tracks.groupby("particle")):
        df = df_.copy()
        x,y,fr = df[['x','y','frame']].values.T
        if velocity_label == 'y':
            ft = np.polyfit(fr, y, 1)
        elif velocity_label == 'x':
            ft = np.polyfit(fr, x, 1)
        else:
            assert False, f"unknown column label: {velocity_label}"
        df[velocity_label] -= np.polyval(ft, fr)
        imsd = tp.imsd(df, mpp = mpp, fps = fps, max_lagtime = fit_1+1)
        x_ft, y_ft = imsd.index.values[ft_slice], imsd.values[ft_slice]
        diff_fit = np.polyfit(x_ft, y_ft, 1)
        D.append(diff_fit[0]/4)
    D = np.stack(D)
    diam = kb*T/(3*np.pi*eta*D*1e-12)*1e9
    return diam


if __name__ == "__main__":
    args = parse_args()
    mean_displacement(args.diameter, args.mpp, args.fps, args.T)
