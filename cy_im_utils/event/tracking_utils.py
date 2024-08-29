"""

                                TRACKING UTILS

"""

import numpy as np
from sys import path
path.append("/usr/lib/python3/dist-packages")

from metavision_sdk_core import PolarityFilterAlgorithm
from metavision_sdk_analytics import TrackingAlgorithm, TrackingConfig,\
                                     draw_tracking_results
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, \
                              TrailFilterAlgorithm, \
                              SpatioTemporalContrastAlgorithm


class null_filter:
    def __init__(self):
        pass

    def process_events(self, a, b):
        """
        If the first filter level is nothing just return the numpy version of
        the events?
        """
        pass

    def process_events_(self, a):
        pass


class event_filters:
    def __init__(self,
                 width: int,
                 height: int,
                 activity_time_ths: int,
                 activity_trail_ths: int,
                 stc_ths: int,
                 polarity_to_keep: int):
        """
        Obnoxious but this is just a wrapper to fetch the filtering
        operations....  with arbitrary permutations
        """
        if activity_time_ths > 0:
            print(f"USING ACTIVITY NOISE FILTER: {activity_time_ths}")
            self.activity_noise_filter = ActivityNoiseFilterAlgorithm(
                    width,
                    height,
                    activity_time_ths,
                    )
        else:
            self.activity_noise_filter = null_filter()

        if activity_trail_ths > 0:
            print(f"USING TRAIL FILTER: {activity_trail_ths}")
            self.trail_filter = TrailFilterAlgorithm(
                    width,
                    height,
                    activity_trail_ths 
                    )
        else:
            self.trail_filter = null_filter()

        if stc_ths > 0:
            print(f"USING SPATIO TEMPORAL CONTRAST FILTER: {stc_ths}")
            stc_cut_trail = False
            self.spatio_temporal_filter = SpatioTemporalContrastAlgorithm(
                                                                width,
                                                                height,
                                                                stc_ths,
                                                                stc_cut_trail)
        else:
            self.spatio_temporal_filter = null_filter()

        if polarity_to_keep != 0:
            print(f"USING POLARITY FILTER: keeping {polarity_to_keep}")
            self.polarity_filter = PolarityFilterAlgorithm(
                                                polarity = polarity_to_keep)
        else:
            self.polarity_filter = null_filter()

    def process_events(self, evs, events_buf):
        self.activity_noise_filter.process_events(evs, events_buf)
        self.trail_filter.process_events_(events_buf)
        self.spatio_temporal_filter.process_events_(events_buf)
        self.polarity_filter.process_events_(events_buf)
        #return evs, events_buf


def fetch_tracker(config: dict, width: int, height: int) -> TrackingAlgorithm:
    # Tracking Algorithm
    tracking_config = TrackingConfig()  # Default configuration

    cluster_maker = 'MedoidShift'
    if cluster_maker == 'SimpleGrid':
        tracking_config.cluster_maker = TrackingConfig.ClusterMaker.SimpleGrid
        tracking_config.cell_width = 5                     # 10 default
        tracking_config.cell_height = tracking_config.cell_width

    elif cluster_maker == 'MedoidShift':
        tracking_config.cluster_maker = TrackingConfig.ClusterMaker.MedoidShift
        tracking_config.medoid_shift_min_size = 2          # 2 default
        tracking_config.medoid_shift_spatial_dist = 5      # 5 default
        tracking_config.medoid_shift_temporal_dist = 10_000   # 10,000 default

    data_association = 'Nearest'
    if data_association == 'IOU':
        tracking_config.data_association = TrackingConfig.DataAssociation.IOU
        tracking_config.iou_max_dist = 25
    elif data_association == 'Nearest':
        tracking_config.data_association = TrackingConfig.DataAssociation.Nearest
        tracking_config.max_dist = 25

    motion_model = 'Smooth'
    if motion_model == 'Simple':
        tracking_config.motion_model = TrackingConfig.MotionModel.Simple
    elif motion_model == 'Instant':
        tracking_config.motion_model = TrackingConfig.MotionModel.Instant
    elif motion_model == 'Smooth':
        tracking_config.motion_model = TrackingConfig.MotionModel.Smooth
    elif motion_model == 'Kalman':
        tracking_config.motion_model = TrackingConfig.MotionModel.Kalman


    tracker = 'Ellipse'
    if tracker == 'Ellipse':
        tracking_config.tracker = TrackingConfig.Tracker.Ellipse
        #tracking_config.ellipse_tracker_update_function = TrackingConfig.EllipseUpdateFunction.Gaussian
        #tracking_config.ellipse_tracker_update_method = TrackingConfig.EllipseUpdateMethod.GaussianFitting

    elif tracker == 'ClusterKF':
        print("WARNING -> NOT USING ELLIPSE TRACKER")
        tracking_config.tracker = TrackingConfig.Tracker.ClusterKF
    else:
        assert False, "unknown tracker"

    
    tracking_algo = TrackingAlgorithm(sensor_width=width,
                                      sensor_height=height,
                                      tracking_config=tracking_config,
                                      )
    tracking_algo.min_size = config['tracking']['event']['min_size']
    tracking_algo.max_size = config['tracking']['event']['max_size']

    return tracking_algo


def imsd_powerlaw_fit(imsd_dict):
    """
    This performs the log-log fit on all the imsd curves
    """
    log_t = imsd_dict.index
    ones = np.ones_like(log_t)
    A = np.vstack([ones, log_t]).T
    b = np.log(im.values, where = im.values > 0)
    return np.linalg.lstsq(A,b)[0]
