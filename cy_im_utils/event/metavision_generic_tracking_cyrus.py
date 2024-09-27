#!/home/mcd4/miniconda3/envs/openeb/bin/python
# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License
# T&C's").  You may not use this file except in compliance with these License
# T&C's.  A copy of these License T&C's is located in the "licensing" folder
# accompanying this file.

"""

Cyrus customized version of "metavision_generic_tracking.py"

"""

import cv2
import numpy as np
import csv
from pathlib import Path
from sys import path
path.append("/usr/lib/python3/dist-packages")
path.append("/home/mcd4/cy_im_utils")
from cy_im_utils.event.tracking_utils import event_filters

from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import TrackingAlgorithm, TrackingConfig,\
                                     draw_tracking_results
from metavision_sdk_core import BaseFrameGenerationAlgorithm,\
                                RollingEventBufferConfig, RollingEventCDBuffer
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, \
                              TrailFilterAlgorithm, \
                              SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction,\
                              UIKeyEvent



def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
                     description='Generic Tracking sample.',
                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base options
    base_options = parser.add_argument_group('Base options')
    base_options.add_argument(
            '-i',
            '--input-event-file',
            dest='event_file_path',
            default="",
            help="Path to input event file (RAW or HDF5)")
    base_options.add_argument(
            '--process-from',
            dest='process_from',
            type=int,
            default=0,
            help='Start time to process events (in us).')
    base_options.add_argument(
            '--process-to',
            dest='process_to',
            type=int,
            default=None,
            help='End time to process events (in us).')
    parser.add_argument(
            '--update-frequency',
            dest='update_frequency',
            type=float,
            default=200.,
            help="Tracker's update frequency, in Hz.")
    parser.add_argument(
            '-a',
            '--acc-time',
            dest='accumulation_time_us',
            type=int,
            default=10000,
            help=('Duration of the time slice to store in the rolling event '
                  'buffer at each tracking step'))

    # Min/Max size options
    minmax_size_options = parser.add_argument_group('Min/Max size options')
    minmax_size_options.add_argument(
            '--min-size',
            dest='min_size',
            type=int,
            default=10,
            help='Minimal size of an object to track (in pixels).')
    minmax_size_options.add_argument(
            '--max-size',
            dest='max_size',
            type=int,
            default=300,
            help='Maximal size of an object to track (in pixels).')

    # Filtering options
    filter_options = parser.add_argument_group('Filtering options')
    filter_options.add_argument('--activity-time-ths',
        dest='activity_time_ths',
        type=int, default=10000,
        help=('Length of the time window for activity filtering (in us, '
              'disabled if == 0).'))
    filter_options.add_argument(
        '--activity-trail-ths',
        dest='activity_trail_ths',
        type=int,
        default=1000,
        help=('Length of the time window for trail filtering (in us, '
              'disabled if == 0).'))
    filter_options.add_argument(
            '--stc-thresh',
            dest='stc_thresh',
            type=int,
            default=10_000,
            help=('Length of the time window for spatio temporal contrast '
                  'filtering?'))
    filter_options.add_argument(
            '--polarity-to-keep',
            dest='polarity_to_keep',
            type=int,
            default=1,
            help="Filter out negative polarity values....."
            )

    tracker_options = parser.add_argument_group('Tracker options')
    tracker_options.add_argument(
            '--cluster-maker',
            type=str,
            default="",
            dest='cluster_maker',
            help="refer to metavision sdk"
            )
    tracker_options.add_argument(
            '--data-association',
            type=str,
            default="",
            dest='data_association',
            help="refer to metavision sdk"
            )
    tracker_options.add_argument(
            '--motion-model',
            type=str,
            dest='motion_model',
            default="",
            help="refer to metavision sdk"
            )
    tracker_options.add_argument(
            '--tracker',
            type=str,
            default="",
            dest='tracker',
            help="refer to metavision sdk"
            )
    tracker_options.add_argument(
            '--med-shift-min',
            type=float,
            default=2.0,
            dest='med_shift_min',
            help="refer to metavision sdk; default is TrackingConfig Default"
            )
    tracker_options.add_argument(
            '--med-shift-spatial',
            type=float,
            default=5.0,
            dest='med_shift_spatial',
            help="refer to metavision sdk; default is TrackingConfig Default"
            )
    tracker_options.add_argument(
            '--med-shift-temporal',
            type=float,
            default=10_000,
            dest='med_shift_temporal',
            help="refer to metavision sdk; default is TrackingConfig Default"
            )
    tracker_options.add_argument(
            '--data-assoc-dist',
            type=float,
            default=0.0,
            dest='data_assoc_dist',
            help="refer to metavision sdk"
            )


    # Outcome Options
    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument(
            '-o',
            '--out-video',
            dest='out_video',
            type=str, default="",
            help=("Path to an output AVI file to save the resulting video. A "
                  "frame is generated every time the tracking callback is "
                  "called."))

    # Replay Option
    replay_options = parser.add_argument_group('Replay options')
    replay_options.add_argument(
            '-f',
            '--replay_factor',
            type=float,
            default=1,
            help=("Replay Factor. If greater than 1.0 we replay with "
                  "slow-motion, otherwise this is a speed-up over real-time."))

    args = parser.parse_args()

    if args.process_to and args.process_from > args.process_to:
        print(("The processing time interval is not valid. "
               f"[{args.process_from,}, {args.process_to}]"))
        exit(1)

    return args


def fetch_tracking_config(cluster_maker: str,
                          data_association: str,
                          motion_model: str,
                          tracker: str,
                          med_shift_min: float,
                          med_shift_spatial: float,
                          med_shift_temporal: float,
                          data_association_distance: int
                          ) -> TrackingConfig:
    """
    Just using this to abstract away some of the craziness
    """
    tracking_config = TrackingConfig()  # Default configuration
    # CLUSTER MAKING
    if cluster_maker == 'SimpleGrid':
        tracking_config.cluster_maker = TrackingConfig.ClusterMaker.SimpleGrid
        tracking_config.cell_width = 5                     # 10 default
        tracking_config.cell_height = tracking_config.cell_width
    elif cluster_maker == 'MedoidShift':
        tracking_config.cluster_maker = TrackingConfig.ClusterMaker.MedoidShift
        tracking_config.medoid_shift_min_size = 2          # 2 default
        tracking_config.medoid_shift_spatial_dist = 5      # 5 default
        tracking_config.medoid_shift_temporal_dist = 5000   # 10,000 default
    elif cluster_maker == 'Default':
        pass
    else:
        assert False, "no cluster maker"

    
    # DATA ASSOCIATION
    # NOTE THE max_dist attribute goes with "Nearest"  and iou_max_dist goes
    # with "IOU"
    max_dist = data_association_distance
    if data_association == "Nearest":
        tracking_config.data_association = TrackingConfig.DataAssociation.Nearest
        tracking_config.max_dist = max_dist
    elif data_association == "IOU":
        tracking_config.data_association = TrackingConfig.DataAssociation.IOU
        tracking_config.iou_max_dist = max_dist
    elif data_association == "Default":
        pass
    else:
        assert False, "no data association"


    # TRACKER INITIALIZATION
    if motion_model == 'Simple':
        tracking_config.motion_model = TrackingConfig.MotionModel.Simple
    elif motion_model == 'Instant':
        tracking_config.motion_model = TrackingConfig.MotionModel.Instant
    elif motion_model == 'Smooth':
        tracking_config.motion_model = TrackingConfig.MotionModel.Smooth
    elif motion_model == 'Kalman':
        tracking_config.motion_model = TrackingConfig.MotionModel.Kalman
    elif motion_model == 'Default':
        pass
    else:
        assert False, "no motion model"

    if tracker == 'Ellipse':
        print("using ellipse tracker")
        tracking_config.tracker = TrackingConfig.Tracker.Ellipse
    elif tracker == 'ClusterKF':
        print("using cluster kf tracker")
        tracking_config.tracker = TrackingConfig.Tracker.ClusterKF
    elif tracker == 'Default':
        pass
    else:
        assert False, "no tracker"
    
    return tracking_config


class track_loop:
    """
    I basically made this class so that the window visualizaiton could be
    toggled off (performance boost hopefully.....?). 
    I don't think this is necessarily the best way to arrange these operations,
    but I don't like having functions with ambiguous scope and different
    versions of operations running around so this now the track_loop holds
    everything as member variables and can toggle the window on and off with a
    boolean.
    I really dislike the "process_tracking" function because it has mixed scope
    and does more than it should. it should be two functions....
    If this turns out to be a bad idea, it won't be hard to just go back to the
    OG and work back up to the earlier levels of abstraction
    """
    def __init__(self,
                 width,
                 height,
                 args,
                 EventLoop,
                 filter_ops,
                 rolling_buffer,
                 events_buf,
                 mv_iterator,
                 tracking_algo,
                 csv_f_name,
                 window_bool: bool = True
                 ):
        self.width = width
        self.height = height
        self.args = args
        self.EventLoop = EventLoop
        self.filter_ops = filter_ops
        self.rolling_buffer = rolling_buffer
        self.events_buf = events_buf
        self.mv_iterator = mv_iterator
        self.tracking_algo = tracking_algo
        self.csv_f_name = csv_f_name
        self.output_img = np.zeros((height, width, 3), np.uint8)
        self.tracking_results = tracking_algo.get_empty_output_buffer()
        self.window_bool = window_bool
        self.window = None

        if window_bool:
            self.operations_with_window()
        else:
            self.operations_no_window()

    def process_tracking(self, evs) -> None:
        if len(evs) != 0:
            self.rolling_buffer.insert_events(evs)
            self.tracking_algo.process_events(
                                             self.rolling_buffer, 
                                             self.tracking_results
                                             )


            if self.window_bool:
                BaseFrameGenerationAlgorithm.generate_frame(
                                             self.rolling_buffer,
                                             self.output_img)
                draw_tracking_results(evs['t'][-1],
                                      self.tracking_results,
                                      self.output_img)

                self.window.show_async(self.output_img)
                if self.args.out_video:
                    self.video_writer.write(self.output_img)

    def operations_with_window(self) -> None:
        with MTWindow(title="Cyrus Particle Tracking",
                      width=self.width, 
                      height=self.height, 
                      mode=BaseWindow.RenderMode.BGR) as self.window:
            self.window.show_async(self.output_img)

            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_name = f"{self.args.out_video}_{self.args.update_frequency}_fps.avi"
            self.video_writer = cv2.VideoWriter(video_name,
                                           fourcc,
                                           20,
                                           (self.width, self.height))

            # [GENERIC_TRACKING_MAIN_LOOP_BEGIN]
            # Process events
            log = self.event_loop()

            # [GENERIC_TRACKING_MAIN_LOOP_END]
            self.video_writer.release()
            print("Video has been saved in " + video_name)
            self.post_process(log)

    def operations_no_window(self) -> None:
        log = self.event_loop()
        self.post_process(log)

    def event_loop(self) -> list:
        """

        """
        log = []
        for j, evs in enumerate(self.mv_iterator):
            # Dispatch system events to the window
            self.EventLoop.poll_and_dispatch()
            self.filter_ops.process_events(evs, self.events_buf)
            self.process_tracking(self.events_buf.numpy())

            if self.window_bool:
                if self.window.should_close():
                    break

            for elem in self.tracking_results.numpy():
                x_int = int(elem[0])
                y_int = int(elem[1])
                timestamp = float(elem[2])
                double_x = float(elem[3])
                double_y = float(elem[4])
                width = int(elem[5])
                height = int(elem[6])
                obj_id = int(elem[7])
                event_id = int(elem[8])
                log.append([x_int,y_int,timestamp, double_x, double_y, width, 
                            height, obj_id, event_id])
        return log

    def post_process(self, log) -> None:
        # output logging
        header = ['x','y','timestamp','x_double','y_double','width',
                  'height','obj id','event_id']
        with open(f"track_output/{self.csv_f_name}", mode = 'w') as f:
            data_writer = csv.writer(f)
            data_writer.writerow(header)
            data_writer.writerows(log)


def main():
    """ Main """
    args = parse_args()
    window_bool = True

    csv_f_name = (
                  "sim_tracking"
                  f"_update_frequency_{args.update_frequency}"
                  f"_acc_time_{args.accumulation_time_us}"
                  f"_med_shift_min_{args.med_shift_min}"
                  f"_med_shift_spatial_{args.med_shift_spatial}"
                  f"_da_{args.data_association}"
                  f"_da_distance_{args.data_assoc_dist}"
                  f"_cm_{args.cluster_maker}"
                  f"_ptk_{args.polarity_to_keep}"
                  f"_mm_{args.motion_model}"
                  f"_stc_thresh_{args.stc_thresh}"
                  f"_tracker_{args.tracker}"
                  f"_med_shift_temporal_{args.med_shift_temporal}"
                  ".csv"
                  )

    #if Path(f"track_output/{csv_f_name}").is_file():
    #    print("File Exists:")
    #    print(f"\t{csv_f_name}")
    #    return None


    # [GENERIC_TRACKING_CREATE_ROLLING_BUFFER_BEGIN]
    # Rolling event buffer
    buffer_config = RollingEventBufferConfig.make_n_us(args.accumulation_time_us)
    rolling_buffer = RollingEventCDBuffer(buffer_config)
    # [GENERIC_TRACKING_CREATE_ROLLING_BUFFER_END]

    # [GENERIC_TRACKING_CREATE_ITERATOR_BEGIN]
    # Events iterator on Camera or event file
    delta_t = int(1_000_000 / args.update_frequency)
    print("delta t =  ", delta_t)
    max_duration=args.process_to - args.process_from if args.process_to else None
    mv_iterator = EventsIterator(input_path=args.event_file_path,
                                 start_ts=args.process_from,
                                 max_duration=max_duration,
                                 delta_t=delta_t,
                                 mode="delta_t")
    # [GENERIC_TRACKING_CREATE_ITERATOR_END]
    if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, 
                                               replay_factor=args.replay_factor)
    height, width = mv_iterator.get_size()  # Camera Geometry
    image_buffer = np.zeros([width, height, 3], dtype = np.uint8)

    # Noise + Trail filter that will be applied to events
    activity_noise_filter = ActivityNoiseFilterAlgorithm(width,
                                                         height,
                                                         args.activity_time_ths)
    trail_filter = TrailFilterAlgorithm(width,
                                        height,
                                        args.activity_trail_ths)

    stc_cut_trail = True
    spatio_temporal_filter = SpatioTemporalContrastAlgorithm(width,
                                                             height,
                                                             args.stc_thresh,
                                                             stc_cut_trail)


    tracking_config = fetch_tracking_config(
            cluster_maker = args.cluster_maker,
            data_association = args.data_association,
            motion_model = args.motion_model,
            tracker = args.tracker,
            med_shift_min = args.med_shift_min,
            med_shift_spatial = args.med_shift_spatial,
            med_shift_temporal = args.med_shift_temporal,
            data_association_distance = args.data_assoc_dist
            )


    tracking_algo = TrackingAlgorithm(sensor_width=width,
                                      sensor_height=height,
                                      tracking_config=tracking_config,
                                      )
    tracking_algo.min_size = args.min_size
    tracking_algo.max_size = args.max_size

    output_img = np.zeros((height, width, 3), np.uint8)

    events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
    tracking_results = tracking_algo.get_empty_output_buffer()

    filter_ops = event_filters(width,
                               height,
                               args.activity_time_ths,
                               args.activity_trail_ths,
                               args.stc_thresh,
                               args.polarity_to_keep)
    # [GENERIC_TRACKING_MAIN_PROCESSING_BEGIN]
    def process_tracking(evs):
        if len(evs) != 0:
            rolling_buffer.insert_events(evs)
            tracking_algo.process_events(rolling_buffer, tracking_results)
            BaseFrameGenerationAlgorithm.generate_frame(rolling_buffer,
                                                        output_img)
            draw_tracking_results(evs['t'][-1], tracking_results, output_img)

        window.show_async(output_img)
        if args.out_video:
            video_writer.write(output_img)
    # [GENERIC_TRACKING_MAIN_PROCESSING_END]

    track_loop(width,
               height,
               args,
               EventLoop,
               filter_ops,
               rolling_buffer,
               events_buf,
               mv_iterator,
               tracking_algo,
               csv_f_name,
               window_bool = window_bool)
    
    """

    # Window - Graphical User Interface (Display tracking results and process keyboard events)
    with MTWindow(title="Cyrus Particle Tracking",
                  width=width, 
                  height=height, 
                  mode=BaseWindow.RenderMode.BGR) as window:
        window.show_async(output_img)

        if False:#args.out_video:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_name = f"{args.out_video}_{args.update_frequency}_fps.avi"
            video_writer = cv2.VideoWriter(video_name,
                                           fourcc,
                                           20,
                                           (width, height))

        # [GENERIC_TRACKING_MAIN_LOOP_BEGIN]
        # Process events
        log = []
        for j, evs in enumerate(mv_iterator):
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()
            filter_ops.process_events(evs, events_buf)
            process_tracking(events_buf.numpy())

            if window.should_close():
                break

            for elem in tracking_results.numpy():
                x_int = int(elem[0])
                y_int = int(elem[1])
                timestamp = float(elem[2])
                double_x = float(elem[3])
                double_y = float(elem[4])
                width = int(elem[5])
                height = int(elem[6])
                obj_id = int(elem[7])
                event_id = int(elem[8])
                log.append([x_int,y_int,timestamp, double_x, double_y, width, 
                            height, obj_id, event_id])

        # [GENERIC_TRACKING_MAIN_LOOP_END]
        if False:#args.out_video:
            video_writer.release()
            print("Video has been saved in " + video_name)

        # output logging
        header = ['x','y','timestamp','x_double','y_double','width',
                  'height','obj id','event_id']
        with open(f"track_output/{csv_f_name}", mode = 'w') as f:
            data_writer = csv.writer(f)
            data_writer.writerow(header)
            data_writer.writerows(log)
    """


if __name__ == "__main__":
    main()
