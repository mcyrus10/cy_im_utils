import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit, void, uint16, float64, int64, boolean

class EventFilterInterpolation:
    def __init__(self, frame_size, filter_length, scale, update_factor, 
                 filtered_ts, interpolation_method):
        """
        Interpolation Methods: 
            - 0 : bilinear
            - 1 : bilinear with interval weights
            - 2 : max
            - 3 : distance
        """
        self.FrameSize = frame_size
        self.FilterLength = filter_length
        self.Scale = scale
        self.UpdateFactor = update_factor
        
        sz_0 = int(np.floor(frame_size[0] / scale))
        sz_1 = int(np.floor(frame_size[1] / scale))
        
        self.TimestampMap = np.zeros([sz_0, sz_1])
        self.IntervalMap = 1e4 * np.ones([sz_0, sz_1])
        self.ActiveMap = np.zeros([sz_0, sz_1], dtype = bool)
        self.ValidEvents = 0
        self.InvalidEvents = 0
        self.CurrentTs = 0
        self.FilteredTs = filtered_ts
        self.InterpolationMethod = interpolation_method

    def processEvents(self, events):
        """
        event = np.array([[x, y, p, t]] ???????
        """
        eventsBin = np.zeros(events.shape[0], dtype = bool)
        for i, event in tqdm(enumerate(events), disable = True):
            eventsBin[i] = self.filterEvent(event)
            self.updateFeatures(event)
            self.CurrentTs = event[3]
        
        self.updateEmpty()
        self.eventsBin = eventsBin
        self.EventsTrue = events[eventsBin]
        self.EventsFalse = events[~eventsBin]
        self.ValidEvents = self.ValidEvents + self.EventsTrue.shape[0]
        self.InvalidEvents = self.InvalidEvents + self.EventsFalse.shape[0]

    def filterEvent(self, event):
        match self.InterpolationMethod:
            case 0:
                thr_ts = self.bilinear_interpolation(event[1], event[0])
            case 1:
                thr_ts = self.bilinear_interval_weights_interpolation(event[1], event[0])
            case 2:
                thr_ts = self.max_interpolation(event[1], event[0])
            case 3:
                thr_ts = self.distance_interpolation(event[1], event[0])
            case _:
                assert False, "unknown interpolation method...."
        
        diff_ts = event[3] - thr_ts
        correct = diff_ts < self.FilterLength
        return correct

    def updateFeatures(self, event):
        self.updateInterval(event)
        self.updateFilteredTimestamp(event)
        self.updateActive(event)

    def updateInterval(self, event):
        cellY = int(np.floor(event[1]/self.Scale))
        cellX = int(np.floor(event[0]/self.Scale))
        cell_time = self.TimestampMap[cellY, cellX]
        cell_interval = self.IntervalMap[cellY, cellX]
        time_diff = event[3] - cell_time
        new_interval = cell_interval * (1 - self.UpdateFactor) + time_diff * self.UpdateFactor;
        self.IntervalMap[cellY, cellX] = new_interval

    def updateFilteredTimestamp(self, event):
        cellY = int(np.floor(event[1]/self.Scale))
        cellX = int(np.floor(event[0]/self.Scale))
        cell_filtered_time = self.TimestampMap[cellY, cellX]
        new_filtered_time = cell_filtered_time * (1 - self.UpdateFactor) + event[3] * self.UpdateFactor
        self.TimestampMap[cellY, cellX] = new_filtered_time
   
    def updateActive(self, event):
        cellY = int(np.floor(event[1]/self.Scale))
        cellX = int(np.floor(event[0]/self.Scale))
        self.ActiveMap[cellY, cellX] = True

    def updateEmpty(self):
        scaled_size = self.ActiveMap.shape
        for y in range(scaled_size[0]):
            for x in range(scaled_size[1]):
                if ~self.ActiveMap[y, x]:
                    # Update interval
                    cell_time = self.TimestampMap[y, x];
                    cell_interval = self.IntervalMap[y, x];
                    time_diff = self.CurrentTs - cell_time;
                    new_interval = cell_interval * (1 - self.UpdateFactor) + time_diff * self.UpdateFactor;
                    self.IntervalMap[y, x] = new_interval;
                    # Update filtered timestamp
                    new_filtered_time = cell_time * (1 - self.UpdateFactor) + self.CurrentTs * self.UpdateFactor;
                    self.TimestampMap[y, x] = new_filtered_time;

        # What does this do? The array is 2d but 3 indices???
        #self.ActiveMap[:, :, 5] = False;

    def displayFilteredTimestamp(self):
        diff_map = self.CurrentTs - self.TimestampMap;
        exp_map = 255 * np.exp(-diff_map / 100);
        exp_map = imresize(exp_map, self.FrameSize, "nearest");
        plg.figure(2);
        plt.imshow(exp_map);
        plt.gcf().suptitle('Filtered timestamp');

    def returnFilteredTimestamp(self):
        print("use skimage resize?")
        diff_map = self.CurrentTs - self.TimestampMap;
        exp_map = 255 * np.exp(-diff_map / 100);
        exp_map = imresize(exp_map, self.FrameSize, "nearest");
        return exp_map
        
    def displayInterval(self):
        exp_map = 255 * np.exp(-self.IntervalMap / 20);
        exp_map = imresize(exp_map, self.FrameSize, "nearest");
        plt.figure(3)
        plt.imshow(exp_map);
        plt.gcf().suptitle('Event intervals');
    
    def returnInterval(self):
        exp_map = 255 * np.exp(-self.IntervalMap / 20);
        exp_map = imresize(exp_map, self.FrameSize, "nearest");
        return exp_map

    def displayFeatures(self):
        image_rgb = np.zeros(self.FrameSize[0], self.FrameSize[1], 3);
        image_rgb[:, :, 0] = self.returnFilteredTimestamp();
        image_rgb[:, :, 1] = self.returnInterval();
        plt.figure(4);
        plt.imshow(image_rgb);
        plt.gcf().suptitle('Features');


    def displayFilter(self):
        image = self.fetch_image()
        plt.figure(5);
        plt.imshow(image);
        self.ImageData = image;

    def fetch_image(self):
        image = 255*np.ones([self.FrameSize[0], self.FrameSize[1], 3], dtype = np.uint8)
        if ~isempty(self.EventsFalse):
            for ind in range(self.EventsFalse.shape[0]):
                x_ = self.EventsFalse[ind, 1]
                y_ = self.EventsFalse[ind, 1]
                image[x_, y_, :] = [255, 0, 0];
        if ~isempty(self.EventsTrue):
            for ind in range(self.EventsTrue.shape[0]):
                x_ = self.EventsTrue[ind, 1]
                y_ = self.EventsTrue[ind, 0]
                image[x_, y_, :] = [0, 255, 0];
        return image

    def distance_interpolation(self, y, x):
        y_cell = int(np.floor(y / self.Scale))
        x_cell = int(np.floor(x / self.Scale))
        w = self.Scale
    
        if (y < self.Scale / 2) or (y >= self.FrameSize[0] - self.Scale / 2):
            
            if (x < self.Scale / 2) or (x >= self.FrameSize[1] - self.Scale / 2):
                ts_filtered_cell = self.TimestampMap[y_cell, x_cell]
                thr = ts_filtered_cell
                
            else:
                i_floor = int(np.floor((x - w / 2) / w))
                dx1 = x - w / 2 - i_floor * w + 0.5
                dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5
    
                if (y < self.Scale / 2):
                    dy =  w / 2 - y - 0.5
                else:
                    j_floor = int(np.floor((y - w / 2) / w))
                    dy = y - w / 2 - j_floor * w + 0.5
    
                dist_1 = np.sqrt(dx1**2 + dy**2)
                dist_2 = np.sqrt(dx2**2 + dy**2)
    
                interval_1 = self.IntervalMap[y_cell, i_floor + 0]
                interval_2 = self.IntervalMap[y_cell, i_floor + 1]
    
                C_1 = dist_2 * interval_2
                C_2 = dist_1 * interval_1
    
                coef_sum = C_1 + C_2
    
                ts_filtered_cell_11 = self.TimestampMap[y_cell, i_floor + 0]
                ts_filtered_cell_12 = self.TimestampMap[y_cell, i_floor + 1]
    
                thr = (ts_filtered_cell_11 * C_1 + ts_filtered_cell_12 * C_2) / coef_sum
        else:
            if (x < self.Scale / 2) or (x >= self.FrameSize[1] - self.Scale / 2):
                if (x < self.Scale / 2):
                    dx = w / 2 - x - 0.5
                else:
                    i_floor = int(np.floor((x - w / 2) / w))
                    dx = x - w / 2 - i_floor * w + 0.5
                
                j_floor = int(np.floor((y - w / 2) / w))
                dy1 = y - w / 2 - j_floor * w + 0.5
                dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5
    
                dist_1 = np.sqrt(dx**2 + dy1**2)
                dist_2 = np.sqrt(dx**2 + dy2**2)
    
                interval_1 = self.IntervalMap[j_floor + 0, x_cell]
                interval_2 = self.IntervalMap[j_floor + 1, x_cell]
    
                C_1 = dist_2 * interval_2
                C_2 = dist_1 * interval_1
    
                coef_sum = C_1 + C_2
    
                ts_filtered_cell_11 = self.TimestampMap[j_floor + 0, x_cell]
                ts_filtered_cell_21 = self.TimestampMap[j_floor + 1, x_cell]
    
                thr = (ts_filtered_cell_11 * C_1 + ts_filtered_cell_21 * C_2) / coef_sum
    
            else:
                i_floor = int(np.floor((x - w / 2) / w))
                j_floor = int(np.floor((y - w / 2) / w))
                dx1 = x - w / 2 - i_floor * w + 0.5
                dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5
                dy1 = y - w / 2 - j_floor * w + 0.5
                dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5
    
                dist_11 = np.sqrt(dx1**2 + dy1**2)
                dist_12 = np.sqrt(dx2**2 + dy1**2)
                dist_21 = np.sqrt(dx1**2 + dy2**2)
                dist_22 = np.sqrt(dx2**2 + dy2**2)
    
                interval_11 = self.IntervalMap[j_floor + 0, i_floor + 0]
                interval_12 = self.IntervalMap[j_floor + 0, i_floor + 1]
                interval_21 = self.IntervalMap[j_floor + 1, i_floor + 0]
                interval_22 = self.IntervalMap[j_floor + 1, i_floor + 1]
    
                coef_11 = dist_11 * interval_11
                coef_12 = dist_12 * interval_12
                coef_21 = dist_21 * interval_21
                coef_22 = dist_22 * interval_22
    
                C_11 = coef_12 * coef_21 * coef_22
                C_12 = coef_11 * coef_21 * coef_22
                C_21 = coef_11 * coef_12 * coef_22
                C_22 = coef_11 * coef_12 * coef_21
    
                coef_sum = C_11 + C_12 + C_21 + C_22
                
                ts_11 = self.TimestampMap[j_floor + 0, i_floor + 0]
                ts_12 = self.TimestampMap[j_floor + 0, i_floor + 1]
                ts_21 = self.TimestampMap[j_floor + 1, i_floor + 0]
                ts_22 = self.TimestampMap[j_floor + 1, i_floor + 1]
    
                thr = (ts_11*C_11 + ts_12 * C_12 + ts_21 * C_21 + ts_22 * C_22) / coef_sum
        return thr

    def max_interpolation(self, y, x):
        y_cell = int(np.floor(y / self.Scale))
        x_cell = int(np.floor(x / self.Scale))
        w = self.Scale
    
        if (y < self.Scale / 2) or (y >= self.FrameSize[0] - self.Scale / 2):
            if (x < self.Scale / 2) or (x >= self.FrameSize[1] - self.Scale / 2):
                ts_filtered_cell = self.TimestampMap[y_cell, x_cell]
                thr = ts_filtered_cell
            else:
                i_floor = int(np.floor((x - w / 2) / w))
                ts_filtered_cell_11 = self.TimestampMap[y_cell, i_floor]
                ts_filtered_cell_12 = self.TimestampMap[y_cell, i_floor + 1]
                thr = np.max(np.array([ts_filtered_cell_11, ts_filtered_cell_12]))
        else:
            if (x < self.Scale / 2) or (x >= self.FrameSize[1] - self.Scale / 2):
                j_floor = int(np.floor((y - w / 2) / w))
                ts_filtered_cell_11 = self.TimestampMap[j_floor , x_cell]
                ts_filtered_cell_21 = self.TimestampMap[j_floor + 1, x_cell]
                thr = np.max(np.array([ts_filtered_cell_11, ts_filtered_cell_21]))
    
            else:
                i_floor = int(np.floor((x - w / 2) / w))
                j_floor = int(np.floor((y - w / 2) / w))
                
                ts_filtered_cell_11 = self.TimestampMap[j_floor , i_floor ]
                ts_filtered_cell_12 = self.TimestampMap[j_floor , i_floor + 1]
                ts_filtered_cell_21 = self.TimestampMap[j_floor + 1, i_floor ]
                ts_filtered_cell_22 = self.TimestampMap[j_floor + 1, i_floor + 1]
    
                thr = np.max(np.array([ts_filtered_cell_11,
                                       ts_filtered_cell_12, 
                                       ts_filtered_cell_21,
                                       ts_filtered_cell_22]))
        return thr

    def bilinear_interpolation(self, y, x):
        y_cell = int(np.floor(y / self.Scale))
        x_cell = int(np.floor(x / self.Scale))
        w = self.Scale
    
        if (y < self.Scale / 2) or (y >= self.FrameSize[0] - self.Scale / 2):
            if (x < self.Scale / 2) or (x >= self.FrameSize[1] - self.Scale / 2):
                ts_filtered_cell = self.TimestampMap[y_cell, x_cell]
                thr = ts_filtered_cell
            else:
                i_floor = int(np.floor((x - w / 2) / w))
                dx1 = x - w / 2 - i_floor * w + 0.5
                dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5
        
                ts_filtered_cell_11 = self.TimestampMap[y_cell, i_floor + 0]
                ts_filtered_cell_12 = self.TimestampMap[y_cell, i_floor + 1]
        
                thr = ts_filtered_cell_11 * (dx2 / w) + ts_filtered_cell_12 * (dx1 / w)
        else:
            if (x < self.Scale / 2) or (x >= self.FrameSize[1] - self.Scale / 2):
                j_floor = int(np.floor((y - w / 2) / w))
                dy1 = y - w / 2 - j_floor * w + 0.5
                dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5
    
                ts_filtered_cell_11 = self.TimestampMap[j_floor + 0, x_cell]
                ts_filtered_cell_21 = self.TimestampMap[j_floor + 1, x_cell]
    
                thr = ts_filtered_cell_11 * (dy2 / w) + ts_filtered_cell_21 * (dy1 / w)
    
            else:
                i_floor = int(np.floor((x - w / 2) / w))
                j_floor = int(np.floor((y - w / 2) / w))
                dx1 = x - w / 2 - i_floor * w + 0.5
                dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5
                dy1 = y - w / 2 - j_floor * w + 0.5
                dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5
    
                ts_filtered_cell_11 = self.TimestampMap[j_floor + 0, i_floor + 0]
                ts_filtered_cell_12 = self.TimestampMap[j_floor + 0, i_floor + 1]
                ts_filtered_cell_21 = self.TimestampMap[j_floor + 1, i_floor + 0]
                ts_filtered_cell_22 = self.TimestampMap[j_floor + 1, i_floor + 1]
    
                thr1 = ts_filtered_cell_11 * (dx2 / w) + ts_filtered_cell_12 * (dx1 / w)
                thr2 = ts_filtered_cell_21 * (dx2 / w) + ts_filtered_cell_22 * (dx1 / w)
                thr = thr1 * (dy2 / w) + thr2 * (dy1 / w)
        return thr

    def bilinear_interval_weights_interpolation(self, y, x):
        y_cell = int(np.floor(y / self.Scale))
        x_cell = int(np.floor(x / self.Scale))
        w = self.Scale
    
        if (y < self.Scale / 2) or (y >= self.FrameSize[0] - self.Scale / 2):
            if (x < self.Scale / 2) or (x >= self.FrameSize[1] - self.Scale / 2):
                ts_filtered_cell = self.TimestampMap[y_cell, x_cell]
                thr = ts_filtered_cell
            else:
                i_floor = int(np.floor((x - w / 2) / w))
                dx1 = x - w / 2 - i_floor * w + 0.5
                dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5
    
                interval_11 = self.IntervalMap[y_cell, i_floor ]
                interval_12 = self.IntervalMap[y_cell, i_floor + 1]
    
                coef_11 = dx2 * interval_12
                coef_12 = dx1 * interval_11
    
                sum_top = coef_11 + coef_12
    
                ts_filtered_cell_11 = self.TimestampMap[y_cell, i_floor ]
                ts_filtered_cell_12 = self.TimestampMap[y_cell, i_floor + 1]
    
                thr = ts_filtered_cell_11 * coef_11 / sum_top + ts_filtered_cell_12 * coef_12 / sum_top
        else:
            if (x < self.Scale / 2) or (x >= self.FrameSize[1] - self.Scale / 2):
                j_floor = int(np.floor((y - w / 2) / w))
                dy1 = y - w / 2 - j_floor * w + 0.5
                dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5
    
                interval_11 = self.IntervalMap[j_floor , x_cell]
                interval_21 = self.IntervalMap[j_floor + 1, x_cell]
    
                coef_top = dy2 * interval_21
                coef_bot = dy1 * interval_11
    
                sum_all = coef_top + coef_bot
    
                ts_filtered_cell_11 = self.TimestampMap[j_floor , x_cell]
                ts_filtered_cell_21 = self.TimestampMap[j_floor + 1, x_cell]
    
                thr = ts_filtered_cell_11 * coef_top / sum_all + ts_filtered_cell_21 * coef_bot / sum_all
    
            else:
                i_floor = int(np.floor((x - w / 2) / w))
                j_floor = int(np.floor((y - w / 2) / w))
                dx1 = x - w / 2 - i_floor * w + 0.5
                dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5
                dy1 = y - w / 2 - j_floor * w + 0.5
                dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5
    
                interval_11 = self.IntervalMap[j_floor , i_floor]
                interval_12 = self.IntervalMap[j_floor , i_floor + 1]
                interval_21 = self.IntervalMap[j_floor + 1, i_floor]
                interval_22 = self.IntervalMap[j_floor + 1, i_floor + 1]
    
                interval_top = interval_11 * interval_12
                interval_bot = interval_21 * interval_22
    
                coef_11 = dx2 * interval_12
                coef_12 = dx1 * interval_11
                coef_21 = dx2 * interval_22
                coef_22 = dx1 * interval_21
    
                sum_top = coef_11 + coef_12
                sum_bot = coef_21 + coef_22
    
                coef_top = dy2 * interval_bot
                coef_bot = dy1 * interval_top
                sum_all = coef_top + coef_bot
                
                ts_filtered_cell_11 = self.IntervalMap[j_floor + 0, i_floor + 0]
                ts_filtered_cell_12 = self.IntervalMap[j_floor + 0, i_floor + 1]
                ts_filtered_cell_21 = self.IntervalMap[j_floor + 1, i_floor + 0]
                ts_filtered_cell_22 = self.IntervalMap[j_floor + 1, i_floor + 1]
    
                thr1 = ts_filtered_cell_11 * coef_11 / sum_top + ts_filtered_cell_12 * coef_12 / sum_top
                thr2 = ts_filtered_cell_21 * coef_21 / sum_bot + ts_filtered_cell_22 * coef_22 / sum_bot
                thr = thr1 * coef_top / sum_all + thr2 * coef_bot / sum_all
        return thr


@njit(float64(uint16, uint16, int64, int64[:], float64[:,:]))
def _max_interpolation_(y, x, Scale, FrameSize, TimestampMap):
    y_cell = int(np.floor(y / Scale))
    x_cell = int(np.floor(x / Scale))
    w = Scale

    if (y < Scale / 2) or (y >= FrameSize[0] - Scale / 2):
        if (x < Scale / 2) or (x >= FrameSize[1] - Scale / 2):
            ts_filtered_cell = TimestampMap[y_cell, x_cell]
            thr = ts_filtered_cell
        else:
            i_floor = int(np.floor((x - w / 2) / w))
            ts_filtered_cell_11 = TimestampMap[y_cell, i_floor]
            ts_filtered_cell_12 = TimestampMap[y_cell, i_floor + 1]
            thr = np.max(np.array([ts_filtered_cell_11, ts_filtered_cell_12]))
    else:
        if (x < Scale / 2) or (x >= FrameSize[1] - Scale / 2):
            j_floor = int(np.floor((y - w / 2) / w))
            ts_filtered_cell_11 = TimestampMap[j_floor , x_cell]
            ts_filtered_cell_21 = TimestampMap[j_floor + 1, x_cell]
            thr = np.max(np.array([ts_filtered_cell_11, ts_filtered_cell_21]))

        else:
            i_floor = int(np.floor((x - w / 2) / w))
            j_floor = int(np.floor((y - w / 2) / w))
            
            ts_filtered_cell_11 = TimestampMap[j_floor , i_floor ]
            ts_filtered_cell_12 = TimestampMap[j_floor , i_floor + 1]
            ts_filtered_cell_21 = TimestampMap[j_floor + 1, i_floor ]
            ts_filtered_cell_22 = TimestampMap[j_floor + 1, i_floor + 1]

            thr = np.max(np.array([ts_filtered_cell_11,
                                   ts_filtered_cell_12, 
                                   ts_filtered_cell_21,
                                   ts_filtered_cell_22]))
    return thr

@njit(float64(uint16, uint16, int64, int64[:], float64[:,:], float64[:,:]))
def _bilinear_interval_weights_interpolation_(y, x, Scale, FrameSize, TimestampMap, IntervalMap):
    y_cell = int(np.floor(y / Scale))
    x_cell = int(np.floor(x / Scale))
    w = Scale

    if (y < Scale / 2) or (y >= FrameSize[0] - Scale / 2):
        if (x < Scale / 2) or (x >= FrameSize[1] - Scale / 2):
            ts_filtered_cell = TimestampMap[y_cell, x_cell]
            thr = ts_filtered_cell
        else:
            i_floor = int(np.floor((x - w / 2) / w))
            dx1 = x - w / 2 - i_floor * w + 0.5
            dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5

            interval_11 = IntervalMap[y_cell, i_floor ]
            interval_12 = IntervalMap[y_cell, i_floor + 1]

            coef_11 = dx2 * interval_12
            coef_12 = dx1 * interval_11

            sum_top = coef_11 + coef_12

            ts_filtered_cell_11 = TimestampMap[y_cell, i_floor ]
            ts_filtered_cell_12 = TimestampMap[y_cell, i_floor + 1]

            thr = ts_filtered_cell_11 * coef_11 / sum_top + ts_filtered_cell_12 * coef_12 / sum_top
    else:
        if (x < Scale / 2) or (x >= FrameSize[1] - Scale / 2):
            j_floor = int(np.floor((y - w / 2) / w))
            dy1 = y - w / 2 - j_floor * w + 0.5
            dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5

            interval_11 = IntervalMap[j_floor , x_cell]
            interval_21 = IntervalMap[j_floor + 1, x_cell]

            coef_top = dy2 * interval_21
            coef_bot = dy1 * interval_11

            sum_all = coef_top + coef_bot

            ts_filtered_cell_11 = TimestampMap[j_floor , x_cell]
            ts_filtered_cell_21 = TimestampMap[j_floor + 1, x_cell]

            thr = ts_filtered_cell_11 * coef_top / sum_all + ts_filtered_cell_21 * coef_bot / sum_all

        else:
            i_floor = int(np.floor((x - w / 2) / w))
            j_floor = int(np.floor((y - w / 2) / w))
            dx1 = x - w / 2 - i_floor * w + 0.5
            dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5
            dy1 = y - w / 2 - j_floor * w + 0.5
            dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5

            interval_11 = IntervalMap[j_floor , i_floor]
            interval_12 = IntervalMap[j_floor , i_floor + 1]
            interval_21 = IntervalMap[j_floor + 1, i_floor]
            interval_22 = IntervalMap[j_floor + 1, i_floor + 1]

            interval_top = interval_11 * interval_12
            interval_bot = interval_21 * interval_22

            coef_11 = dx2 * interval_12
            coef_12 = dx1 * interval_11
            coef_21 = dx2 * interval_22
            coef_22 = dx1 * interval_21

            sum_top = coef_11 + coef_12
            sum_bot = coef_21 + coef_22

            coef_top = dy2 * interval_bot
            coef_bot = dy1 * interval_top
            sum_all = coef_top + coef_bot
            
            ts_filtered_cell_11 = IntervalMap[j_floor + 0, i_floor + 0]
            ts_filtered_cell_12 = IntervalMap[j_floor + 0, i_floor + 1]
            ts_filtered_cell_21 = IntervalMap[j_floor + 1, i_floor + 0]
            ts_filtered_cell_22 = IntervalMap[j_floor + 1, i_floor + 1]

            thr1 = ts_filtered_cell_11 * coef_11 / sum_top + ts_filtered_cell_12 * coef_12 / sum_top
            thr2 = ts_filtered_cell_21 * coef_21 / sum_bot + ts_filtered_cell_22 * coef_22 / sum_bot
            thr = thr1 * coef_top / sum_all + thr2 * coef_bot / sum_all
    return thr

@njit(float64(uint16, uint16, int64, int64[:], float64[:,:]))
def _bilinear_interpolation_( y, x, Scale, FrameSize, TimestampMap):
    y_cell = int(np.floor(y / Scale))
    x_cell = int(np.floor(x / Scale))
    w = Scale

    if (y < Scale / 2) or (y >= FrameSize[0] - Scale / 2):
        if (x < Scale / 2) or (x >= FrameSize[1] - Scale / 2):
            ts_filtered_cell = TimestampMap[y_cell, x_cell]
            thr = ts_filtered_cell
        else:
            i_floor = int(np.floor((x - w / 2) / w))
            dx1 = x - w / 2 - i_floor * w + 0.5
            dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5
    
            ts_filtered_cell_11 = TimestampMap[y_cell, i_floor + 0]
            ts_filtered_cell_12 = TimestampMap[y_cell, i_floor + 1]
    
            thr = ts_filtered_cell_11 * (dx2 / w) + ts_filtered_cell_12 * (dx1 / w)
    else:
        if (x < Scale / 2) or (x >= FrameSize[1] - Scale / 2):
            j_floor = int(np.floor((y - w / 2) / w))
            dy1 = y - w / 2 - j_floor * w + 0.5
            dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5

            ts_filtered_cell_11 = TimestampMap[j_floor + 0, x_cell]
            ts_filtered_cell_21 = TimestampMap[j_floor + 1, x_cell]

            thr = ts_filtered_cell_11 * (dy2 / w) + ts_filtered_cell_21 * (dy1 / w)

        else:
            i_floor = int(np.floor((x - w / 2) / w))
            j_floor = int(np.floor((y - w / 2) / w))
            dx1 = x - w / 2 - i_floor * w + 0.5
            dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5
            dy1 = y - w / 2 - j_floor * w + 0.5
            dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5

            ts_filtered_cell_11 = TimestampMap[j_floor + 0, i_floor + 0]
            ts_filtered_cell_12 = TimestampMap[j_floor + 0, i_floor + 1]
            ts_filtered_cell_21 = TimestampMap[j_floor + 1, i_floor + 0]
            ts_filtered_cell_22 = TimestampMap[j_floor + 1, i_floor + 1]

            thr1 = ts_filtered_cell_11 * (dx2 / w) + ts_filtered_cell_12 * (dx1 / w)
            thr2 = ts_filtered_cell_21 * (dx2 / w) + ts_filtered_cell_22 * (dx1 / w)
            thr = thr1 * (dy2 / w) + thr2 * (dy1 / w)
    return thr

@njit(float64(uint16, uint16, int64, int64[:], float64[:,:], float64[:,:]))
def _distance_interpolation_(y, x, Scale, FrameSize, TimestampMap, IntervalMap):
    y_cell = int(np.floor(y / Scale))
    x_cell = int(np.floor(x / Scale))

    w = Scale

    if (y < Scale / 2) or (y >= FrameSize[0] - Scale / 2):
        
        if (x < Scale / 2) or (x >= FrameSize[1] - Scale / 2):
            ts_filtered_cell = TimestampMap[y_cell, x_cell]
            thr = ts_filtered_cell
            
        else:
            i_floor = int(np.floor((x - w / 2) / w))
            dx1 = x - w / 2 - i_floor * w + 0.5
            dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5

            if (y < Scale / 2):
                dy =  w / 2 - y - 0.5
            else:
                j_floor = int(np.floor((y - w / 2) / w))
                dy = y - w / 2 - j_floor * w + 0.5

            dist_1 = np.sqrt(dx1**2 + dy**2)
            dist_2 = np.sqrt(dx2**2 + dy**2)

            interval_1 = IntervalMap[y_cell, i_floor + 0]
            interval_2 = IntervalMap[y_cell, i_floor + 1]

            C_1 = dist_2 * interval_2
            C_2 = dist_1 * interval_1

            coef_sum = C_1 + C_2

            ts_filtered_cell_11 = TimestampMap[y_cell, i_floor + 0]
            ts_filtered_cell_12 = TimestampMap[y_cell, i_floor + 1]

            thr = (ts_filtered_cell_11 * C_1 + ts_filtered_cell_12 * C_2) / coef_sum
    else:
        if (x < Scale / 2) or (x >= FrameSize[1] - Scale / 2):
            if (x < Scale / 2):
                dx = w / 2 - x - 0.5
            else:
                i_floor = int(np.floor((x - w / 2) / w))
                dx = x - w / 2 - i_floor * w + 0.5
            
            j_floor = int(np.floor((y - w / 2) / w))
            dy1 = y - w / 2 - j_floor * w + 0.5
            dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5

            dist_1 = np.sqrt(dx**2 + dy1**2)
            dist_2 = np.sqrt(dx**2 + dy2**2)

            interval_1 = IntervalMap[j_floor + 0, x_cell]
            interval_2 = IntervalMap[j_floor + 1, x_cell]

            C_1 = dist_2 * interval_2
            C_2 = dist_1 * interval_1

            coef_sum = C_1 + C_2

            ts_filtered_cell_11 = TimestampMap[j_floor + 0, x_cell]
            ts_filtered_cell_21 = TimestampMap[j_floor + 1, x_cell]

            thr = (ts_filtered_cell_11 * C_1 + ts_filtered_cell_21 * C_2) / coef_sum

        else:
            i_floor = int(np.floor((x - w / 2) / w))
            j_floor = int(np.floor((y - w / 2) / w))
            dx1 = x - w / 2 - i_floor * w + 0.5
            dx2 = (i_floor + 1) * w - (x - w / 2) - 0.5
            dy1 = y - w / 2 - j_floor * w + 0.5
            dy2 = (j_floor + 1) * w - (y - w / 2) - 0.5

            dist_11 = np.sqrt(dx1**2 + dy1**2)
            dist_12 = np.sqrt(dx2**2 + dy1**2)
            dist_21 = np.sqrt(dx1**2 + dy2**2)
            dist_22 = np.sqrt(dx2**2 + dy2**2)

            interval_11 = IntervalMap[j_floor + 0, i_floor + 0]
            interval_12 = IntervalMap[j_floor + 0, i_floor + 1]
            interval_21 = IntervalMap[j_floor + 1, i_floor + 0]
            interval_22 = IntervalMap[j_floor + 1, i_floor + 1]

            coef_11 = dist_11 * interval_11
            coef_12 = dist_12 * interval_12
            coef_21 = dist_21 * interval_21
            coef_22 = dist_22 * interval_22

            C_11 = coef_12 * coef_21 * coef_22
            C_12 = coef_11 * coef_21 * coef_22
            C_21 = coef_11 * coef_12 * coef_22
            C_22 = coef_11 * coef_12 * coef_21

            coef_sum = C_11 + C_12 + C_21 + C_22
            
            ts_11 = TimestampMap[j_floor + 0, i_floor + 0]
            ts_12 = TimestampMap[j_floor + 0, i_floor + 1]
            ts_21 = TimestampMap[j_floor + 1, i_floor + 0]
            ts_22 = TimestampMap[j_floor + 1, i_floor + 1]

            thr = (ts_11*C_11 + ts_12 * C_12 + ts_21 * C_21 + ts_22 * C_22) / coef_sum
    return thr

@njit(boolean(int64, uint16, uint16, int64, int64, int64, int64[:], float64[:,:], float64[:,:]))
def _filterEvent_(interpolation_method, x, y, t, FilterLength, Scale, FrameSize, TimestampMap, IntervalMap):
    if interpolation_method == 0:
        thr_ts = _bilinear_interpolation_(y, x, Scale, FrameSize, TimestampMap)
    elif interpolation_method == 1:
        thr_ts = _bilinear_interval_weights_interpolation_(y, x, Scale, FrameSize, TimestampMap, IntervalMap)
    elif interpolation_method == 2:
        thr_ts = _max_interpolation_(y, x, Scale, FrameSize, TimestampMap)
    elif interpolation_method == 3:
        thr_ts = _distance_interpolation_(y, x, Scale, FrameSize, TimestampMap, IntervalMap)
    else:
        return False
    diff_ts = t - thr_ts
    correct = diff_ts < FilterLength
    return correct

@njit(void(uint16, uint16, int64, int64, float64[:,:], float64[:,:], float64))
def _updateInterval_(x, y, t, Scale, TimestampMap, IntervalMap, UpdateFactor):
    cellY = int(np.floor(y / Scale))
    cellX = int(np.floor(x / Scale))
    cell_time = TimestampMap[cellY, cellX]
    cell_interval = IntervalMap[cellY, cellX]
    time_diff = t - cell_time
    new_interval = cell_interval * (1 - UpdateFactor) + time_diff * UpdateFactor;
    IntervalMap[cellY, cellX] = new_interval

@njit(void(uint16, uint16, int64, int64, float64[:,:], float64))
def _updateFilteredTimestamp_(x, y, t , Scale, TimestampMap, UpdateFactor):
    cellY = int(np.floor(y / Scale))
    cellX = int(np.floor(x / Scale))
    cell_filtered_time = TimestampMap[cellY, cellX]
    new_filtered_time = cell_filtered_time * (1 - UpdateFactor) + t * UpdateFactor
    TimestampMap[cellY, cellX] = new_filtered_time

@njit(void(uint16, uint16, int64, boolean[:,:]))
def _updateActive_(x, y, Scale, ActiveMap):
    cellY = int(np.floor(y / Scale))
    cellX = int(np.floor(x / Scale))
    ActiveMap[cellY, cellX] = True

@njit(void(uint16, uint16, int64, int64, float64[:,:], float64[:,:], boolean[:,:], float64))
def _updateFeatures_(
        x,
        y,
        t,
        Scale, TimestampMap, IntervalMap, ActiveMap, UpdateFactor):
    _updateInterval_(x, y, t, Scale, TimestampMap, IntervalMap, UpdateFactor)
    _updateFilteredTimestamp_(x, y, t, Scale, TimestampMap, UpdateFactor)
    _updateActive_(x, y, Scale, ActiveMap)

@njit(void(uint16[:], uint16[:], int64[:], boolean[:], int64, int64, int64, int64[:], float64[:,:], float64[:,:], boolean[:,:], float64))
def _processEvents_(
        ev_x: np.uint16,
        ev_y: np.uint16,
        ev_t: np.int64,
        eventsBin: np.bool,
        interpolation_method: int,
        FilterLength: int, 
        Scale: int,
        FrameSize: np.int64,
        TimestampMap: np.int64,
        IntervalMap: np.float64, 
        ActiveMap: np.bool,
        UpdateFactor: float):
    """
    event = np.array([[x, y, p, t]] ???????
    """
    numel = ev_x.shape[0]
    for i in range(numel):
        x = ev_x[i]
        y = ev_y[i]
        t = ev_t[i]
        eventsBin[i] = _filterEvent_(interpolation_method, x, y, t, FilterLength, Scale, FrameSize, TimestampMap, IntervalMap)
        _updateFeatures_(x, y, t, Scale, TimestampMap, IntervalMap, ActiveMap, UpdateFactor)

class event_filter_interpolation_compiled:
    def __init__(self, frame_size, filter_length, scale, update_factor, 
                 interpolation_method, filtered_ts = None):
        """
        Parameters:
        -----------
            frame_size: np.array of np.int64 - image size
            filter_length: np.int64 - threshold that determines which events to
                           filter out
            scale: np.int64 - size of areas to sub-divide the frames
            update_rate: np.int64 - size of areas to sub-divide the frames
            interpolation_method: np.int64 - 
                                    - 0 : bilinear
                                    - 1 : bilinear with interval weights
                                    - 2 : max
                                    - 3 : distance
            filtered_ts: ???? artifact from previous version that does nothing...
        """
        msg =  "scale must divide evenly into frame size"
        assert np.sum((frame_size / scale) % 1) == 0, msg
        self.FrameSize = frame_size
        self.FilterLength = filter_length
        self.Scale = scale
        self.UpdateFactor = update_factor
        
        sz_0 = int(np.floor(frame_size[0] / scale))
        sz_1 = int(np.floor(frame_size[1] / scale))
        
        self.TimestampMap = np.zeros([sz_0, sz_1], dtype = np.float64)
        self.IntervalMap = 1e4 * np.ones([sz_0, sz_1], dtype = np.float64)
        self.ActiveMap = np.zeros([sz_0, sz_1], dtype = bool)
        self.ValidEvents = 0
        self.InvalidEvents = 0
        self.CurrentTs = 0
        self.FilteredTs = filtered_ts
        self.InterpolationMethod = interpolation_method

    def processEvents(self, events):
        """
        event = np.array([[x, y, p, t]] ???????
        """
        eventsBin = np.zeros(events.shape[0], dtype = bool)
        _processEvents_(
                        events['x'],
                        events['y'],
                        events['t'],
                        #events,
                        eventsBin,
                        self.InterpolationMethod, 
                        self.FilterLength,
                        self.Scale,
                        np.array(self.FrameSize),
                        self.TimestampMap,
                        self.IntervalMap,
                        self.ActiveMap,
                        self.UpdateFactor,
                        )
        self.eventsBin = eventsBin
