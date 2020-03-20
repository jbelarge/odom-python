import torch
import numpy as np

class OxfordDataset(torch.utils.data.Dataset):

    # ===================================================
    # Note about window samples and overlap samples:
    # Those values are important for when we are
    # getting data from the data set. They define the
    # following:
    #  WS: The number of samples you want.
    #  OS: This is how much you are okay with overlapping.
    #
    # Example:
    #   Data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #   Index = 3, WS=2, OS=0
    #    return ([4,5])
    #   Index = 3, WS=2, OS=1
    #    return ([3,4])
    # You get it.
    # ===================================================

    def __init__(self,
                 truthFile='../data/trolley/data1/syn/vi1.csv',
                 dataFile='../data/trolley/data1/syn/imu1.csv',
                 window_samples=200,
                 overlap_samples=0):
        self.sensor = np.loadtxt(dataFile, delimiter=',').astype(np.float32)
        self.truth  = np.loadtxt(truthFile, delimiter=',').astype(np.float32)
        self.len    = self.sensor.shape[0] // (window_samples - overlap_samples)
        self.window_samples  = window_samples
        self.overlap_samples = overlap_samples

    def __getitem__(self, index):
        ws  = self.window_samples
        pos = index * (ws - self.overlap_samples)

        # Sensor Data:
        time                      = self.sensor[pos:(pos+ws), 0]
        attitude_rad              = self.sensor[pos:(pos+ws), 1:4]
        rotation_rate_rad_per_sec = self.sensor[pos:(pos+ws), 4:7]
        gravity                   = self.sensor[pos:(pos+ws), 7:10]
        acc                       = self.sensor[pos:(pos+ws), 10:13]
        mag_field_microteslas     = self.sensor[pos:(pos+ws), 13:16]

        # Truth Data:
        time_truth  = self.truth[pos:(pos+ws), 0]
        translation = self.truth[pos:(pos+ws), 2:5]
        rotation    = self.truth[pos:(pos+ws), 5:10]

        return time, attitude_rad, rotation_rate_rad_per_sec, gravity, acc, mag_field_microteslas, time_truth, translation, rotation

    def __len__(self):
        return self.len

