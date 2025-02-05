import numpy as np
from scipy.signal import butter, filtfilt


class Operator:

    @staticmethod
    def moving_average(x: np.array, window_size: int):
        """
        Compute the moving average of a signal
        .
        Parameters
        ----------
        x: np.array
            The signal to be averaged (for now, this can only be a vector)
        window_size: int
            The size of the window to compute the average on
        .
        Returns
        -------
        x_averaged: np.array
            The signal processed using the moving average
        """
        # Checks
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd number")
        if len(x.shape) != 1:
            if len(x.shape) == 2 and x.shape[1] == 1:
                x = x.flatten()
            else:
                raise ValueError("x must be a vector")
        if x.shape[0] / 2 < window_size:
            raise ValueError("window_size must be smaller than half of the length of the signal")

        # Compute the moving average
        x_averaged = np.zeros_like(x)
        for i in range(len(x)):
            if i < window_size // 2:
                x_averaged[i] = np.mean(x[: i + window_size // 2 + 1])
            elif i >= len(x) - window_size // 2:
                x_averaged[i] = np.mean(x[i - window_size // 2 :])
            else:
                x_averaged[i] = np.mean(x[i - window_size // 2 : i + window_size // 2 + 1])
        return x_averaged


    @staticmethod
    def apply_filtfilt(data: np.ndarray, order: int, sampling_rate: int, cutoff_freq: int):
        """
        TODO: @ophlariviere -> This was taken from biomechanics tools, could you provide a ref for it ?
        .
        Apply a low-pass Butterworth filter to the data using scipy.filtfilt
        .
        Parameters
        ----------
        data: np.ndarray
            The data to be filtered (for now, this can only be a nb_data x nb_frames array)
        order: int
            The order of the Butterworth filter
        sampling_rate: int
            The sampling rate of the data in Hz
        cutoff_freq: int
            The cutoff frequency of the filter in Hz
        .
        Returns
        -------
        filtered_data: np.ndarray
            The filtered data
        """
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = np.zeros_like(data)
        filtered_data[:, :] = np.nan
        for i_data in range(data.shape[0]):
            non_nan_idx = ~np.isnan(data[i_data, :])
            filtered_data[i_data, non_nan_idx] = filtfilt(b, a,  data[i_data, non_nan_idx], axis=0)
        return filtered_data

