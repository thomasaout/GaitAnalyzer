import numpy as np


class Operator:

    def help(self):
        print("This class contains some useful functions for signal analysis")
        print("The following functions are available:")
        for method in dir(self):
            if callable(getattr(self, method)) and not method.startswith("__"):
                print(f"- {method}")


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
        if x.shape[0]/2 < window_size:
            raise ValueError("window_size must be smaller than half of the length of the signal")

        # Compute the moving average
        x_averaged = np.zeros_like(x)
        for i in range(len(x)):
            if i < window_size // 2:
                x_averaged[i] = np.mean(x[:i + window_size // 2 + 1])
            elif i >= len(x) - window_size // 2:
                x_averaged[i] = np.mean(x[i - window_size // 2:])
            else:
                x_averaged[i] = np.mean(x[i - window_size // 2:i + window_size //
                                            2 + 1])
        return x_averaged
