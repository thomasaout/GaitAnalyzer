import ezc3d
import biorbd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


class ExperimentalData:
    """
    This class contains all the experimental data from a trial (markers, EMG, force plates data, gait parameters).
    """
    def __init__(self, c3d_file_name: str, biorbd_model: biorbd.Model, animate_c3d: bool):
        """
        Initialize the ExperimentalData.
        .
        Parameters
        ----------
        c3d_file_name: str
            The name of the trial's c3d file.
        biorbd_model: biorbd.Model
            The subject's personalized biorbd model.
        animate_c3d: bool
            If True, the c3d file will be animated.
        """
        # Checks
        if not isinstance(c3d_file_name, str):
            raise ValueError("c3d_file_name must be a string")

        # Initial attributes
        self.c3d_file_name = c3d_file_name
        self.c3d_full_file_path = "../data/" + c3d_file_name
        self.biorbd_model = biorbd_model

        # Extended attributes
        self.model_marker_names = None
        self.marker_sampling_frequency = None
        self.markers_dt = None
        self.nb_marker_frames = None
        self.markers_sorted = None
        self.analogs_sampling_frequency = None
        self.analogs_dt = None
        self.nb_analog_frames = None
        self.grf_sorted = None
        self.marker_time_vector = None
        self.analogs_time_vector = None

        # Extract data from the c3d file
        self.perform_initial_treatment()
        self.extract_gait_parameters()
        if animate_c3d:
            self.animate_c3d()


    def perform_initial_treatment(self):
        """
        Extract important information and sort the data
        """
        # Load model
        self.model_marker_names = [m.to_string() for m in self.biorbd_model.markerNames()]
        # model_muscle_names = [m.to_string() for m in self.biorbd_model.muscleNames()]

        # Get an array of the position of the experimental markers
        c3d = ezc3d.c3d(self.c3d_full_file_path)
        markers = c3d["data"]["points"]
        self.marker_sampling_frequency = c3d["parameters"]["POINT"]["RATE"]["value"][0]  # Hz
        self.markers_dt = 1 / c3d["header"]["points"]["frame_rate"]
        self.nb_marker_frames = markers.shape[2]
        exp_marker_names = c3d["parameters"]["POINT"]["LABELS"]["value"]
        markers_units = 1
        if c3d["parameters"]["POINT"]["UNITS"]["value"][0] == "mm":
            marker_units = 0.001
        # if len(self.model_marker_names) > len(exp_marker_names):
        #     supplementary_marker_names = [name for name in self.model_marker_names if name not in exp_marker_names]
        #     for name in supplementary_marker_names:
        #         if not name.endswith("JC"):
        #             raise ValueError(f"The marker {name} is not in the c3d file.")
        #             # TODO: Flo -> The JC markers were added before the OpenSim scaling, what should we do with those ?
        markers_sorted = np.zeros((3, len(self.model_marker_names), self.nb_marker_frames))
        for i_marker, name in enumerate(self.model_marker_names):
            marker_idx = exp_marker_names.index(name)
            markers_sorted[:, marker_idx, :] = markers[:3, marker_idx, :] * marker_units
        self.markers_sorted = markers_sorted

        # Get an array of the experimental muscle activity
        analogs = c3d["data"]["analogs"]
        self.nb_analog_frames = analogs.shape[2]
        self.analogs_sampling_frequency = c3d["parameters"]["ANALOG"]["RATE"]["value"][0]  # Hz
        self.analogs_dt = 1 / c3d["header"]["analogs"]["frame_rate"]
        analog_names = c3d["parameters"]["ANALOG"]["LABELS"]["value"]
        # print(analog_names)
        # emg_sorted = np.zeros((len(model_muscle_names), self.nb_analog_frames))
        # for i_muscle, name in enumerate(model_muscle_names):
        #     muscle_idx = analog_names.index(name)
        #     emg_sorted[i_muscle, :] = analogs[muscle_idx, :]
        # self.emg_sorted = emg_sorted
        # # TODO: Charbie -> treatment of the EMG signal to remove stimulation artifacts

        # Get the experimental ground reaction forces
        force_platform_1_channels = c3d["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"][:, 0]
        force_platform_2_channels = c3d["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"][:, 1]
        grf_sorted = np.zeros((2, 6, self.nb_analog_frames))
        for i in range(6):
            platform_1_idx = analog_names.index(f"Channel_{force_platform_1_channels[i]:02d}")
            platform_2_idx = analog_names.index(f"Channel_{force_platform_2_channels[i]:02d}")
            grf_sorted[0, i, :] = analogs[0, platform_1_idx, :]
            grf_sorted[1, i, :] = analogs[0, platform_2_idx, :]
        self.grf_sorted = grf_sorted

        # from scipy import signal
        # b, a = signal.butter(2, 1/50, btype='low')
        # y = signal.filtfilt(b, a, grf_sorted[0, 2, :], padlen=150)
        # # 4th 6-10

        self.marker_time_vector = np.linspace(0, self.markers_dt * self.nb_marker_frames, self.nb_marker_frames)
        self.analogs_time_vector = np.linspace(0, self.analogs_dt * self.nb_analog_frames, self.nb_analog_frames)


    def animate_c3d(self):
        # TODO: Charbie -> animate the c3d file with pyorerun
        pass


    def extract_gait_parameters(self):
        """
        TODO: Guys -> please provide code :)
        """
        pass


    def inputs(self):
        return {
            "c3d_file_path": self.c3d_file_path,
            "biorbd_model": self.biorbd_model,
        }

    def outputs(self):
        return {
            "model_marker_names": self.model_marker_names,
            "marker_sampling_frequency": self.marker_sampling_frequency,
            "markers_dt": self.markers_dt,
            "nb_marker_frames": self.nb_marker_frames,
            "markers_sorted": self.markers_sorted,
            "analogs_sampling_frequency": self.analogs_sampling_frequency,
            "analogs_dt": self.analogs_dt,
            "nb_analog_frames": self.nb_analog_frames,
            "grf_sorted": self.grf_sorted,
            "marker_time_vector": self.marker_time_vector,
            "analogs_time_vector": self.analogs_time_vector,
        }