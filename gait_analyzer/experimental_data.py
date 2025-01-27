import ezc3d
import biorbd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


class ExperimentalData:
    def __init__(self, c3d_file_path: str, biorbd_model):

        # Checks
        if not isinstance(c3d_file_path, str):
            raise ValueError("c3d_file_path must be a string")

        # Initial attributes
        self.c3d_file_path = c3d_file_path

        # Extract data from the c3d file
        self.perform_initial_treatment()
        self.extract_gait_parameters(biorbd_model)


    def perform_initial_treatment(self):
        """
        Extract important information and sort the data
        """
        # Load model
        model = biorbd.Model(self.biorbd_model_path)
        self.model_marker_names = [m.to_string() for m in model.markerNames()]
        # model_muscle_names = [m.to_string() for m in model.muscleNames()]

        # Get an array of the position of the experimental self.markers
        c3d = ezc3d.c3d(self.file_path)
        markers = c3d["data"]["points"]
        self.marker_sampling_frequency = c3d["parameters"]["POINT"]["RATE"]["value"][0]  # Hz
        self.markers_dt = 1 / c3d["header"]["points"]["frame_rate"]
        self.nb_marker_frames = markers.shape[2]
        exp_marker_names = c3d["parameters"]["POINT"]["LABELS"]["value"]
        markers_sorted = np.zeros((3, len(self.model_marker_names), self.nb_marker_frames))
        for i_marker, name in enumerate(self.model_marker_names):
            marker_idx = exp_marker_names.index(name)
            markers_sorted[:, marker_idx, :] = markers[:3, marker_idx, :]
        self.markers_sorted = markers_sorted
        self.right_leg_grf = np.vstack((markers[:3, exp_marker_names.index("moment2"), :],
                                   markers[:3, exp_marker_names.index("force2"), :]))
        self.left_leg_grf = np.vstack((markers[:3, exp_marker_names.index("moment1"), :],
                                  markers[:3, exp_marker_names.index("force1"), :]))

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
        # plt.figure()
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 2, :], '-r')
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 1, :], '-g')
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 2, :], '.b')
        # plt.plot(self.analogs_time_vector, y, '-b')
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 3, :], '-m')
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 4, :], '-c')
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 5, :], '-k')
        # cal_velovity = np.diff(markers_sorted[2, self.model_marker_names.index("LCAL"), :]) / self.markers_dt
        # time_velocity = (self.marker_time_vector[1:] + self.marker_time_vector[:-1]) / 2
        # plt.plot(self.marker_time_vector, markers_sorted[2, self.model_marker_names.index("LCAL"), :], '-b')
        # plt.plot(time_velocity, cal_velovity, '-k')
        # plt.xlim(0, 1.4)
        # plt.show()

    def extract_gait_parameters(self, biorbd_model):
        """
        TODO: Guys -> please provide code :)
        """
        pass