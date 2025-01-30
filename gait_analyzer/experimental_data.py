import ezc3d
import biorbd
import numpy as np


class ExperimentalData:
    """
    This class contains all the experimental data from a trial (markers, EMG, force plates data, gait parameters).
    """

    def __init__(self, c3d_file_name: str, biorbd_model: biorbd.Model, animate_c3d_flag: bool):
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
        self.f_ext_sorted = None
        self.markers_time_vector = None
        self.analogs_time_vector = None

        # Extract data from the c3d file
        self.perform_initial_treatment()
        self.extract_gait_parameters()
        if animate_c3d_flag:
            self.animate_c3d()

    def perform_initial_treatment(self):
        """
        Extract important information and sort the data
        """
        def load_model():
            self.model_marker_names = [m.to_string() for m in self.biorbd_model.markerNames()]
            # model_muscle_names = [m.to_string() for m in self.biorbd_model.muscleNames()]

        def sort_markers():
            self.c3d = ezc3d.c3d(self.c3d_full_file_path)
            markers = self.c3d["data"]["points"]
            self.marker_sampling_frequency = self.c3d["parameters"]["POINT"]["RATE"]["value"][0]  # Hz
            self.markers_dt = 1 / self.c3d["header"]["points"]["frame_rate"]
            self.nb_marker_frames = markers.shape[2]
            exp_marker_names = self.c3d["parameters"]["POINT"]["LABELS"]["value"]
            marker_units = 1
            if self.c3d["parameters"]["POINT"]["UNITS"]["value"][0] == "mm":
                marker_units = 0.001
            if len(self.model_marker_names) > len(exp_marker_names):
                supplementary_marker_names = [name for name in self.model_marker_names if name not in exp_marker_names]
                raise ValueError(f"The markers {supplementary_marker_names} are not in the c3d file, but are in the model.")
            markers_sorted = np.zeros((3, len(self.model_marker_names), self.nb_marker_frames))
            for i_marker, name in enumerate(self.model_marker_names):
                marker_idx = exp_marker_names.index(name)
                markers_sorted[:, marker_idx, :] = markers[:3, marker_idx, :] * marker_units
            self.markers_sorted = markers_sorted

        def sort_analogs():
            # Get an array of the experimental muscle activity
            analogs = self.c3d["data"]["analogs"]
            self.nb_analog_frames = analogs.shape[2]
            self.analogs_sampling_frequency = self.c3d["parameters"]["ANALOG"]["RATE"]["value"][0]  # Hz
            self.analogs_dt = 1 / self.c3d["header"]["analogs"]["frame_rate"]
            analog_names = self.c3d["parameters"]["ANALOG"]["LABELS"]["value"]
            analog_units = np.ones((len(analog_names, )))
            if self.c3d["parameters"]["ANALOG"]["UNITS"]["value"][0] == "mm":
                analog_units = 0.001
            # print(analog_names)
            # emg_sorted = np.zeros((len(model_muscle_names), self.nb_analog_frames))
            # for i_muscle, name in enumerate(model_muscle_names):
            #     muscle_idx = analog_names.index(name)
            #     emg_sorted[i_muscle, :] = analogs[muscle_idx, :]
            # self.emg_sorted = emg_sorted
            # # TODO: Charbie -> treatment of the EMG signal to remove stimulation artifacts
            return analog_names, analogs

        def extract_force_platform_data(analog_names, analogs):
            # TODO: Find how to get "platform" field !!!!! (@Ophlariviere)
            # TODO: Charbie/Flo -> make sure this is generalizable to Vicon
            # Get the experimental ground reaction forces
            # 0:3 -> forces, 3:6 -> moments
            force_platform_1_channels = self.c3d["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"][:, 0]
            force_platform_2_channels = self.c3d["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"][:, 1]
            force_platform_1_zeros = self.c3d["parameters"]["FORCE_PLATFORM"]['ZERO']['value'][0]
            force_platform_2_zeros = self.c3d["parameters"]["FORCE_PLATFORM"]['ZERO']['value'][1]
            force_platform_1_calibration_matrix = self.c3d["parameters"]["FORCE_PLATFORM"]['CAL_MATRIX']['value'][:, :, 0]
            force_platform_2_calibration_matrix = self.c3d["parameters"]["FORCE_PLATFORM"]['CAL_MATRIX']['value'][:, :, 1]
            f_ext = np.zeros((2, 6, self.nb_analog_frames))
            for i in range(6):
                platform_1_idx = analog_names.index(f"Channel_{force_platform_1_channels[i]:02d}")
                platform_2_idx = analog_names.index(f"Channel_{force_platform_2_channels[i]:02d}")
                f_ext[0, i, :] = analogs[0, platform_1_idx, :]
                f_ext[1, i, :] = analogs[0, platform_2_idx, :]
            f_ext_sorted = np.zeros((2, 9, self.nb_analog_frames))
            f_ext_sorted[0, 3:, :] = np.vstack((f_ext[0, 3:6, :], f_ext[0, :3, :]))
            f_ext_sorted[1, 3:, :] = np.vstack((f_ext[1, 3:6, :], f_ext[1, :3, :]))

            # Apply calibration matrix and zero
            # TODO: Charbie -> to be removed ! -------------------------------- # not the right order anymore !
            # f_ext_sorted[0, :, :] = force_platform_1_calibration_matrix @ (f_ext_sorted[0, :, :] - force_platform_1_zeros)
            # f_ext_sorted[1, :, :] = force_platform_2_calibration_matrix @ (f_ext_sorted[1, :, :] - force_platform_2_zeros)
            # Modify moment units from mm to m
            # f_ext_sorted[:, 3:6 :] /= 1000
            # Get the force plate origin
            force_platform_1_origin = np.mean(self.c3d["parameters"]["FORCE_PLATFORM"]['CORNERS']['value'][:, :, 0], axis=1)
            force_platform_2_origin = np.mean(self.c3d["parameters"]["FORCE_PLATFORM"]['CORNERS']['value'][:, :, 1], axis=1)
            """"
            center_of_pressure_1 = self.c3d["data"]["platform"]['center_of_pressure'] / 1000  # .....
            center_of_pressure_2 = self.c3d["data"]["platform"]['center_of_pressure'] / 1000  # .....
            distance_cop_to_origin_1 = force_platform_1_origin - center_of_pressure_1
            distance_cop_to_origin_2 = force_platform_2_origin - center_of_pressure_2
            transportation_moment_1 = np.cross(distance_cop_to_origin_1, f_ext_sorted[0, 3:6, :])
            transportation_moment_2 = np.cross(distance_cop_to_origin_2, f_ext_sorted[1, 3:6, :])
            f_ext_sorted[0, 6:9, :] += transportation_moment_1
            f_ext_sorted[1, 6:9, :] += transportation_moment_2
            f_ext_sorted[0, :3, :] = center_of_pressure_1
            f_ext_sorted[1, :3, :] = center_of_pressure_2
            """
            self.f_ext_sorted = f_ext_sorted

            # f_extfilt[contact, :, :] = self.forcedatafilter(f_ext, 4, 2000, 20)
            # moment_extfilt[contact, :, :] = self.forcedatafilter(moment_ext, 4, 2000, 20)
            # cop_ext = self.c3d['data']['platform'][contact]['center_of_pressure'] / 1000
            # cop_extfilt[contact, :, :] = self.forcedatafilter(cop_ext, 4, 2000, 10)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(np.linalg.norm(f_ext_sorted[0, 0:3, :], axis=0))
            # plt.plot(np.linalg.norm(f_ext_sorted[0, 3:6, :], axis=0))
            # plt.plot(np.linalg.norm(f_ext_sorted[1, 0:3, :], axis=0))
            # plt.plot(np.linalg.norm(f_ext_sorted[1, 3:6, :], axis=0))
            # plt.plot(np.linalg.norm(f_ext_sorted[0, 0:3, :], axis=0) + np.linalg.norm(f_ext_sorted[1, 0:3, :], axis=0))
            # plt.plot(np.linalg.norm(f_ext_sorted[0, 3:6, :], axis=0) + np.linalg.norm(f_ext_sorted[1, 3:6, :], axis=0))
            # plt.savefig('test.png')
            # plt.show()

            # from scipy import signal
            # b, a = signal.butter(2, 1/50, btype='low')
            # y = signal.filtfilt(b, a, f_ext_sorted[0, 2, :], padlen=150)
            # # 4th 6-10


        def compute_time_vectors():
            self.markers_time_vector = np.linspace(0, self.markers_dt * self.nb_marker_frames, self.nb_marker_frames)
            self.analogs_time_vector = np.linspace(0, self.analogs_dt * self.nb_analog_frames, self.nb_analog_frames)

        # Perform the initial treatment
        load_model()
        sort_markers()
        analog_names, analogs = sort_analogs()
        extract_force_platform_data(analog_names, analogs)
        compute_time_vectors()


    def get_f_ext_at_frame(self, i_node: int):
        """
        Constructs a biorbd external forces set object at a specific frame.
        .
        Parameters
        ----------
        i_node: int
            The frame index.
        .
        Returns
        -------
        f_ext_set: biorbd externalForceSet
            The external forces set at the frame.
        """
        f_ext_set = self.biorbd_model.externalForceSet()
        f_ext_set.add("calcn_l", self.f_ext_sorted[0, 3:, i_node], self.f_ext_sorted[0, :3, i_node])
        f_ext_set.add("calcn_r", self.f_ext_sorted[1, 3:, i_node], self.f_ext_sorted[1, :3, i_node])
        return f_ext_set


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
            "c3d_full_file_path": self.c3d_full_file_path,
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
            "f_ext_sorted": self.f_ext_sorted,
            "markers_time_vector": self.markers_time_vector,
            "analogs_time_vector": self.analogs_time_vector,
        }
