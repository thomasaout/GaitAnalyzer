import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import biorbd
from pyorerun import BiorbdModel, PhaseRerun
from pyomeca import Markers

from gait_analyzer.operator import Operator
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.biomod_model_creator import BiomodModelCreator
from gait_analyzer.events import Events


# TODO: Charbie -> See if we keep the virtual markers and if we impose this way of reconstructing

segment_dict = {"pelvis": {"dof_idx": [0, 1, 2, 3, 4, 5],
                           "markers_idx": [0, 1, 2, 3, 49],
                           "min_bound": [-3, -3, -3, -np.pi / 4, -np.pi / 4, -np.pi],
                           "max_bound": [3, 3, 3, np.pi / 4, np.pi / 4, np.pi]},
                "femur_r": {"dof_idx": [6, 7, 8],
                            "markers_idx": [4, 5, 6, 50, 51],
                            "min_bound": [-0.6981317007977318, -1.0471975511965976, -0.5235987755982988],
                            "max_bound": [2.0943951023931953, 0.5235987755982988, 0.5235987755982988]},
                "tibia_r": {"dof_idx": [9],
                            "markers_idx": [7, 8, 9, 52, 53],
                            "min_bound": [-2.6179938779914944],
                            "max_bound": [0.0]},
                "calcn_r": {"dof_idx": [10, 11],
                            "markers_idx": [10, 11, 12, 54, 55],
                            "min_bound": [-0.8726646259971648, -0.2617993877991494],
                            "max_bound": [0.5235987755982988, 0.2617993877991494]},
                "toes_r": {"dof_idx": [12],
                           "markers_idx": [13],
                           "min_bound": [-0.8726646259971648],
                           "max_bound": [1.0471975511965976]},
                "femur_l": {"dof_idx": [13, 14, 15],
                            "markers_idx": [14, 15, 16, 56, 57],
                            "min_bound": [-0.6981317007977318, -1.0471975511965976, -0.5235987755982988],
                            "max_bound": [2.0943951023931953, 0.5235987755982988, 0.5235987755982988]},
                "tibia_l": {"dof_idx": [16],
                            "markers_idx": [17, 18, 19, 58, 59],
                            "min_bound": [-2.6179938779914944],
                            "max_bound": [0.0]},
                "calcn_l": {"dof_idx": [17, 18],
                            "markers_idx": [20, 21, 22, 60, 61],
                            "min_bound": [-0.8726646259971648, -0.2617993877991494],
                            "max_bound": [0.5235987755982988, 0.2617993877991494]},
                "toes_l": {"dof_idx": [19],
                           "markers_idx": [23],
                           "min_bound": [-0.8726646259971648],
                           "max_bound": [1.0471975511965976]},
                "torso": {"dof_idx": [20, 21, 22],
                          "markers_idx": [24, 25, 26, 27, 28, 62],
                          "min_bound": [-1.5707963267948966, -0.6108652381980153, -0.7853981633974483],
                          "max_bound": [0.7853981633974483, 0.6108652381980153, 0.7853981633974483]},
                "head": {"dof_idx": [23, 24, 25],
                         "markers_idx": [29, 30, 31, 32, 33],
                         "min_bound": [-0.8726646259971648, -0.59999999999999998, -1.2217],
                         "max_bound": [0.7853981633974483, 0.59999999999999998, 1.2217]},
                "humerus_r": {"dof_idx": [26, 27, 28],
                              "markers_idx": [34, 35, 63],
                              "min_bound": [-1.5707963300000001, -3.8397000000000001, -1.5707963300000001],
                              "max_bound": [3.1415999999999999, 1.5707963300000001, 1.5707963300000001]},
                "radius_r": {"dof_idx": [29, 30],
                             "markers_idx": [36, 37, 64],
                             "min_bound": [0.0, -3.1415999999999999],
                             "max_bound": [3.1415999999999999, 3.1415999999999999]},
                "hand_r": {"dof_idx": [31, 32],
                           "markers_idx": [38, 39, 65],
                           "min_bound": [-1.5708, -0.43633231],
                           "max_bound": [1.5708, 0.61086523999999998]},
                "fingers_r": {"dof_idx": [33],
                              "markers_idx": [40],
                              "min_bound": [-1.5708],
                              "max_bound": [1.5708]},
                "humerus_l": {"dof_idx": [34, 35, 36],
                              "markers_idx": [41, 42, 66],
                              "min_bound": [-1.5707963300000001, -3.8397000000000001, -1.5707963300000001],
                              "max_bound": [3.1415999999999999, 1.5707963300000001, 1.5707963300000001]},
                "radius_l": {"dof_idx": [37, 38],
                             "markers_idx": [43, 44, 67],
                             "min_bound": [0.0, -3.1415999999999999],
                             "max_bound": [3.1415999999999999, 3.1415999999999999]},
                "hand_l": {"dof_idx": [39, 40],
                           "markers_idx": [45, 46, 68],
                           "min_bound": [-1.5708, -0.43633231],
                           "max_bound": [1.5708, 0.61086523999999998]},
                "fingers_l": {"dof_idx": [41],
                              "markers_idx": [47, 48],
                              "min_bound": [-1.5708],
                              "max_bound": [1.5708]},
                }

class KinematicsReconstructor:
    """
    This class reconstruct the kinematics based on the marker position and the model predefined.
    """

    def __init__(self, 
                 experimental_data: ExperimentalData, 
                 biorbd_model_creator: BiomodModelCreator, 
                 events: Events,
                 skip_if_existing: bool,
                 animate_kinematics_flag: bool,
                 plot_kinematics_flag: bool):
        """
        Initialize the KinematicsReconstructor.
        .
        Parameters
        ----------
        experimental_data: ExperimentalData
            The experimental data from the trial
        biorbd_model_creator: BiomodModelCreator
            The biorbd model to use for the kinematics reconstruction
        events: Events
            The events to use for the kinematics reconstruction since we exploit the fact that the movement is cyclic.
        skip_if_existing: bool
            If True, the kinematics will not be reconstructed if the output file already exists
        animate_kinematics_flag: bool
            If True, the kinematics will be animated through pyorerun
        plot_kinematics_flag: bool
            If True, the kinematics will be plotted and saved in a .png
        """

        # Checks
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError(
                "experimental_data must be an instance of ExperimentalData. You can declare it by running ExperimentalData(file_path)."
            )
        if not isinstance(biorbd_model_creator, BiomodModelCreator):
            raise ValueError(
                "biorbd_model_creator must be an instance of BiomodModelCreator."
            )
        if not isinstance(events, Events):
            raise ValueError(
                "events must be an instance of Events."
            )

        # Initial attributes
        self.experimental_data = experimental_data
        self.biorbd_model_creator = biorbd_model_creator
        self.events = events

        # Extended attributes
        self.biorbd_model = None
        self.t = None
        self.q = None
        self.q_filtered = None
        self.qdot = None
        self.qddot = None
        self.is_loaded_kinematics = False

        if skip_if_existing and self.check_if_existing():
            self.is_loaded_kinematics = True
        else:
            # Perform the kinematics reconstruction
            self.perform_kinematics_reconstruction()
            self.filter_kinematics()
            self.save_kinematics_reconstruction()

        if animate_kinematics_flag:
            self.animate_kinematics()

        if plot_kinematics_flag:
            self.plot_kinematics()


    def check_if_existing(self):
        """
        Check if the kinematics reconstruction already exists.
        If it exists, load the q.
        .
        Returns
        -------
        bool
            If the kinematics reconstruction already exists
        """
        result_file_full_path = self.get_result_file_full_path()
        if os.path.exists(result_file_full_path):
            with open(result_file_full_path, "rb") as file:
                data = pickle.load(file)
                self.t = data["t"]
                self.q = data["q"]
                self.q_filtered = data["q_filtered"]
                self.qdot = data["qdot"]
                self.qddot = data["qddot"]
                self.biorbd_model = self.biorbd_model_creator.biorbd_model
            return True
        else:
            return False


    def perform_kinematics_reconstruction(self):

        print("Performing inverse kinematics reconstruction using 'only_lm'")

        model = biorbd.Model(self.biorbd_model_creator.biorbd_model_virtual_markers_full_path)
        self.biorbd_model = model
        markers = self.experimental_data.markers_sorted_with_virtual
        nb_frames = markers.shape[2]
        self.max_frames = nb_frames  # TODO: add the possibility to chose the frame range to reconstruct

        q_recons = np.ndarray((nb_frames, model.nbQ()))
        ik = biorbd.InverseKinematics(model, markers[:, :, :self.max_frames])
        q_recons[:self.max_frames, :] = ik.solve(method="only_lm").T

        self.q = q_recons
        self.t = self.experimental_data.markers_time_vector[:self.max_frames]


    def filter_kinematics(self):
        """
        Unwrap and filter the joint angles.
        """
        def unwrap_kinematics(biorbd_model: biorbd.Model, q: np.ndarray):
            """
            Performs unwrap on the kinematics from which it re-expressed in terms of matrix rotation before
            (which makes it more likely to be in the same quadrant)

            Returns
            -------

            """
            dof_names = [n.to_string() for n in biorbd_model.nameDof()]
            for i_segment, segment_name in enumerate(segment_dict):
                rotation_sequence = ""
                rot_idx = []
                for dof in segment_dict[segment_name]["dof_idx"]:
                    if dof_names[dof][-6:-1] != "Trans":  # Skip translations
                        rotation_sequence += dof_names[dof][-1]
                        rot_idx += [dof]
                if rotation_sequence != "XX":
                    rotation_sequence = rotation_sequence.lower()
                    for i_frame in range(q.shape[0]):
                        rot = q[i_frame, rot_idx]
                        rotation_matrix = biorbd.Rotation.fromEulerAngles(rot, rotation_sequence)
                        q[i_frame, rot_idx] = biorbd.Rotation.toEulerAngles(rotation_matrix, rotation_sequence).to_array()
                    q[:, rot_idx] = np.unwrap(q[:, rot_idx], axis=0)
            return q


        def filter_kinematics(q_unwrapped):
            filter_type = "savgol"  # "filtfilt"

            # Filter q
            sampling_rate = 1 / (self.t[1] - self.t[0])
            if filter_type == "savgol":
                q_filtered = Operator.apply_savgol(q_unwrapped, window_length=31, polyorder=3)
            elif filter_type == "filtfilt":
                q_filtered = Operator.apply_filtfilt(q_unwrapped, order=4, sampling_rate=sampling_rate, cutoff_freq=6)
            else:
                raise NotImplementedError(f"filter_type {filter_type} not implemented. It must be 'savgol' or 'filtfilt'.")

            # Compute and filter qdot
            qdot = np.zeros_like(q_unwrapped)
            for i_data in range(qdot.shape[1]):
                qdot[0, i_data] = (q_filtered[1, i_data] - q_filtered[0, i_data]) / (self.t[1] - self.t[0])  # Forward finite diff
                qdot[1:-1, i_data] = (q_filtered[2:, i_data] - q_filtered[:-2, i_data]) / (self.t[2:] - self.t[:-2])  # Centered finite diff
                qdot[-1, i_data] = (q_filtered[-1, i_data] - q_filtered[-2, i_data]) / (self.t[-1] - self.t[-2])  # Backward finite diff

            # Compute and filter qddot
            qddot = np.zeros_like(q_unwrapped)
            for i_data in range(qddot.shape[1]):
                qddot[0, i_data] = (qdot[1, i_data] - qdot[0, i_data]) / (self.t[1] - self.t[0])
                qddot[1:-1, i_data] = (qdot[2:, i_data] - qdot[:-2, i_data]) / (self.t[2:] - self.t[:-2])
                qddot[-1, i_data] = (qdot[-1, i_data] - qdot[-2, i_data]) / (self.t[-1] - self.t[-2])

            return q_filtered, qdot, qddot

        q_unwrapped = unwrap_kinematics(self.biorbd_model, self.q)
        self.q_filtered, self.qdot, self.qddot = filter_kinematics(q_unwrapped)


    def plot_kinematics(self):
        all_in_one = True
        if all_in_one:
            fig = plt.figure(figsize=(10, 10))
            for i_dof in range(self.q.shape[1]):
                if i_dof < 3:
                    plt.plot(self.t, self.q_filtered[:, i_dof], label=f"{self.biorbd_model.nameDof()[i_dof].to_string()} [m]")
                else:
                    plt.plot(self.t, self.q_filtered[:, i_dof] * 180 / np.pi, label=f"{self.biorbd_model.nameDof()[i_dof].to_string()} [" + r"$^\circ$" + "]")
            plt.legend()
            fig.tight_layout()
            result_file_full_path = self.get_result_file_full_path()
            fig.savefig(result_file_full_path.replace(".pkl", "_ALL_IN_ONE.png"))
        else:
            fig, axs = plt.subplots(7, 6, figsize=(10, 10))
            axs = axs.ravel()
            for i_dof in range(self.q.shape[1]):
                if i_dof < 3:
                    axs[i_dof].plot(self.t, self.q_filtered[:, i_dof])
                    axs[i_dof].set_title(f"{self.biorbd_model.nameDof()[i_dof].to_string()} [m]")
                else:
                    axs[i_dof].plot(self.t, self.q_filtered[:, i_dof] * 180 / np.pi)
                    axs[i_dof].set_title(f"{self.biorbd_model.nameDof()[i_dof].to_string()} [" + r"$^\circ$" + "]")
            fig.tight_layout()
            result_file_full_path = self.get_result_file_full_path()
            fig.savefig(result_file_full_path.replace(".pkl", ".png"))


    def animate_kinematics(self):
        """
        Animate the kinematics
        """

        # Model
        # model = BiorbdModel.from_biorbd_object(self.biorbd_model)
        model = BiorbdModel(self.biorbd_model_creator.biorbd_model_virtual_markers_full_path)
        model.options.transparent_mesh = False

        # Markers
        marker_names = [m.to_string() for m in self.biorbd_model.markerNames()]
        markers = Markers(data=self.experimental_data.markers_sorted_with_virtual[:, :, :self.max_frames], channels=marker_names)

        # Visualization
        viz = PhaseRerun(self.t)
        if self.q.shape[0] == self.biorbd_model.nbQ():
            q_animation = self.q[:, :self.max_frames].reshape(self.biorbd_model.nbQ(), self.max_frames)
        else:
            q_animation = self.q[:self.max_frames, :].T
        viz.add_animated_model(model, q_animation, tracked_markers=markers)
        viz.rerun_by_frame("Kinematics reconstruction")


    def get_result_file_full_path(self):
        result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_file_name.split('/')[-1][:-4]
        result_file_full_path = f"{result_folder}/inv_kin_{trial_name}.pkl"
        return result_file_full_path


    def save_kinematics_reconstruction(self):
        """
        Save the kinematics reconstruction.
        """
        result_file_full_path = self.get_result_file_full_path()
        with open(result_file_full_path, "wb") as file:
            pickle.dump(self.outputs(), file)

    def inputs(self):
        return {
            "biorbd_model": self.biorbd_model,
            "c3d_full_file_path": self.experimental_data.c3d_full_file_path,
        }

    def outputs(self):
        return {
            "t": self.t,
            "q": self.q,
            "q_filtered": self.q_filtered,
            "qdot": self.qdot,
            "qddot": self.qddot,
            "is_loaded_kinematics": self.is_loaded_kinematics,
        }

