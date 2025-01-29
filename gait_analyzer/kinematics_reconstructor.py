import numpy as np
import biorbd

from gait_analyzer.experimental_data import ExperimentalData


class KinematicsReconstructor:
    """
    This class reconstruct the kinematics based on the marker position and the model predefined.
    """

    def __init__(self, experimental_data: ExperimentalData, biorbd_model: biorbd.Model, animate_kinematics_flag: bool = False):
        """
        Initialize the KinematicsReconstructor.
        .
        Parameters
        ----------
        experimental_data: ExperimentalData
            The experimental data from the trial
        biomod_model: biorbd.Model
            The biorbd model to use for the kinematics reconstruction
        animate_kinematics_flag: bool
            If True, the kinematics will be animated through pyorerun
        """

        # Checks
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError(
                "experimental_data must be an instance of ExperimentalData. You can declare it by running ExperimentalData(file_path)."
            )
        if not isinstance(biorbd_model, biorbd.Model):
            raise ValueError(
                "biorbd_model must be an instance of biorbd.Model. You can declare it by running biorbd.Model('path_to_model.bioMod')."
            )

        # Initial attributes
        self.experimental_data = experimental_data
        self.biorbd_model = biorbd_model

        # Extended attributes
        self.q = np.ndarray(())
        self.qdot = np.ndarray(())
        self.qddot = np.ndarray(())

        # Perform the kinematics reconstruction
        self.perform_kinematics_reconstruction()

        if animate_kinematics_flag:
            self.animate_kinematics()

    def perform_kinematics_reconstruction(self):
        self.t, self.q, self.qdot, self.qddot = biorbd.extended_kalman_filter(
            self.biorbd_model, self.experimental_data.c3d_full_file_path
        )

    def animate_kinematics(self):
        """
        Animate the kinematics
        """
        # TODO: Charbie -> Animate the kinematics using pyorerun
        pass

    def inputs(self):
        return {
            "biorbd_model": self.biorbd_model,
            "c3d_full_file_path": self.experimental_data.c3d_full_file_path,
        }

    def outputs(self):
        return {
            "q": self.q,
            "qdot": self.qdot,
            "qddot": self.qddot,
        }
