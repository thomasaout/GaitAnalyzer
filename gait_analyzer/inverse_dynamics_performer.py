import numpy as np
import biorbd

from gait_analyzer import Operator
from gait_analyzer.experimental_data import ExperimentalData


class InverseDynamicsPerformer:
    """
    This class performs the inverse dynamics based on the kinematics and the external forces.
    """

    def __init__(
        self,
        experimental_data: ExperimentalData,
        biorbd_model: biorbd.Model,
        q_filtered: np.ndarray,
        qdot_filtered: np.ndarray,
        qddot_filtered: np.ndarray,
    ):
        """
        Initialize the InverseDynamicsPerformer.
        .
        Parameters
        ----------
        experimental_data: ExperimentalData
            The experimental data from the trial
        biorbd_model: biorbd.Model
            The biorbd model to use for the inverse dynamics
        q_filtered: np.ndarray()
            The generalized coordinates
        qdot_filtered: np.ndarray()
            The generalized velocities
        qddot_filtered: np.ndarray()
            The generalized accelerations
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
        self.q_filtered = q_filtered
        self.qdot_filtered = qdot_filtered
        self.qddot_filtered = qddot_filtered

        # Extended attributes
        self.tau = np.ndarray(())

        # Perform the inverse dynamics
        self.perform_inverse_dynamics()

    def perform_inverse_dynamics(self):
        print("Performing inverse dynamics...")
        tau = np.zeros_like(self.q_filtered)
        for i_node in range(self.q_filtered.shape[0]):
            f_ext = self.get_f_ext_at_frame(i_node)
            tau[i_node, :] = self.biorbd_model.InverseDynamics(
                self.q_filtered[i_node, :], self.qdot_filtered[i_node, :], self.qddot_filtered[i_node, :], f_ext
            ).to_array()
        self.tau = tau

    def get_f_ext_at_frame(self, i_marker_node: int):
        """
        Constructs a biorbd external forces set object at a specific frame.
        .
        Parameters
        ----------
        i_marker_node: int
            The marker frame index.
        .
        Returns
        -------
        f_ext_set: biorbd externalForceSet
            The external forces set at the frame.
        """
        f_ext_set = self.biorbd_model.externalForceSet()
        i_analog_node = Operator.from_marker_frame_to_analog_frame(
            self.experimental_data.analogs_time_vector, self.experimental_data.markers_time_vector, i_marker_node
        )
        analog_to_marker_ratio = int(
            round(
                self.experimental_data.analogs_time_vector.shape[0]
                / self.experimental_data.markers_time_vector.shape[0]
            )
        )
        frame_range = list(
            range(i_analog_node - (int(analog_to_marker_ratio / 2)), i_analog_node + (int(analog_to_marker_ratio / 2)))
        )
        # Average over the marker frame time lapse
        f_ext_set.add(
            "calcn_l",
            np.mean(self.experimental_data.f_ext_sorted[0, 3:, frame_range], axis=0),
            np.mean(self.experimental_data.f_ext_sorted[0, :3, frame_range], axis=0),
        )
        f_ext_set.add(
            "calcn_r",
            np.mean(self.experimental_data.f_ext_sorted[1, 3:, frame_range], axis=0),
            np.mean(self.experimental_data.f_ext_sorted[1, :3, frame_range], axis=0),
        )
        return f_ext_set

    def inputs(self):
        return {
            "biorbd_model": self.biorbd_model,
            "experimental_data": self.experimental_data,
            "q_filtered": self.q_filtered,
            "qdot_filtered": self.qdot_filtered,
            "qddot_filtered": self.qddot_filtered,
        }

    def outputs(self):
        return {
            "tau": self.tau,
        }
