import numpy as np
import biorbd

from gait_analyzer.experimental_data import ExperimentalData


class InverseDynamicsPerformer:
    """
    This class performs the inverse dynamics based on the kinematics and the external forces.
    """

    def __init__(self, 
                 experimental_data: ExperimentalData, 
                 biorbd_model: biorbd.Model, 
                 q: np.ndarray, 
                 qdot: np.ndarray, 
                 qddot: np.ndarray):
        """
        Initialize the InverseDynamicsPerformer.
        .
        Parameters
        ----------
        experimental_data: ExperimentalData
            The experimental data from the trial
        biorbd_model: biorbd.Model
            The biorbd model to use for the inverse dynamics
        q: np.ndarray()
            The generalized coordinates
        qdot: np.ndarray()
            The generalized velocities
        qddot: np.ndarray()
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
        self.q = q
        self.qdot = qdot
        self.qddot = qddot

        # Extended attributes
        self.tau = np.ndarray(())

        # Perform the inverse dynamics
        self.perform_inverse_dynamics()


    def perform_inverse_dynamics(self):
        """
        Code adapted from ophlariviere's biomechanics_tools
        Modifications:
        - Do not filter q, qdot, qddot since the kalman filter should already do the job.
        - Use only the force plate data at the frames where the marker positions were recorded (taking the mean of frames [i-5:i+5])
        """
        tau = np.zeros_like(self.q)
        for i_node in range(self.q.shape[1]):
            f_ext = self.experimental_data.get_f_ext_at_frame(i_node)
            tau[:, i_node] = self.biorbd_model.InverseDynamics(self.q[:, i_node], self.qdot[:, i_node], self.qddot[:, i_node], f_ext)
        self.tau = tau


    def inputs(self):
        return {
            "biorbd_model": self.biorbd_model,
            "experimental_data": self.experimental_data,
            "q": self.q,
            "qdot": self.qdot,
            "qddot": self.qddot,
        }

    def outputs(self):
        return {
            "tau": self.tau,
        }
