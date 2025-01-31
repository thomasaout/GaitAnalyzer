import numpy as np

import casadi as cas
import biorbd
import biorbd_casadi
from pyorerun import BiorbdModel, PhaseRerun
from pyomeca import Markers

from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.biomod_model_creator import BiomodModelCreator


class KinematicsReconstructor:
    """
    This class reconstruct the kinematics based on the marker position and the model predefined.
    """

    def __init__(self, experimental_data: ExperimentalData, biorbd_model_creator: BiomodModelCreator, animate_kinematics_flag: bool = False):
        """
        Initialize the KinematicsReconstructor.
        .
        Parameters
        ----------
        experimental_data: ExperimentalData
            The experimental data from the trial
        biorbd_model_creator: BiomodModelCreator
            The biorbd model to use for the kinematics reconstruction
        animate_kinematics_flag: bool
            If True, the kinematics will be animated through pyorerun
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

        # Initial attributes
        self.experimental_data = experimental_data
        self.biorbd_model_creator = biorbd_model_creator

        # Extended attributes
        self.t = None
        self.q = None
        self.qdot = None
        self.qddot = None

        # Perform the kinematics reconstruction
        self.perform_kinematics_reconstruction()

        if animate_kinematics_flag:
            self.animate_kinematics()


    def extended_kalman_filter(self):

        model = self.biorbd_model_creator.biorbd_model
        markers = self.experimental_data.markers_sorted

        # Dispatch markers in biorbd structure so EKF can use it
        markers_over_frames = []
        for i in range(markers.shape[2]):
            markers_over_frames.append([biorbd.NodeSegment(m) for m in markers[:, :, i].T])

        # Create a Kalman filter structure
        params = biorbd.KalmanParam(frequency=self.experimental_data.marker_sampling_frequency,
                                    noiseFactor=1e-3,
                                    errorFactor=1e-2)
        kalman = biorbd.KalmanReconsMarkers(model, params)

        # Perform the kalman filter for each frame (the first frame is much longer than the next)
        q = biorbd.GeneralizedCoordinates(model)
        qdot = biorbd.GeneralizedVelocity(model)
        qddot = biorbd.GeneralizedAcceleration(model)
        q_recons = np.ndarray((model.nbQ(), len(markers_over_frames)))
        qdot_recons = np.ndarray((model.nbQ(), len(markers_over_frames)))
        qddot_recons = np.ndarray((model.nbQ(), len(markers_over_frames)))
        for i, targetMarkers in enumerate(markers_over_frames):
            kalman.reconstructFrame(model, targetMarkers, q, qdot, qddot)
            q_recons[:, i] = q.to_array()
            qdot_recons[:, i] = qdot.to_array()
            qddot_recons[:, i] = qddot.to_array()
        return q_recons, qdot_recons, qddot_recons


    def constrained_optimization_reconstruction(self):

        model = biorbd_casadi.Model(self.biorbd_model_creator.biorbd_model_full_path)
        markers = self.experimental_data.markers_sorted
        nb_markers = markers.shape[1]
        nb_frames = markers.shape[2]
        nb_frames_window = 30  # Number of frames to consider for the optimization (must be odd)
        dt = self.experimental_data.markers_dt

        q_recons = np.zeros((model.nbQ(), nb_frames))
        qdot_recons = np.zeros((model.nbQ(), nb_frames))
        qddot_recons = np.zeros((model.nbQ(), nb_frames))

        # Min max bounds (constant for each window)
        lbx = []
        ubx = []
        for i_frame in range(nb_frames_window):
            for segment in model.segments():
                if len(segment.QRanges()) > 0:
                    for dof in segment.QRanges():
                        lbx += [dof.min()]
                        ubx += [dof.min()]
        lbx += [-10] * model.nbQ() * nb_frames_window
        ubx += [10] * model.nbQ() * nb_frames_window
        lbx += [-100] * model.nbQ() * nb_frames_window
        ubx += [100] * model.nbQ() * nb_frames_window
        x0 = np.random.random((model.nbQ() * nb_frames_window * 3)) * 0.1

        q_mx, qdot_mx, qddot_mx, x, obj = None, None, None, None, None

        for current_index in range(int(nb_frames_window / 2), nb_frames - int(nb_frames_window / 2)):

            # Initialize variables
            del q_mx, qdot_mx, qddot_mx, x, obj
            x = cas.MX.sym("x", model.nbQ() * nb_frames_window * 3)
            q_mx = cas.MX.zeros(model.nbQ(), nb_frames_window)
            qdot_mx = cas.MX.zeros(model.nbQ(), nb_frames_window)
            qddot_mx = cas.MX.zeros(model.nbQ(), nb_frames_window)
            for i_frame in range(nb_frames_window):
                q_mx[:, i_frame] = x[i_frame * model.nbQ() : (i_frame + 1) * model.nbQ()]
                qdot_mx[:, i_frame] = x[model.nbQ() * nb_frames_window + i_frame * model.nbQ() : model.nbQ() * nb_frames_window + (i_frame + 1) * model.nbQ()]
                qddot_mx[:, i_frame] = x[model.nbQ() * nb_frames_window * 2 + i_frame * model.nbQ() : model.nbQ() * nb_frames_window * 2 + (i_frame + 1) * model.nbQ()]

            obj = 0
            g = []
            lbg = []
            ubg = []
            for i in range(nb_frames_window):
                i_data = current_index - int(nb_frames_window / 2) + i
                biorbd_markers = model.markers(q_mx[:, i])
                for i_marker in range(nb_markers):
                    obj += cas.sumsqr(markers[:, i_marker, i_data] - biorbd_markers[i_marker].to_mx())
                g += [q_mx[:, i] - q_mx[:, i - 1] - dt * qdot_mx[:, i]]
                lbg += [0] * model.nbQ()
                ubg += [0] * model.nbQ()
                if 0 < i < nb_frames_window - 1:
                    dq = (q_mx[:, i] - q_mx[:, i - 1]) / dt
                    g += [dq - qdot_mx[:, i]]
                    lbg += [0] * model.nbQ()
                    ubg += [0] * model.nbQ()
                if 2 < i < nb_frames_window - 2:
                    ddq = (qdot_mx[:, i] - qdot_mx[:, i - 1]) / dt
                    g += [ddq - qddot_mx[:, i]]
                    lbg += [0] * model.nbQ()
                    ubg += [0] * model.nbQ()
                # TODO: Charbie -> add Kalman update minimization here if estimation is bad

            nlp = {"x": x, "f": obj, "g": cas.vertcat(*g)}
            opts = {}
            solver = cas.nlpsol("solver", "ipopt", nlp, opts)
            res = solver(x0=x0, lbx=cas.vertcat(*lbx), ubx=cas.vertcat(*ubx), lbg=cas.vertcat(*lbg), ubg=cas.vertcat(*ubg))

            # Keep the middle frame
            q_recons[:, current_index] = np.reshape(res["x"][model.nbQ() * int(nb_frames_window / 2) : model.nbQ() * (int(nb_frames_window / 2) + 1)], (model.nbQ(), ))
            qdot_recons[:, current_index] = np.reshape(res["x"][model.nbQ() * nb_frames_window + model.nbQ() * int(nb_frames_window / 2) : model.nbQ() * nb_frames_window + model.nbQ() * (int(nb_frames_window / 2) + 1)], (model.nbQ(), ))
            qddot_recons[:, current_index] = np.reshape(res["x"][model.nbQ() * nb_frames_window * 2 + model.nbQ() * int(nb_frames_window / 2) : model.nbQ() * nb_frames_window * 2 + model.nbQ() * (int(nb_frames_window / 2) + 1)], (model.nbQ(), ))

        return q_recons, qdot_recons, qddot_recons



    def perform_kinematics_reconstruction(self):
        # self.q, self.qdot, self.qddot = self.extended_kalman_filter()
        self.q, self.qdot, self.qddot = self.constrained_optimization_reconstruction()
        self.t = self.experimental_data.markers_time_vector

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(self.t, self.q[3, :])
        # plt.plot(self.t, self.q[6, :])
        # plt.plot(self.t, self.q[9, :])
        # plt.savefig("qqq.png")
        # plt.show()


    def animate_kinematics(self):
        """
        Animate the kinematics
        """

        # Model
        model = BiorbdModel.from_biorbd_object(self.biorbd_model)
        model.options.transparent_mesh = False

        # Markers
        marker_names = [m.to_string() for m in self.biorbd_model.markerNames()]
        markers = Markers(data=self.experimental_data.markers_sorted, channels=marker_names)

        # Visualization
        viz = PhaseRerun(self.t)
        viz.add_animated_model(model, self.q, tracked_markers=markers)
        viz.rerun_by_frame("Kinematics reconstruction")


    def inputs(self):
        return {
            "biorbd_model": self.biorbd_model,
            "c3d_full_file_path": self.experimental_data.c3d_full_file_path,
        }

    def outputs(self):
        return {
            "t": self.t,
            "q": self.q,
            "qdot": self.qdot,
            "qddot": self.qddot,
        }
