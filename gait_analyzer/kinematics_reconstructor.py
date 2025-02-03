import numpy as np

import casadi as cas
import biorbd
import biorbd_casadi
from pyorerun import BiorbdModel, PhaseRerun
from pyomeca import Markers

from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.biomod_model_creator import BiomodModelCreator
from gait_analyzer.events import Events


# TODO: Charbie -> See if we keep the virtual markers and if we impose this way of reconstructing


class KinematicsReconstructor:
    """
    This class reconstruct the kinematics based on the marker position and the model predefined.
    """

    def __init__(self, 
                 experimental_data: ExperimentalData, 
                 biorbd_model_creator: BiomodModelCreator, 
                 events: Events,
                 animate_kinematics_flag: bool = False):
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
        self.qdot = None
        self.qddot = None

        # Perform the kinematics reconstruction
        self.perform_kinematics_reconstruction()

        if animate_kinematics_flag:
            self.animate_kinematics()


    def extended_kalman_filter(self):

        print("Performing kinematics reconstruction using Extended Kalman Filter...")

        model = biorbd.Model(self.biorbd_model_creator.biorbd_model_virtual_markers_full_path)
        self.biorbd_model = model
        markers = self.experimental_data.markers_sorted_with_virtual

        # Dispatch markers in biorbd structure so EKF can use it
        markers_over_frames = []
        for i in range(markers.shape[2]):
            markers_over_frames.append([biorbd.NodeSegment(m) for m in markers[:, :, i].T])

        # Create a Kalman filter structure
        params = biorbd.KalmanParam(frequency=self.experimental_data.marker_sampling_frequency,
                                    noiseFactor=1e-8,
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
        """
        Sliding window version
        """
        model = biorbd_casadi.Model(self.biorbd_model_creator.biorbd_model_virtual_markers_full_path)
        markers = self.experimental_data.markers_sorted_with_virtual
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
                        ubx += [dof.max()]
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



    def constrained_optimization_reconstruction_cyclic(self):
        """
        Cyclic version
        """
        model = biorbd_casadi.Model(self.biorbd_model_creator.biorbd_model_virtual_markers_full_path)
        markers = self.experimental_data.markers_sorted_with_virtual
        nb_markers = markers.shape[1]
        dt = self.experimental_data.markers_dt

        q_mx, qdot_mx, qddot_mx, x, obj = None, None, None, None, None
        q_recons = np.zeros((model.nbQ(), markers.shape[2]))
        qdot_recons = np.zeros((model.nbQ(), markers.shape[2]))
        qddot_recons = np.zeros((model.nbQ(), markers.shape[2]))
        
        heel_touch_idx = self.events.events["right_leg_heel_touch"]
        for i_cycle in range(5):  # range(len(heel_touch_idx)-1):
            start_idx = heel_touch_idx[i_cycle]
            end_idx = heel_touch_idx[i_cycle+1]
            nb_frames = end_idx - start_idx

            # Min max bounds (constant for each window)
            lbx = []
            ubx = []
            for i_frame in range(nb_frames):
                for segment in model.segments():
                    if len(segment.QRanges()) > 0:
                        for dof in segment.QRanges():
                            lbx += [dof.min()]
                            ubx += [dof.max()]
            lbx += [-10] * model.nbQ() * nb_frames
            ubx += [10] * model.nbQ() * nb_frames
            lbx += [-100] * model.nbQ() * nb_frames
            ubx += [100] * model.nbQ() * nb_frames
            
            # Initialization
            x0 = np.zeros((model.nbQ() * nb_frames * 3))
            if i_cycle > 0:
                last_start_idx = heel_touch_idx[i_cycle-1]
                for i_frame in range(nb_frames):
                    x0[i_frame * model.nbQ() : (i_frame + 1) * model.nbQ()] = q_recons[:, last_start_idx+i_frame]
                    x0[model.nbQ() * nb_frames + i_frame * model.nbQ() : model.nbQ() * nb_frames + model.nbQ() * (i_frame+1)] = qdot_recons[:, last_start_idx+i_frame]
                    x0[model.nbQ() * nb_frames * 2 + i_frame * model.nbQ() : model.nbQ() * nb_frames * 2 + model.nbQ() * (i_frame+1)] = qddot_recons[:, last_start_idx+i_frame]

            # Initialize variables
            del q_mx, qdot_mx, qddot_mx, x, obj
            x = cas.MX.sym("x", model.nbQ() * nb_frames * 3)
            q_mx = cas.MX.zeros(model.nbQ(), nb_frames)
            qdot_mx = cas.MX.zeros(model.nbQ(), nb_frames)
            qddot_mx = cas.MX.zeros(model.nbQ(), nb_frames)
            for i_frame in range(nb_frames):
                q_mx[:, i_frame] = x[i_frame * model.nbQ() : (i_frame + 1) * model.nbQ()]
                qdot_mx[:, i_frame] = x[model.nbQ() * nb_frames + i_frame * model.nbQ() : model.nbQ() * nb_frames + (i_frame + 1) * model.nbQ()]
                qddot_mx[:, i_frame] = x[model.nbQ() * nb_frames * 2 + i_frame * model.nbQ() : model.nbQ() * nb_frames * 2 + (i_frame + 1) * model.nbQ()]

            obj = 0
            g = []
            lbg = []
            ubg = []
            for i_frame in range(nb_frames):
                i_data = start_idx + i_frame
                biorbd_markers = model.markers(q_mx[:, i_frame])
                for i_marker in range(nb_markers):
                    obj += cas.sumsqr(markers[:, i_marker, i_data] - biorbd_markers[i_marker].to_mx())
                obj += cas.sumsqr((q_mx[:, i_frame] - q_mx[:, i_frame - 1]) - dt * qdot_mx[:, i_frame])
                # g += [q_mx[:, i] - q_mx[:, i - 1] - dt * qdot_mx[:, i]]
                # lbg += [0] * model.nbQ()
                # ubg += [0] * model.nbQ()
                if 0 < i_frame < nb_frames - 1:
                    dq = (q_mx[:, i_frame] - q_mx[:, i_frame - 1]) / dt
                    g += [dq - qdot_mx[:, i_frame]]
                    lbg += [0] * model.nbQ()
                    ubg += [0] * model.nbQ()
                if 2 < i_frame < nb_frames - 2:
                    ddq = (qdot_mx[:, i_frame] - qdot_mx[:, i_frame - 1]) / dt
                    g += [ddq - qddot_mx[:, i_frame]]
                    lbg += [0] * model.nbQ()
                    ubg += [0] * model.nbQ()
                # TODO: Charbie -> add Kalman update minimization here if estimation is bad

            nlp = {"x": x, "f": obj, "g": cas.vertcat(*g)}
            opts = {}
            solver = cas.nlpsol("solver", "ipopt", nlp, opts)
            res = solver(x0=x0, lbx=cas.vertcat(*lbx), ubx=cas.vertcat(*ubx), lbg=cas.vertcat(*lbg), ubg=cas.vertcat(*ubg))

            # Keep the middle frame
            for i_frame in range(nb_frames):
                q_recons[:, start_idx+i_frame] = np.reshape(res["x"][model.nbQ() * i_frame : model.nbQ() * (i_frame+1)], (model.nbQ(), ))
                qdot_recons[:, start_idx+i_frame] = np.reshape(res["x"][model.nbQ() * nb_frames + model.nbQ() * i_frame : model.nbQ() * nb_frames + model.nbQ() * (i_frame+1)], (model.nbQ(), ))
                qddot_recons[:, start_idx+i_frame] = np.reshape(res["x"][model.nbQ() * nb_frames * 2 + model.nbQ() * i_frame : model.nbQ() * nb_frames * 2 + model.nbQ() * (i_frame+1)], (model.nbQ(), ))
        
        return q_recons, qdot_recons, qddot_recons


    def perform_kinematics_reconstruction(self):
        # self.q, self.qdot, self.qddot = self.extended_kalman_filter()
        # self.q, self.qdot, self.qddot = self.constrained_optimization_reconstruction()
        self.q, self.qdot, self.qddot = self.constrained_optimization_reconstruction_cyclic()
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
        markers = Markers(data=self.experimental_data.markers_sorted_with_virtual, channels=marker_names)

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
