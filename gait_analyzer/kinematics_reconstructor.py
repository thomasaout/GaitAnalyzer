import numpy as np

import casadi as cas
import biorbd
import biorbd_casadi
from pyorerun import BiorbdModel, PhaseRerun
from pyomeca import Markers
from scipy.optimize import minimize

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
        q_recons = np.ndarray((len(markers_over_frames), model.nbQ()))
        qdot_recons = np.ndarray((len(markers_over_frames), model.nbQ()))
        qddot_recons = np.ndarray((len(markers_over_frames), model.nbQ()))
        for i, targetMarkers in enumerate(markers_over_frames):
            kalman.reconstructFrame(model, targetMarkers, q, qdot, qddot)
            q_recons[i, :] = q.to_array()
            qdot_recons[i, :] = qdot.to_array()
            qddot_recons[i, :] = qddot.to_array()
        return q_recons, qdot_recons, qddot_recons


    def constrained_extended_kalman_filter(self):
        # kalman = CustomConstrainedKalmanFilter(
        #     model = self.biorbd_model,
        #     initial_state = np.zeros((nb_q, )),
        #     initial_covariance = np.eye(self.biorbd_model.nbQ()) * 1e-5,
        #     process_noise = np.eye(self.biorbd_model.nbQ()) * 1e-5,
        #     measurement_noise = np.eye(self.biorbd_model.nbQ()) * 1e-5
        # )

        self.biorbd_model = biorbd.Model(self.biorbd_model_creator.biorbd_model_virtual_markers_full_path)
        nb_frames = self.experimental_data.markers_sorted_with_virtual.shape[2]
        nb_markers = self.experimental_data.markers_sorted_with_virtual.shape[1]
        nb_q = self.biorbd_model.nbQ()
        kalman = ConstrainedKalmanRecons(self.biorbd_model,
                                         nb_frames = nb_frames,
                                         nb_markers = nb_markers,
                                         noise_factor = 1e-10,
                                         error_factor = 1e-5)

        q_recons = np.zeros((nb_q, 10))
        for i in range(10):
            q_recons[:, i] = kalman.get_optimal_states(self.biorbd_model_creator, self.experimental_data.markers_sorted_with_virtual[:, :, i])
        qdot_recons = None
        qddot_recons = None
        #
        #
        # # Perform the kalman filter for each frame (the first frame is much longer than the next)
        # q_recons = np.zeros((nb_q, self.experimental_data.markers_sorted_with_virtual.shape[2]))
        # qdot_recons = np.zeros((nb_q, self.experimental_data.markers_sorted_with_virtual.shape[2]))
        # qddot_recons = np.zeros((nb_q, self.experimental_data.markers_sorted_with_virtual.shape[2]))
        #
        # for i_frame in range(nb_frames):
        #
        #     # Get current experimental markers (nb_markers x 3)
        #     current_markers = self.experimental_data.markers_sorted_with_virtual[:, :, i_frame]
        #     occlusion = [i for i, marker in enumerate(current_markers) if np.all(np.isnan(marker))]
        #
        #     # If this is not the first frame, use previous q as initial guess
        #     if i_frame == 0:
        #         q_init, qdot_init, qddot_init = kalman.set_initial_states(current_markers)
        #         # q_init, qdot_init, qddot_init = kalman.set_optimal_initial_states(current_markers)
        #     else:
        #         q_init = q_recons[:, i_frame - 1]
        #         qdot_init = qdot_recons[:, i_frame - 1]
        #         qddot_init = qddot_recons[:, i_frame - 1]
        #     kalman.set_states(q_init = q_init,
        #                       qdot_init = qdot_init,
        #                       qddot_init = qddot_init)
        #
        #     # Get current state estimate
        #     q_current = kalman.xp[:kalman.nb_dof]
        #
        #     # Calculate expected marker positions and Jacobian
        #     markers_projected = np.zeros((3 * nb_markers))
        #     H = np.zeros((3 * nb_markers, 3 * self.biorbd_model.nbQ()))
        #     for i_marker in range(nb_markers):
        #         if i_marker not in occlusion:
        #             markers_projected[3 * i_marker:3 * (i_marker + 1)] = self.biorbd_model.markers(q_current)[i_marker].to_array()
        #             H[3 * i_marker:3 * (i_marker + 1), :nb_q] = self.biorbd_model.markersJacobian(q_current)[i_marker].to_array()
        #
        #     # Perform Kalman iteration
        #     q, qdot, qddot = kalman.iteration(
        #         measure=current_markers.T.flatten(),  # Flatten to 3*nb_markers vector
        #         projected_measure=markers_projected,
        #         hessian=H,
        #         occlusion=occlusion  # Add occluded marker indices if needed
        #     )
        #     if np.all(np.abs(q) > 5*np.pi):
        #         break
        #
        #     # Store results
        #     q_recons[:, i_frame] = q
        #     qdot_recons[:, i_frame] = qdot
        #     qddot_recons[:, i_frame] = qddot

        return q_recons, qdot_recons, qddot_recons


    def markers_jacobian_thing(self):
        self.biorbd_model = biorbd.Model(self.biorbd_model_creator.biorbd_model_virtual_markers_full_path)
        nb_frames = self.experimental_data.markers_sorted_with_virtual.shape[2]
        nb_markers = self.experimental_data.markers_sorted_with_virtual.shape[1]
        nb_q = self.biorbd_model.nbQ()
        kalman = JacobianThing(self.experimental_data.marker_sampling_frequency,
                                 self.biorbd_model,
                                 nb_markers = nb_markers,
                                 noise_factor = 1e-10,
                                 error_factor = 1e-5)

        q_recons = np.zeros((nb_q, self.experimental_data.markers_sorted_with_virtual.shape[2]))
        # self.max_frames = nb_frames
        self.max_frames = 30
        for i_frame in range(self.max_frames):
            current_markers = self.experimental_data.markers_sorted_with_virtual[:, :, i_frame]
            q_current = kalman.set_initial_states(current_markers)
            kalman.xp = q_current

            if np.all(np.abs(q_current) > 5*np.pi):
                break

            q_recons[:, i_frame] = q_current

        return q_recons


    def constrained_optimization_reconstruction(self):
        """
        Sliding window version
        """
        model = biorbd_casadi.Model(self.biorbd_model_creator.biorbd_model_virtual_markers_full_path)
        self.biorbd_model = model
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

        for current_index in range(int(nb_frames_window / 2), int(nb_frames_window / 2) + 5):  # nb_frames - int(nb_frames_window / 2)):

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
        self.biorbd_model = model
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
            print(f"Reconstructing optimally frames {start_idx} to {end_idx}...")

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


    def inverse_kinematics(self):

        print("Performing inverse kinematics reconstruction using !!!!!!! ...")

        model = biorbd.Model(self.biorbd_model_creator.biorbd_model_virtual_markers_full_path)
        self.biorbd_model = model
        markers = self.experimental_data.markers_sorted_with_virtual
        nb_frames = markers.shape[2]
        self.max_frames = nb_frames
        # self.max_frames = 30

        # qinit = np.array([0.1, 0.1, -0.3, 0.35, 1.15, -0.35, 1.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        q_recons = np.ndarray((nb_frames, model.nbQ()))
        ik = biorbd.InverseKinematics(model, markers[:, :, :self.max_frames])
        q_recons[:self.max_frames, :] = ik.solve(method="only_lm").T

        return q_recons


    def iterative_kinematics_reconstruction(self):
        model = biorbd.Model(self.biorbd_model_creator.biorbd_model_virtual_markers_full_path)
        self.biorbd_model = model

        # Initialize reconstructor
        # nb_frames = self.experimental_data.markers_sorted_with_virtual.shape[2]
        # self.max_frames = nb_frames
        self.max_frames = 30
        reconstructor = IterativeKinematicsReconstructor(
            model=self.biorbd_model,
            nb_frames=self.max_frames,
            nb_markers=self.experimental_data.markers_sorted_with_virtual.shape[1],
            noise_factor=1e-10
        )

        # Reconstruct all frames
        q_recons = reconstructor.reconstruct_markers(
            self.experimental_data.markers_sorted_with_virtual[:, :, :self.max_frames]
        )
        return q_recons


    def perform_kinematics_reconstruction(self):
        # self.q, self.qdot, self.qddot = self.extended_kalman_filter()
        # self.q, self.qdot, self.qddot = self.constrained_extended_kalman_filter()
        # self.q = self.markers_jacobian_thing()
        # self.q, self.qdot, self.qddot = self.constrained_optimization_reconstruction()
        # self.q, self.qdot, self.qddot = self.constrained_optimization_reconstruction_cyclic()
        self.q = self.inverse_kinematics()
        # self.q = self.iterative_kinematics_reconstruction()
        self.t = self.experimental_data.markers_time_vector[:self.max_frames]

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


class ConstrainedKalmanRecons:
    def __init__(self,
                 model: biorbd.Model,
                 nb_frames: int,
                 nb_markers: int,
                 acquisition_frequency: float = 100.0,
                 noise_factor: float = 1e-3,
                 error_factor: float = 1e-3):
        self.acquisition_frequency = acquisition_frequency
        self.noise_factor = noise_factor
        self.error_factor = error_factor
        self.dt = 1.0 / self.acquisition_frequency
        self.model = model
        self.nb_dof = self.model.nbQ()
        self.nb_frames = nb_frames
        self.nb_markers = nb_markers
        self.Pp_initial = self._init_covariance()  # Store initial covariance

        # State vector contains [Q, Qdot, Qddot]
        self.xp = np.zeros(3 * self.nb_dof)

        # Initialize matrices
        self.A = self._evolution_matrix()
        self.Q = self._process_noise_matrix()
        self.R = self._measurement_noise_matrix()
        self.Pp = self._init_covariance()

        # Initialize joint constraints
        self._constraints = {}  # Dictionary to store joint bounds

    def set_joint_constraints(self):
        """
        Set joint angle constraints in the for "segment_name_dof_index": (min, max)
        """
        for segment in self.model.segments():
            if len(segment.QRanges()) > 0:
                for i_dof, dof in enumerate(segment.QRanges()):
                    self._constraints[f"{segment.name()}_{i_dof}"] += (dof.min(), dof.max())


    def _evolution_matrix(self) -> np.ndarray:
        """Create evolution matrix (equivalent to biorbd's evolutionMatrix)"""
        n = 2  # Order of Taylor development + 1 (position, velocity, acceleration)
        A = np.eye(self.nb_dof * (n + 1))
        c = 1.0

        for i in range(2, n + 2):
            j = (i - 1) * self.nb_dof
            c = c / (i - 1)

            for cmp in range(self.nb_dof * (n + 1) - j):
                A[cmp, j + cmp] += c * self.dt ** (i - 1)

        return A

    def _process_noise_matrix(self) -> np.ndarray:
        """Create process noise matrix (equivalent to biorbd's processNoiseMatrix)"""
        dt = self.dt
        c1 = 1 / 20.0 * dt ** 5
        c2 = 1 / 8.0 * dt ** 4
        c3 = 1 / 6.0 * dt ** 3
        c4 = 1 / 3.0 * dt ** 3
        c5 = 1 / 2.0 * dt ** 2
        c6 = dt

        q = np.zeros((3 * self.nb_dof, 3 * self.nb_dof))
        for j in range(self.nb_dof):
            q[j, j] = c1
            q[j, self.nb_dof + j] = c2
            q[j, 2 * self.nb_dof + j] = c3
            q[self.nb_dof + j, j] = c2
            q[self.nb_dof + j, self.nb_dof + j] = c4
            q[self.nb_dof + j, 2 * self.nb_dof + j] = c5
            q[2 * self.nb_dof + j, j] = c3
            q[2 * self.nb_dof + j, self.nb_dof + j] = c5
            q[2 * self.nb_dof + j, 2 * self.nb_dof + j] = c6

        return q

    def _measurement_noise_matrix(self) -> np.ndarray:
        """Create measurement noise matrix"""
        return np.eye(3 * self.nb_markers) * self.noise_factor

    def _init_covariance(self) -> np.ndarray:
        """Initialize covariance matrix"""
        return np.eye(3 * self.nb_dof) * self.error_factor

    def _apply_constraints(self):
        """Apply joint angle constraints"""
        for joint_idx, (min_bound, max_bound) in self._constraints.items():
            self.xp[joint_idx] = np.clip(self.xp[joint_idx], min_bound, max_bound)

            # If position is constrained, set velocity to 0 at bounds
            if self.xp[joint_idx] in (min_bound, max_bound):
                self.xp[self.nb_dof + joint_idx] = 0
                self.xp[2 * self.nb_dof + joint_idx] = 0

    def iteration(self, measure: np.ndarray,
                  projected_measure: np.ndarray,
                  hessian: np.ndarray,
                  occlusion = None):
        """
        Perform one iteration of the constrained Kalman filter
        Args:
            measure: Actual marker positions
            projected_measure: Expected marker positions from model
            hessian: Jacobian of marker positions with respect to joint angles
            occlusion: List of occluded marker indices
        """
        # Prediction
        xkm = self.A @ self.xp
        Pkm = self.A @ self.Pp @ self.A.T # + self.Q  TODO: Charbie -> see why Q is detrimental

        # Handle occlusions
        if occlusion:
            for idx in occlusion:
                measure[idx:idx + 3] = 0
                self.R[idx:idx + 3, idx:idx + 3] = np.inf * np.eye(3)

        # Innovation
        S = hessian @ Pkm @ hessian.T + self.R
        K = Pkm @ hessian.T @ np.linalg.inv(S)

        # Update
        self.xp = xkm + K @ (measure - projected_measure)
        temp = np.eye(3 * self.nb_dof) - K @ hessian
        self.Pp = temp @ Pkm @ temp.T + K @ self.R @ K.T

        # Apply constraints
        self._apply_constraints()

        # Get state estimates
        q, qdot, qddot = self.get_states()

        return q, qdot, qddot

    def get_optimal_states(self, biorbd_model_creator, experimental_markers):

        casadi_model = biorbd_casadi.Model(biorbd_model_creator.biorbd_model_virtual_markers_full_path)
        q_ok = np.zeros((casadi_model.nbQ(), ))
        q_current = cas.MX.zeros(casadi_model.nbQ())
        for i_segment, segment_name in enumerate(segment_dict):
            dof_idx = segment_dict[segment_name]["dof_idx"]
            nb_dofs = len(dof_idx)
            markers_idx = segment_dict[segment_name]["markers_idx"]
            lbx = segment_dict[segment_name]["min_bound"]
            ubx = segment_dict[segment_name]["max_bound"]
            x = cas.MX.sym("x", nb_dofs)
            q_current[:] = q_ok[:]
            q_current[dof_idx] = x
            markers = cas.horzcat(*[m.to_mx() for m in casadi_model.markers(q_current)])
            obj = cas.sumsqr(markers[:, markers_idx] - experimental_markers[:, markers_idx])
            obj += cas.sum1(1e-6 * x)
            nlp = {"x": x, "f": obj}
            solver = cas.nlpsol("solver", "ipopt", nlp, {})
            res = solver(lbx=cas.vertcat(*lbx), ubx=cas.vertcat(*ubx))
            q_ok[dof_idx] = np.reshape(res["x"], (nb_dofs, ))
        return q_ok


    def get_states(self):
        """Get current state estimates"""
        q = self.xp[:self.nb_dof]
        qdot = self.xp[self.nb_dof:2 * self.nb_dof]
        qddot = self.xp[2 * self.nb_dof:3 * self.nb_dof]
        return q, qdot, qddot

    def set_states(self,
                  q_init = None,
                  qdot_init = None,
                  qddot_init = None):
        """Set state values"""
        if q_init is not None:
            self.xp[:self.nb_dof] = q_init
        if qdot_init is not None:
            self.xp[self.nb_dof:2 * self.nb_dof] = qdot_init
        if qddot_init is not None:
            self.xp[2 * self.nb_dof:3 * self.nb_dof] = qddot_init
        # TODO : Charbie -> Ensure initial state satisfies constraints

    def set_initial_states(self, markers):
        """
        Initialize state using the strategy from biorbd:
        1. First estimate root position using only root markers (50 iterations)
        2. Then estimate full body position using all markers (50 iterations)

        Args:
            markers: Marker positions (3 x nb_markers)
        """
        markers_flat = markers.T.flatten()

        # First phase: root estimation
        nb_root_markers = 4
        markers_root = markers_flat.copy()

        # Zero out all non-root markers
        markers_root[3 * nb_root_markers:] = 0

        # Store original Pp
        Pp_original = self.Pp.copy()

        # First phase: estimate root position
        for _ in range(50):
            # Get current state estimate
            q_current = self.xp[:self.nb_dof]

            # Create Hessian matrix and projected markers
            H = np.zeros((3 * self.nb_markers, 3 * self.nb_dof))
            markers_projected = np.zeros(3 * self.nb_markers)

            # Fill only for root markers
            for i_marker in range(nb_root_markers):
                markers_projected[3 * i_marker:3 * (i_marker + 1)] = self.model.markers(q_current)[i_marker].to_array()
                H[3 * i_marker:3 * (i_marker + 1), :self.nb_dof] = self.model.markersJacobian(q_current)[i_marker].to_array()

            # Perform Kalman iteration
            self.iteration(
                measure=markers_root,
                projected_measure=markers_projected,
                hessian=H,
                occlusion=[]
            )

            # Reset Pp and velocities/accelerations
            self.Pp = self.Pp_initial.copy()
            self.xp[self.nb_dof:] = 0  # Set velocities and accelerations to zero

        # Second phase: full body estimation
        for _ in range(50):
            # Get current state estimate
            q_current = self.xp[:self.nb_dof]

            # Create Hessian matrix and projected markers
            H = np.zeros((3 * self.nb_markers, 3 * self.nb_dof))
            markers_projected = np.zeros(3 * self.nb_markers)

            # Fill for all markers
            for i_marker in range(self.nb_markers):
                marker_pos = self.model.markers(q_current)[i_marker].to_array()
                marker_jacobian = self.model.markersJacobian(q_current)[i_marker].to_array()

                H[3 * i_marker:3 * (i_marker + 1), :self.nb_dof] = marker_jacobian
                markers_projected[3 * i_marker:3 * (i_marker + 1)] = marker_pos

            # Perform Kalman iteration
            self.iteration(
                measure=markers_flat,
                projected_measure=markers_projected,
                hessian=H,
                occlusion=[]
            )

            # Reset Pp and velocities/accelerations
            self.Pp = self.Pp_initial.copy()
            self.xp[self.nb_dof:] = 0  # Set velocities and accelerations to zero

        # Restore original Pp
        self.Pp = Pp_original

        # Return the states
        q = self.xp[:self.nb_dof]
        qdot = self.xp[self.nb_dof:2 * self.nb_dof]
        qddot = self.xp[2 * self.nb_dof:3 * self.nb_dof]

        return q, qdot, qddot



    def set_optimal_initial_states(self, markers):
        markers_flat = markers.T.flatten()

        # First phase: root estimation
        nb_root_markers = 4  # Get number of markers on root segment
        markers_root = markers_flat.copy()

        # Zero out all non-root markers
        markers_root[3 * nb_root_markers:] = 0

        # Store original Pp
        Pp_original = self.Pp.copy()

        # First phase: estimate root position
        for _ in range(50):
            # Get current state estimate
            q_current = self.xp[:self.nb_dof]

            # Create Hessian matrix and projected markers
            H = np.zeros((3 * self.nb_markers, 3 * self.nb_dof))
            markers_projected = np.zeros(3 * self.nb_markers)

            # Fill only for root markers
            for i_marker in range(nb_root_markers):
                markers_projected[3 * i_marker:3 * (i_marker + 1)] = self.model.markers(q_current)[i_marker].to_array()
                H[3 * i_marker:3 * (i_marker + 1), :self.nb_dof] = self.model.markersJacobian(q_current)[i_marker].to_array()

            # Perform Kalman iteration
            self.iteration(
                measure=markers_root,
                projected_measure=markers_projected,
                hessian=H,
                occlusion=[]
            )

            # Reset Pp and velocities/accelerations
            self.Pp = self.Pp_initial.copy()
            self.xp[self.nb_dof:] = 0  # Set velocities and accelerations to zero

        # Second phase: full body estimation
        for _ in range(50):
            # Get current state estimate
            q_current = self.xp[:self.nb_dof]

            # Create Hessian matrix and projected markers
            H = np.zeros((3 * self.nb_markers, 3 * self.nb_dof))
            markers_projected = np.zeros(3 * self.nb_markers)

            # Fill for all markers
            for i_marker in range(self.nb_markers):
                marker_pos = self.model.markers(q_current)[i_marker].to_array()
                marker_jacobian = self.model.markersJacobian(q_current)[i_marker].to_array()

                H[3 * i_marker:3 * (i_marker + 1), :self.nb_dof] = marker_jacobian
                markers_projected[3 * i_marker:3 * (i_marker + 1)] = marker_pos

            # Perform Kalman iteration
            self.iteration(
                measure=markers_flat,
                projected_measure=markers_projected,
                hessian=H,
                occlusion=[]
            )

            # Reset Pp and velocities/accelerations
            self.Pp = self.Pp_initial.copy()
            self.xp[self.nb_dof:] = 0  # Set velocities and accelerations to zero

        # Restore original Pp
        self.Pp = Pp_original

        # Return the states
        q = self.xp[:self.nb_dof]
        qdot = self.xp[self.nb_dof:2 * self.nb_dof]
        qddot = self.xp[2 * self.nb_dof:3 * self.nb_dof]

        return q, qdot, qddot

class JacobianThing:
    def __init__(self,
                 acquisition_frequency: float,
                 model: biorbd.Model,
                 nb_markers: int,
                 noise_factor: float = 1e-3,
                 error_factor: float = 1e-3):
        self.acquisition_frequency = acquisition_frequency
        self.noise_factor = noise_factor
        self.error_factor = error_factor
        self.dt = 1.0 / self.acquisition_frequency
        self.model = model
        self.nb_dof = self.model.nbQ()
        self.nb_markers = nb_markers
        self.Pp_initial = self._init_covariance()  # Store initial covariance

        # State vector contains [Q]
        self.xp = np.zeros(self.nb_dof)

        # Initialize matrices
        self.A = self._evolution_matrix()
        self.R = self._measurement_noise_matrix()
        self.Pp = self._init_covariance()

    def _evolution_matrix(self) -> np.ndarray:
        """Create evolution matrix (equivalent to biorbd's evolutionMatrix)"""
        return np.eye(self.nb_dof) * self.dt

    def _measurement_noise_matrix(self) -> np.ndarray:
        """Create measurement noise matrix"""
        return np.eye(3 * self.nb_markers) * self.noise_factor

    def _init_covariance(self) -> np.ndarray:
        """Initialize covariance matrix"""
        return np.eye(self.nb_dof) * self.error_factor

    def iteration(self, measure: np.ndarray,
                  projected_measure: np.ndarray,
                  hessian: np.ndarray):
        """
        Perform one iteration of the constrained Kalman filter
        Args:
            measure: Actual marker positions
            projected_measure: Expected marker positions from model
            hessian: Jacobian of marker positions with respect to joint angles
        """
        # Prediction
        xkm = self.A @ self.xp
        Pkm = self.A @ self.Pp @ self.A.T  # + self.Q  TODO: Charbie -> see why Q is detrimental

        # Innovation
        S = hessian @ Pkm @ hessian.T + self.R
        K = Pkm @ hessian.T @ np.linalg.inv(S)

        # Update
        self.xp = xkm + K @ (measure - projected_measure)
        temp = np.eye(self.nb_dof) - K @ hessian
        self.Pp = temp @ Pkm @ temp.T + K @ self.R @ K.T

        # Get state estimates
        q = self.xp

        return q


    def set_initial_states(self, markers):
        """
        Initialize state using the strategy from biorbd:
        1. First estimate root position using only root markers (50 iterations)
        2. Then estimate full body position using all markers (50 iterations)

        Args:
            markers: Marker positions (3 x nb_markers)
        """
        markers_flat = markers.T.flatten()

        # First phase: root estimation
        nb_root_markers = 4
        markers_root = markers_flat.copy()

        # Zero out all non-root markers
        markers_root[3 * nb_root_markers:] = 0

        # Store original Pp
        Pp_original = self.Pp.copy()

        # First phase: estimate root position
        for _ in range(50):
            # Get current state estimate
            q_current = self.xp

            # Create Hessian matrix and projected markers
            H = np.zeros((3 * self.nb_markers, self.nb_dof))
            markers_projected = np.zeros(3 * self.nb_markers)

            # Fill only for root markers
            for i_marker in range(nb_root_markers):
                markers_projected[3 * i_marker:3 * (i_marker + 1)] = self.model.markers(q_current)[
                    i_marker].to_array()
                H[3 * i_marker:3 * (i_marker + 1), :] = self.model.markersJacobian(q_current)[
                    i_marker].to_array()

            # Perform Kalman iteration
            self.iteration(
                measure=markers_root,
                projected_measure=markers_projected,
                hessian=H,
            )

            # Reset Pp and velocities/accelerations
            self.Pp = self.Pp_initial.copy()

        # Second phase: full body estimation
        for _ in range(50):
            # Get current state estimate
            q_current = self.xp

            # Create Hessian matrix and projected markers
            H = np.zeros((3 * self.nb_markers, self.nb_dof))
            markers_projected = np.zeros(3 * self.nb_markers)

            # Fill for all markers
            for i_marker in range(self.nb_markers):
                marker_pos = self.model.markers(q_current)[i_marker].to_array()
                marker_jacobian = self.model.markersJacobian(q_current)[i_marker].to_array()

                H[3 * i_marker:3 * (i_marker + 1), :] = marker_jacobian
                markers_projected[3 * i_marker:3 * (i_marker + 1)] = marker_pos

            # Perform Kalman iteration
            self.iteration(
                measure=markers_flat,
                projected_measure=markers_projected,
                hessian=H,
            )

            # Reset Pp and velocities/accelerations
            self.Pp = self.Pp_initial.copy()

        # Restore original Pp
        self.Pp = Pp_original

        # Return the states
        q = self.xp

        return q


class IterativeKinematicsReconstructor:
    def __init__(self,
                 model: biorbd.Model,
                 nb_frames: int,
                 nb_markers: int,
                 noise_factor: float = 1e-3):
        """
        Initialize the iterative kinematics reconstructor

        Parameters:
        -----------
        model: biorbd.Model
            The biorbd model
        nb_frames: int
            Number of frames in the data
        nb_markers: int
            Number of markers
        noise_factor: float
            Weight for marker noise in the objective function
        """
        self.model = model
        self.nb_frames = nb_frames
        self.nb_markers = nb_markers
        self.nb_dof = model.nbQ()
        self.R = np.eye(3 * nb_markers) * noise_factor

        # Current position estimate
        self.q_current = np.zeros(self.nb_dof)


    def set_joint_constraints(self):
        """Set joint angle constraints from the model"""
        bounds = []
        for i_segment, segment_name in enumerate(segment_dict):
            min_bound = segment_dict[segment_name]["min_bound"]
            max_bound = segment_dict[segment_name]["max_bound"]
            for i_dof in range(len(min_bound)):
                bounds += [(min_bound[i_dof], max_bound[i_dof])]
        return bounds


    def _optimize_root_position(self, markers: np.ndarray,
                                initial_guess: np.ndarray = None,
                                nb_iterations: int = 50) -> np.ndarray:
        """
        Optimize root position using only root markers

        Parameters:
        -----------
        markers: np.ndarray
            Marker positions (3 x nb_markers)
        initial_guess: np.ndarray
            Initial guess for joint angles
        nb_iterations: int
            Number of optimization iterations
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.nb_dof)

        q_estimate = initial_guess.copy()
        nb_root_markers = 4
        nb_root = 6

        # Prepare root-only markers
        markers_root = markers[:, :nb_root_markers].T.flatten()

        for _ in range(nb_iterations):
            # def objective(x):
            #     # Calculate marker positions
            #     model_markers = np.array([marker.to_array()
            #                               for marker in self.model.markers(x)])
            #
            #     # Zero out non-root markers in model markers
            #     model_markers[nb_root_markers:] = 0
            #
            #     # Calculate error
            #     error = (markers_root.T.flatten() - model_markers.flatten())
            #     weighted_error = error.T @ np.linalg.solve(self.R, error)
            #
            #     return weighted_error
            #
            # # Optimize
            # result = minimize(
            #     objective,
            #     q_estimate,
            #     method='SLSQP',
            #     bounds=self.set_joint_constraints(),
            #     options={
            #         'ftol': 1e-8,
            #         'maxiter': 100,
            #         'disp': False
            #     }
            # )

            # Create Hessian matrix and projected markers
            H = np.zeros((3 * nb_root_markers, nb_root))
            markers_projected = np.zeros(3 * nb_root_markers)

            # Fill only for root markers
            for i_marker in range(nb_root_markers):
                markers_projected[3 * i_marker:3 * (i_marker + 1)] = self.model.markers(q_estimate)[i_marker].to_array()
                H[3 * i_marker:3 * (i_marker + 1), :] = self.model.markersJacobian(q_estimate)[i_marker].to_array()[:, :nb_root]

            A = np.eye(nb_root)*1e-5
            S = H @ A @ H.T
            K = A @ H.T @ np.linalg.inv(S)

            # Update
            q_estimate[:6] = A @ q_estimate[:6] + K @ (markers_projected - markers_root.flatten())

        return q_estimate


    def _optimize_full_body(self, markers: np.ndarray,
                            initial_guess: np.ndarray,
                            nb_iterations: int = 50) -> np.ndarray:
        """
        Optimize full body position using all markers

        Parameters:
        -----------
        markers: np.ndarray
            Marker positions (3 x nb_markers)
        initial_guess: np.ndarray
            Initial guess for joint angles (from root optimization)
        nb_iterations: int
            Number of optimization iterations
        """
        q_estimate = initial_guess.copy()

        for _ in range(nb_iterations):
            # def objective(x):
            #     # Calculate marker positions
            #     model_markers = np.array([marker.to_array()
            #                               for marker in self.model.markers(x)])
            #
            #     # Calculate error
            #     error = (markers.T.flatten() - model_markers.flatten())
            #     weighted_error = error.T @ np.linalg.solve(self.R, error)
            #
            #     return weighted_error
            #
            # # Get bounds for optimization
            # bounds = []
            # for joint_name, (min_val, max_val) in self._constraints.items():
            #     bounds.append((min_val, max_val))
            #
            # # Optimize
            # result = minimize(
            #     objective,
            #     q_estimate,
            #     method='SLSQP',
            #     bounds=bounds,
            #     options={
            #         'ftol': 1e-8,
            #         'maxiter': 100,
            #         'disp': False
            #     }
            # )
            #
            # q_estimate = result.x

            # Create Hessian matrix and projected markers
            H = np.zeros((3 * self.nb_markers, self.nb_dof))
            markers_projected = np.zeros(3 * self.nb_markers)

            # Fill only for root markers
            for i_marker in range(self.nb_markers):
                markers_projected[3 * i_marker:3 * (i_marker + 1)] = self.model.markers(q_estimate)[i_marker].to_array()
                H[3 * i_marker:3 * (i_marker + 1), :] = self.model.markersJacobian(q_estimate)[i_marker].to_array()

            A = np.eye(self.nb_dof) * 1e-5
            S = H @ A @ H.T
            K = A @ H.T @ np.linalg.inv(S)

            # Update
            q_estimate = A @ q_estimate + K @ (markers_projected - markers_root)

        return q_estimate


    def reconstruct_frame(self, markers: np.ndarray,
                          initial_guess: np.ndarray = None) -> np.ndarray:
        """
        Reconstruct joint angles for a single frame

        Parameters:
        -----------
        markers: np.ndarray
            Marker positions for current frame (3 x nb_markers)
        initial_guess: np.ndarray
            Initial guess for joint angles (optional)

        Returns:
        --------
        np.ndarray
            Estimated joint angles
        """
        # First optimize root position
        q_root = self._optimize_root_position(markers, initial_guess)

        # Then optimize full body
        q_full = self._optimize_full_body(markers, q_root)

        self.q_current = q_full
        return q_full

    def reconstruct_markers(self, markers: np.ndarray) -> np.ndarray:
        """
        Reconstruct joint angles for all frames

        Parameters:
        -----------
        markers: np.ndarray
            All marker positions (3 x nb_markers x nb_frames)

        Returns:
        --------
        np.ndarray
            Estimated joint angles for all frames (nb_dof x nb_frames)
        """
        q_estimates = np.zeros((self.nb_dof, self.nb_frames))

        for i_frame in range(self.nb_frames):
            # Get initial guess from previous frame if available
            initial_guess = q_estimates[:, i_frame - 1] if i_frame > 0 else None

            # Reconstruct frame
            current_markers = markers[:, :, i_frame]
            q_estimates[:, i_frame] = self.reconstruct_frame(
                current_markers, initial_guess)

            if i_frame % 10 == 0:
                print(f"Processed frame {i_frame}/{self.nb_frames}")

        return q_estimates