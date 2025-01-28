import numpy as np

import biorbd
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    ObjectiveList,
    ObjectiveFcn,
    InterpolationType,
    Node,
    Solver,
)

from gait_analyzer.experimental_data import ExperimentalData


class OptimalEstimator:
    """
    This class create an optimal control problem and solve it.
    The goal is to match as closely as possible the experimental data.
    This method allows estimating the muscle forces and the joint torques more accurately.
    However, it is quite long.
    """
    def __init__(self,
                 biorbd_model_path: str,
                 experimental_data: ExperimentalData,
                 q: np.ndarray,
                 qdot: np.ndarray,
                 phases: dict):
        """
        Initialize the OptimalEstimator.
        .
        Parameters
        ----------
        biorbd_model_path: str
            The full path to the biorbd model.
        """
        # Checks

        # Initial attributes
        self.biorbd_model_path = biorbd_model_path
        self.experimental_data = experimental_data
        self.q = q
        self.qdot = qdot
        self.phases = phases

        # Extended attributes
        self.ocp = None
        self.solution = None
        self.tau = None
        self.muscle_forces = None

        # Perform the optimal estimation
        self.model_ocp = None
        self.q_exp_ocp = None
        self.qdot_exp_ocp = None
        self.f_ext_exp_ocp = None
        self.markers_exp_ocp = None
        self.emg_exp_ocp = None
        self.n_shooting = None
        self.phase_time = None
        self.prepare_reduced_experimental_data()
        self.prepare_ocp()
        self.solve()
        self.extract_muscle_forces()


    def prepare_reduced_experimental_data(self):
        """
        To reduce the optimization time, only one cycle is treated at a time
        (and the number of degrees of freedom is reduced?).
        """
        # Temporarily I will try with everything!
        self.model_ocp = self.biorbd_model_path.replace(".bioMod", "_heelL_toesL.bioMod")
        model_new = biorbd.Model(self.biorbd_model_path)
        model_new.

        # Only the 10th right leg swing (while left leg in flat foot)
        swing_timings = np.where(self.phases["heelL_toesL"])[0]
        right_swing_sequence = np.array_split(
            swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1
        )
        this_sequence_analogs = right_swing_sequence[9]
        this_sequence_markers = np.arange(int(this_sequence_analogs[0] * self.experimental_data.analogs_dt/self.experimental_data.markers_dt),
                                         int(this_sequence_analogs[-1] * self.experimental_data.analogs_dt/self.experimental_data.markers_dt))
        self.q_exp_ocp = self.q[:, this_sequence_markers]
        self.qdot_exp_ocp = self.qdot[:, this_sequence_markers]
        self.f_ext_exp_ocp = {"left_leg": np.zeros((6, self.q_exp_ocp.shape[1])),
                              "right_leg": np.zeros((6, self.q_exp_ocp.shape[1]))}
        for i_node in range(self.q_exp_ocp.shape[1]):
            # TODO: Guys/Charbie -> This method allows for rounding errors, there should be a mapping of frames instead
            idx_beginning = int(this_sequence_markers[i_node] * self.experimental_data.markers_dt/self.experimental_data.analogs_dt) - 3
            idx_end = int(this_sequence_markers[i_node] * self.experimental_data.markers_dt/self.experimental_data.analogs_dt) + 3
            self.f_ext_exp_ocp["left_leg"][:3, i_node] = np.mean(self.experimental_data.grf_sorted[0, 3:6, idx_beginning:idx_end], axis=1)  # Moments
            self.f_ext_exp_ocp["left_leg"][3:, i_node] = np.mean(self.experimental_data.grf_sorted[0, 0:3, idx_beginning:idx_end], axis=1)  # Forces
            self.f_ext_exp_ocp["right_leg"][:3, i_node] = np.mean(self.experimental_data.grf_sorted[1, 3:6, idx_beginning:idx_end], axis=1)  # Moments
            self.f_ext_exp_ocp["right_leg"][3:, i_node] = np.mean(self.experimental_data.grf_sorted[1, 0:3, idx_beginning:idx_end], axis=1)  # Forces
        self.markers_exp_ocp = self.experimental_data.markers_sorted[:, this_sequence_markers]

        self.n_shooting = self.q_exp_ocp.shape[1]
        self.phase_time = self.n_shooting * self.experimental_data.markers_dt

    def prepare_ocp(self):
        """
        Let's say swing phase only for now
        """

        bio_model = BiorbdModel(self.biorbd_model_path)

        # Declaration of the objectives
        objective_functions = ObjectiveList()
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau_joints",
            weight=1.0,
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_MARKERS,
            weight=10.0,
            target=self.markers_exp_ocp
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_STATE,
            key="q",
            weight=0.1,
            target=self.q_exp_ocp
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_STATE,
            key="qdot",
            weight=0.01,
            target=self.qdot_exp_ocp
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_CONTACT_FORCES,
            weight=0.01,
            target=self.f_ext_exp_ocp,
        )

        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

        # Declaration of optimization variables bounds and initial guesses
        # Path constraint
        x_bounds = BoundsList()
        x_initial_guesses = InitialGuessList()

        u_bounds = BoundsList()
        u_initial_guesses = InitialGuessList()

        x_bounds.add(
            "q_roots",
            min_bound=[
                [0.0, -1.0, -0.1],
                [0.0, -1.0, -0.1],
                [0.0, -0.1, -0.1],
                [0.0, -0.1, 2 * np.pi - 0.1],
                [0.0, -np.pi / 4, -np.pi / 4],
                [0.0, -0.1, -0.1],
            ],
            max_bound=[
                [0.0, 1.0, 0.1],
                [0.0, 1.0, 0.1],
                [0.0, 10.0, 0.1],
                [0.0, 2 * np.pi + 0.1, 2 * np.pi + 0.1],
                [0.0, np.pi / 4, np.pi / 4],
                [0.0, np.pi + 0.1, np.pi + 0.1],
            ],
        )
        x_bounds.add(
            "q_joints",
            min_bound=[
                [2.9, -0.05, -0.05],
                [-2.9, -3.0, -3.0],
            ],
            max_bound=[
                [2.9, 3.0, 3.0],
                [-2.9, 0.05, 0.05],
            ],
        )

        x_bounds.add(
            "qdot_roots",
            min_bound=[
                [-0.5, -10.0, -10.0],
                [-0.5, -10.0, -10.0],
                [5.0, -100.0, -100.0],
                [0.5, 0.5, 0.5],
                [0.0, -100.0, -100.0],
                [0.0, -100.0, -100.0],
            ],
            max_bound=[
                [0.5, 10.0, 10.0],
                [0.5, 10.0, 10.0],
                [10.0, 100.0, 100.0],
                [20.0, 20.0, 20.0],
                [-0.0, 100.0, 100.0],
                [-0.0, 100.0, 100.0],
            ],
        )
        x_bounds.add(
            "qdot_joints",
            min_bound=[
                [0.0, -100.0, -100.0],
                [0.0, -100.0, -100.0],
            ],
            max_bound=[
                [-0.0, 100.0, 100.0],
                [-0.0, 100.0, 100.0],
            ],
        )

        x_initial_guesses.add(
            "q_roots",
            initial_guess=[
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 2 * np.pi],
                [0.0, 0.0],
                [0.0, np.pi],
            ],
            interpolation=InterpolationType.LINEAR,
        )
        x_initial_guesses.add(
            "q_joints",
            initial_guess=[
                [2.9, 0.0],
                [-2.9, 0.0],
            ],
            interpolation=InterpolationType.LINEAR,
        )

        x_initial_guesses.add(
            "qdot_roots",
            initial_guess=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            interpolation=InterpolationType.CONSTANT,
        )
        x_initial_guesses.add(
            "qdot_joints",
            initial_guess=[0.0, 0.0],
            interpolation=InterpolationType.CONSTANT,
        )

        u_bounds.add("tau_joints", min_bound=[-100, -100], max_bound=[100, 100], interpolation=InterpolationType.CONSTANT)

        self.ocp = OptimalControlProgram(
            bio_model=bio_model,
            n_shooting=self.n_shooting,
            phase_time=self.phase_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_initial_guesses,
            u_init=u_initial_guesses,
            objective_functions=objective_functions,
            use_sx=False,
        )

    def solve(self, show_online_optim: bool = False):
        solver = Solver.IPOPT(show_online_optim=show_online_optim)
        self.solution = self.ocp.solve(solver=solver)


    def extract_muscle_forces(self):
        # TODO: Charbie -> Extract muscle forces from the solution
        self.muscle_forces = None