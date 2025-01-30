import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from gait_analyzer.operator import Operator
from gait_analyzer.experimental_data import ExperimentalData


class Events:
    """
    This class contains all the events detected from the experimental data.
    """

    def __init__(self, experimental_data: ExperimentalData, plot_phases_flag: bool = False):
        """
        Initialize the Events.
        .
        Parameters
        ----------
        experimental_data: ExperimentalData
            The experimental data from the trial
        """
        # Checks
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError(
                "experimental_data must be an instance of ExperimentalData. You can declare it by running ExperimentalData(file_path)."
            )

        # Initial attributes
        self.experimental_data = experimental_data

        # Extended attributes
        self.events = {
            "right_leg_heel_touch": [],  # heel strike
            "right_leg_toes_touch": [],  # beginning of flat foot
            "right_leg_heel_off": [],  # end of flat foot
            "right_leg_toes_off": [],  # beginning of swing
            "left_leg_heel_touch": [],  # heel strike
            "left_leg_toes_touch": [],  # beginning of flat foot
            "left_leg_heel_off": [],  # end of flat foot
            "left_leg_toes_off": [],  # beginning of swing
        }
        nb_analog_frames = self.experimental_data.nb_analog_frames
        self.phases_right_leg = {
            "flat_foot": np.zeros((nb_analog_frames,)),
            "toes_only": np.zeros((nb_analog_frames,)),
            "swing": np.zeros((nb_analog_frames,)),
            "heel_only": np.zeros((nb_analog_frames,)),
        }
        self.phases_left_leg = {
            "flat_foot": np.zeros((nb_analog_frames,)),
            "toes_only": np.zeros((nb_analog_frames,)),
            "swing": np.zeros((nb_analog_frames,)),
            "heel_only": np.zeros((nb_analog_frames,)),
        }
        self.phases = {
            "heelR_toesR": np.zeros((nb_analog_frames,)),
            "toesR": np.zeros((nb_analog_frames,)),
            "toesR_heelL": np.zeros((nb_analog_frames,)),
            "toesR_heelL_toesL": np.zeros((nb_analog_frames,)),
            "heelL_toesL": np.zeros((nb_analog_frames,)),
            "toesL": np.zeros((nb_analog_frames,)),
            "toesL_heelR": np.zeros((nb_analog_frames,)),
            "toesL_heelR_toesR": np.zeros((nb_analog_frames,)),
        }
        self.find_event_timestamps()
        if plot_phases_flag:
            self.plot_events()

    def detect_heel_touch(self):
        """
        Detect the heel touch event when the antero-posterior GRF reaches a certain threshold after the swing phase
        """
        maximal_forward_force_threshold = 0.005
        grf_left_y_filtered = Operator.moving_average(self.experimental_data.grf_sorted[0, 1, :], 21)
        grf_right_y_filtered = Operator.moving_average(self.experimental_data.grf_sorted[1, 1, :], 21)

        # Left
        swing_timings = np.where(self.phases_left_leg["swing"])[0]
        left_swing_sequence = np.array_split(swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1)
        for i_swing, swing_phase in enumerate(left_swing_sequence):
            idx = swing_phase[-1] - 5
            while (
                idx < self.experimental_data.nb_analog_frames - 1
                and np.abs(grf_left_y_filtered[idx]) < maximal_forward_force_threshold
            ):
                idx += 1
            if idx <= self.experimental_data.nb_analog_frames - 1:
                idx -= 1
                self.events["left_leg_heel_touch"] += [int(((swing_phase[-1] + idx) / 2))]

        # Right
        swing_timings = np.where(self.phases_right_leg["swing"])[0]
        right_swing_sequence = np.array_split(swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1)
        for i_swing, swing_phase in enumerate(right_swing_sequence):
            idx = swing_phase[-1] - 5
            while (
                idx < self.experimental_data.nb_analog_frames - 1
                and np.abs(grf_right_y_filtered[idx]) < maximal_forward_force_threshold
            ):
                idx += 1
            if idx <= self.experimental_data.nb_analog_frames - 1:
                idx -= 1
                self.events["right_leg_heel_touch"] += [int(((swing_phase[-1] + idx) / 2))]

    def detect_toes_touch(self):
        """
        Detect the toes touch event when the vertical GRF is maximal
        """
        grf_left_z_filtered = Operator.moving_average(self.experimental_data.grf_sorted[0, 2, :], 35)
        grf_right_z_filtered = Operator.moving_average(self.experimental_data.grf_sorted[1, 2, :], 35)

        swing_timings = np.where(self.phases_left_leg["swing"])[0]
        left_swing_sequence = np.array_split(swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1)
        for i_swing, swing_phase in enumerate(left_swing_sequence):
            end_swing_idx = swing_phase[-1]
            if end_swing_idx == self.experimental_data.nb_analog_frames - 1:
                continue
            if i_swing == len(left_swing_sequence) - 1:
                beginning_next_swing_idx = self.experimental_data.nb_analog_frames - 1
            else:
                beginning_next_swing_idx = left_swing_sequence[i_swing + 1][0]
            mid_stance = int((end_swing_idx + beginning_next_swing_idx) / 2)
            partial_idx_first_peak_z = np.argmax(np.abs(grf_left_z_filtered[end_swing_idx:mid_stance]))
            self.events["left_leg_toes_touch"] += [int(end_swing_idx + partial_idx_first_peak_z)]

        # Right
        swing_timings = np.where(self.phases_right_leg["swing"])[0]
        right_swing_sequence = np.array_split(swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1)
        for i_swing, swing_phase in enumerate(right_swing_sequence):
            end_swing_idx = swing_phase[-1]
            if end_swing_idx == self.experimental_data.nb_analog_frames - 1:
                continue
            if i_swing == len(right_swing_sequence) - 1:
                beginning_next_swing_idx = self.experimental_data.nb_analog_frames - 1
            else:
                beginning_next_swing_idx = right_swing_sequence[i_swing + 1][0]
            mid_stance = int((end_swing_idx + beginning_next_swing_idx) / 2)
            partial_idx_first_peak_z = np.argmax(np.abs(grf_right_z_filtered[end_swing_idx:mid_stance]))
            self.events["right_leg_toes_touch"] += [int(end_swing_idx + partial_idx_first_peak_z)]

    def detect_heel_off(self):
        """
        Detect hell off events when the heel marker moves faster than 0.1 m/s
        """
        # Left
        left_cal_velocity = (
            np.diff(
                self.experimental_data.markers_sorted[2, self.experimental_data.model_marker_names.index("LCAL"), :]
            )
            / self.experimental_data.markers_dt
        )
        # left_cal_velocity = np.abs(left_cal_velocity)
        swing_timings = np.where(self.phases_left_leg["swing"])[0]
        left_swing_sequence = np.array_split(swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1)
        # TODO: Flo -> Visiblement, ces données sont filtrées. Est-ce que je pourrais avoir les raw data avec les bons marqueurs svp ?
        for i_swing, swing_phase in enumerate(left_swing_sequence):
            mid_swing_idx = int((swing_phase[-1] + swing_phase[0]) / 2)
            left_heel_moving = np.where(
                left_cal_velocity[
                    int(mid_swing_idx * self.experimental_data.analogs_dt / self.experimental_data.markers_dt) :
                ]
                > 0.1
            )[0]
            # if left_heel_moving.shape == (0,):
            # plt.figure()
            # plt.plot((self.experimental_data.marker_time_vector[1:] + self.experimental_data.marker_time_vector[:-1]) / 2, left_cal_velocity, label="Left heel velocity")
            # plt.plot(self.experimental_data.marker_time_vector, self.experimental_data.markers_sorted[2, self.experimental_data.model_marker_names.index("LCAL"), :], label="Left heel height")
            # plt.plot(self.experimental_data.analogs_time_vector, self.experimental_data.grf_sorted[0, 2, :], label="Vertical Left GRF")
            # plt.plot(np.array([self.experimental_data.analogs_time_vector[0], self.experimental_data.analogs_time_vector[-1]]), np.array([0.1, 0.1]), '--k', label="Velocity  threshold")
            # plt.legend()
            # plt.show()
            # raise RuntimeError("The left heel marker (LCAL) is not moving, please double check the data.")
            left_heel_moving = left_heel_moving[0] + mid_swing_idx
            self.events["left_leg_heel_off"] += [
                int(left_heel_moving * self.experimental_data.markers_dt / self.experimental_data.analogs_dt)
            ]
        # Right
        right_cal_velocity = (
            np.diff(
                self.experimental_data.markers_sorted[2, self.experimental_data.model_marker_names.index("RCAL"), :]
            )
            / self.experimental_data.markers_dt
        )
        right_heel_moving = np.where(right_cal_velocity > 0.1)[0]
        right_heel_moving_sequence = np.array_split(
            right_heel_moving, np.flatnonzero(np.diff(right_heel_moving) > 1) + 1
        )
        for sequence in right_heel_moving_sequence:
            self.events["right_leg_heel_off"] += [
                int(sequence[0] * self.experimental_data.markers_dt / self.experimental_data.analogs_dt)
            ]

    def detect_toes_off(self):
        """
        Detect the toes off event when the vertical GRF is lower than a threshold
        """
        # Left
        swing_timings = np.where(self.phases_left_leg["swing"])[0]
        left_swing_sequence = np.array_split(swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1)
        for i_swing, swing_phase in enumerate(left_swing_sequence):
            beginning_swing_idx = swing_phase[0]
            if beginning_swing_idx == self.experimental_data.nb_analog_frames - 1 or beginning_swing_idx == 0:
                continue
            self.events["left_leg_toes_off"] += [int(beginning_swing_idx)]

        # Right
        swing_timings = np.where(self.phases_right_leg["swing"])[0]
        right_swing_sequence = np.array_split(swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1)
        for i_swing, swing_phase in enumerate(right_swing_sequence):
            beginning_swing_idx = swing_phase[0]
            if beginning_swing_idx == self.experimental_data.nb_analog_frames - 1 or beginning_swing_idx == 0:
                continue
            self.events["right_leg_toes_off"] += [int(beginning_swing_idx)]

    def detect_swing_phases_temporary(self):
        """
        Detect the swing phase when the vertical GRF is lower than a threshold
        """
        minimal_vertical_force_threshold = 0.05
        grf_right_z_filtered = Operator.moving_average(self.experimental_data.grf_sorted[0, 2, :], 21)
        grf_left_z_filtered = Operator.moving_average(self.experimental_data.grf_sorted[1, 2, :], 21)
        self.phases_left_leg["swing"][:] = np.abs(grf_right_z_filtered) < minimal_vertical_force_threshold
        self.phases_right_leg["swing"][:] = np.abs(grf_left_z_filtered) < minimal_vertical_force_threshold
        return

    def detect_leg_phases_between_events(self, phase_name, init_event_name, closing_event_name):
        # Left
        for init_idx in self.events["left_leg_" + init_event_name]:
            list_index_idx = np.where(np.array(self.events["left_leg_" + closing_event_name]) - init_idx > 0)[0]
            if list_index_idx.shape != (0,):
                list_index_idx = list_index_idx[0]
                next_closing_idx = self.events["left_leg_" + closing_event_name][list_index_idx]
                self.phases_left_leg[phase_name][init_idx : next_closing_idx + 1] = 1

        # Right
        for init_idx in self.events["right_leg_" + init_event_name]:
            list_index_idx = np.where(np.array(self.events["right_leg_" + closing_event_name]) - init_idx > 0)[0]
            if list_index_idx.shape != (0,):
                list_index_idx = list_index_idx[0]
                next_closing_idx = self.events["right_leg_" + closing_event_name][list_index_idx]
                self.phases_right_leg[phase_name][init_idx : next_closing_idx + 1] = 1
        return

    def detect_phases_both_legs(self, phase_name, left_leg_phase_name, right_leg_phase_name):
        self.phases[phase_name] = np.logical_and(
            self.phases_left_leg[left_leg_phase_name], self.phases_right_leg[right_leg_phase_name]
        )
        return

    def find_event_timestamps(self):
        # Detect when each foot is in the air as a preliminary step for the detection of events
        # Please not that anything happening before the first temporary swing phase is not considered
        # TODO: Charbie -> Add references to the articles where these methods are described
        # TODO: Charbie -> Add an alternative AI detection method

        self.detect_swing_phases_temporary()

        # Detect events
        self.detect_toes_off()
        self.detect_heel_off()
        self.detect_heel_touch()
        self.detect_toes_touch()

        # Detect phases for each leg
        self.phases_left_leg["swing"] = np.zeros_like(self.phases_left_leg["swing"])
        self.phases_right_leg["swing"] = np.zeros_like(self.phases_right_leg["swing"])
        self.detect_leg_phases_between_events("swing", "toes_off", "heel_touch")
        self.detect_leg_phases_between_events("heel_only", "heel_touch", "toes_touch")
        self.detect_leg_phases_between_events("flat_foot", "toes_touch", "heel_off")
        self.detect_leg_phases_between_events("toes_only", "heel_off", "toes_off")

        # Detect combined phases (both legs)
        self.detect_phases_both_legs("heelR_toesR", "swing", "flat_foot")
        self.detect_phases_both_legs("toesR", "swing", "toes_only")
        self.detect_phases_both_legs("toesR_heelL", "heel_only", "toes_only")
        self.detect_phases_both_legs("toesR_heelL_toesL", "flat_foot", "toes_only")
        self.detect_phases_both_legs("heelL_toesL", "flat_foot", "swing")
        self.detect_phases_both_legs("toesL", "toes_only", "swing")
        self.detect_phases_both_legs("toesL_heelR", "toes_only", "heel_only")
        self.detect_phases_both_legs("toesL_heelR_toesR", "toes_only", "flat_foot")

    def plot_events(self):
        """
        Plot the GRF and the detected phases
        """
        fig, axs = plt.subplots(3, 1, figsize=(15, 7))

        axs[0].plot(
            self.experimental_data.analogs_time_vector,
            self.experimental_data.grf_sorted[0, 0, :],
            "-r",
            label="Medio-lateral",
        )
        axs[0].plot(
            self.experimental_data.analogs_time_vector,
            self.experimental_data.grf_sorted[0, 1, :],
            "-g",
            label="Antero-posterior",
        )
        axs[0].plot(
            self.experimental_data.analogs_time_vector,
            self.experimental_data.grf_sorted[0, 2, :],
            "-b",
            label="Vertical",
        )

        axs[1].plot(
            self.experimental_data.analogs_time_vector,
            self.experimental_data.grf_sorted[1, 0, :],
            "-r",
            label="Medio-lateral",
        )
        axs[1].plot(
            self.experimental_data.analogs_time_vector,
            self.experimental_data.grf_sorted[1, 1, :],
            "-g",
            label="Antero-posterior",
        )
        axs[1].plot(
            self.experimental_data.analogs_time_vector,
            self.experimental_data.grf_sorted[1, 2, :],
            "-b",
            label="Vertical",
        )

        color = colormaps["magma"]
        for i_phase, key in enumerate(self.phases_left_leg):
            # Left
            axs[0].plot(
                self.experimental_data.analogs_time_vector,
                (self.phases_left_leg[key] * 0.1 * (i_phase + 1)) + 0.3,
                ".",
                color=color(i_phase / 4),
            )
            start_idx = 0
            is_label = False
            for idx in range(1, self.phases_left_leg[key].shape[0]):
                if self.phases_left_leg[key][idx]:
                    if not self.phases_left_leg[key][idx - 1]:
                        start_idx = idx
                if not self.phases_left_leg[key][idx]:
                    if self.phases_left_leg[key][idx - 1]:
                        end_idx = idx
                        if not is_label:
                            axs[0].axvspan(
                                self.experimental_data.analogs_time_vector[start_idx],
                                self.experimental_data.analogs_time_vector[end_idx],
                                alpha=0.2,
                                color=color(i_phase / 4),
                                label=key,
                            )
                            is_label = True
                        else:
                            axs[0].axvspan(
                                self.experimental_data.analogs_time_vector[start_idx],
                                self.experimental_data.analogs_time_vector[end_idx],
                                alpha=0.2,
                                color=color(i_phase / 4),
                            )
            # Right
            axs[1].plot(
                self.experimental_data.analogs_time_vector,
                (self.phases_left_leg[key] * 0.1 * (i_phase + 1)) + 0.3,
                ".",
                color=color(i_phase / 4),
            )
            start_idx = 0
            is_label = False
            for idx in range(1, self.phases_right_leg[key].shape[0]):
                if self.phases_right_leg[key][idx]:
                    if not self.phases_right_leg[key][idx - 1]:
                        start_idx = idx
                if not self.phases_right_leg[key][idx]:
                    if self.phases_right_leg[key][idx - 1]:
                        end_idx = idx
                        if not is_label:
                            axs[1].axvspan(
                                self.experimental_data.analogs_time_vector[start_idx],
                                self.experimental_data.analogs_time_vector[end_idx],
                                alpha=0.2,
                                color=color(i_phase / 4),
                                label=key,
                            )
                            is_label = True
                        else:
                            axs[1].axvspan(
                                self.experimental_data.analogs_time_vector[start_idx],
                                self.experimental_data.analogs_time_vector[end_idx],
                                alpha=0.2,
                                color=color(i_phase / 4),
                            )

        # Both legs
        for i_phase, key in enumerate(self.phases):
            axs[2].plot(
                self.experimental_data.analogs_time_vector,
                self.phases[key] * 0.1 * (i_phase + 1),
                ".",
                color=color(i_phase / 4),
            )
            start_idx = 0
            is_label = False
            for idx in range(1, self.phases[key].shape[0]):
                if self.phases[key][idx]:
                    if not self.phases[key][idx - 1]:
                        start_idx = idx
                if not self.phases[key][idx]:
                    if self.phases[key][idx - 1]:
                        end_idx = idx
                        if not is_label:
                            axs[2].axvspan(
                                self.experimental_data.analogs_time_vector[start_idx],
                                self.experimental_data.analogs_time_vector[end_idx],
                                alpha=0.2,
                                color=color(i_phase / 8),
                                label=key,
                            )
                            is_label = True
                        else:
                            axs[2].axvspan(
                                self.experimental_data.analogs_time_vector[start_idx],
                                self.experimental_data.analogs_time_vector[end_idx],
                                alpha=0.2,
                                color=color(i_phase / 8),
                            )

        axs[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        axs[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        axs[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        axs[0].set_ylabel("Left leg GRF")
        axs[1].set_ylabel("Right leg GRF")
        axs[2].set_ylabel("Phases both legs")
        plt.savefig("GRF.png")
        plt.show()

    def inputs(self):
        return {
            "experimental_data": self.experimental_data,
        }

    def outputs(self):
        return {
            "events": self.events,
            "phases_left_leg": self.phases_left_leg,
            "phases_right_leg": self.phases_right_leg,
            "phases": self.phases,
        }
