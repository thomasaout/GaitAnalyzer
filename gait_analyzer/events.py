import ezc3d
import biorbd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


class Events:
    def __init__(self, file_path: str, biorbd_model_path:str):
        self.file_path = file_path
        self.biorbd_model_path = biorbd_model_path
        self.initial_treatment()

        # Class Events
        self.events = {"right_leg_heel_touch": [],  # heel strike
                        "right_leg_toes_touch": [],  # beginning of flat foot
                        "right_leg_heel_off": [],  # end of flat foot
                        "right_leg_toes_off": [],  # beginning of swing
                        "left_leg_heel_touch": [],  # heel strike
                        "left_leg_toes_touch": [],  # beginning of flat foot
                        "left_leg_heel_off": [],  # end of flat foot
                        "left_leg_toes_off": [],  # beginning of swing
                       }
        self.phases_right_leg = {"flat_foot": np.zeros((self.nb_analog_frames, )),
                          "toes_only": np.zeros((self.nb_analog_frames, )),
                          "swing": np.zeros((self.nb_analog_frames, )),
                          "heel_only": np.zeros((self.nb_analog_frames, ))}
        self.phases_left_leg = {"flat_foot": np.zeros((self.nb_analog_frames, )),
                              "toes_only": np.zeros((self.nb_analog_frames, )),
                              "swing": np.zeros((self.nb_analog_frames, )),
                              "heel_only": np.zeros((self.nb_analog_frames, ))}
        self.phases = {"heelR_toesR": np.zeros((self.nb_analog_frames, )),
                       "toesR": np.zeros((self.nb_analog_frames, )),
                       "toesR_heelL": np.zeros((self.nb_analog_frames, )),
                       "toesR_heelL_toesL": np.zeros((self.nb_analog_frames, )),
                       "heelL_toesL": np.zeros((self.nb_analog_frames,)),
                       "toesL": np.zeros((self.nb_analog_frames,)),
                       "toesL_heelR": np.zeros((self.nb_analog_frames,)),
                       "toesL_heelR_toesR": np.zeros((self.nb_analog_frames,)),
                       }


    def initial_treatment(self):
        """
        Extract important information and sort the data
        """
        # Load model
        model = biorbd.Model(self.biorbd_model_path)
        self.model_marker_names = [m.to_string() for m in model.markerNames()]
        # model_muscle_names = [m.to_string() for m in model.muscleNames()]

        # Get an array of the position of the experimental self.markers
        c3d = ezc3d.c3d(self.file_path)
        markers = c3d["data"]["points"]
        self.marker_sampling_frequency = c3d["parameters"]["POINT"]["RATE"]["value"][0]  # Hz
        self.markers_dt = 1 / c3d["header"]["points"]["frame_rate"]
        self.nb_marker_frames = markers.shape[2]
        exp_marker_names = c3d["parameters"]["POINT"]["LABELS"]["value"]
        markers_sorted = np.zeros((3, len(self.model_marker_names), self.nb_marker_frames))
        for i_marker, name in enumerate(self.model_marker_names):
            marker_idx = exp_marker_names.index(name)
            markers_sorted[:, marker_idx, :] = markers[:3, marker_idx, :]
        self.markers_sorted = markers_sorted
        self.right_leg_grf = np.vstack((markers[:3, exp_marker_names.index("moment2"), :],
                                   markers[:3, exp_marker_names.index("force2"), :]))
        self.left_leg_grf = np.vstack((markers[:3, exp_marker_names.index("moment1"), :],
                                  markers[:3, exp_marker_names.index("force1"), :]))

        # Get an array of the experimental muscle activity
        analogs = c3d["data"]["analogs"]
        self.nb_analog_frames = analogs.shape[2]
        self.analogs_sampling_frequency = c3d["parameters"]["ANALOG"]["RATE"]["value"][0]  # Hz
        self.analogs_dt = 1 / c3d["header"]["analogs"]["frame_rate"]
        analog_names = c3d["parameters"]["ANALOG"]["LABELS"]["value"]
        # print(analog_names)
        # emg_sorted = np.zeros((len(model_muscle_names), self.nb_analog_frames))
        # for i_muscle, name in enumerate(model_muscle_names):
        #     muscle_idx = analog_names.index(name)
        #     emg_sorted[i_muscle, :] = analogs[muscle_idx, :]
        # self.emg_sorted = emg_sorted
        # # TODO: Charbie -> treatment of the EMG signal to remove stimulation artifacts

        # Get the experimental ground reaction forces
        force_platform_1_channels = c3d["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"][:, 0]
        force_platform_2_channels = c3d["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"][:, 1]
        grf_sorted = np.zeros((2, 6, self.nb_analog_frames))
        for i in range(6):
            platform_1_idx = analog_names.index(f"Channel_{force_platform_1_channels[i]:02d}")
            platform_2_idx = analog_names.index(f"Channel_{force_platform_2_channels[i]:02d}")
            grf_sorted[0, i, :] = analogs[0, platform_1_idx, :]
            grf_sorted[1, i, :] = analogs[0, platform_2_idx, :]
        self.grf_sorted = grf_sorted

        # from scipy import signal
        # b, a = signal.butter(2, 1/50, btype='low')
        # y = signal.filtfilt(b, a, grf_sorted[0, 2, :], padlen=150)
        # # 4th 6-10

        self.marker_time_vector = np.linspace(0, self.markers_dt * self.nb_marker_frames, self.nb_marker_frames)
        self.analogs_time_vector = np.linspace(0, self.analogs_dt * self.nb_analog_frames, self.nb_analog_frames)
        # plt.figure()
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 2, :], '-r')
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 1, :], '-g')
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 2, :], '.b')
        # plt.plot(self.analogs_time_vector, y, '-b')
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 3, :], '-m')
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 4, :], '-c')
        # plt.plot(self.analogs_time_vector, grf_sorted[0, 5, :], '-k')
        # cal_velovity = np.diff(markers_sorted[2, self.model_marker_names.index("LCAL"), :]) / self.markers_dt
        # time_velocity = (self.marker_time_vector[1:] + self.marker_time_vector[:-1]) / 2
        # plt.plot(self.marker_time_vector, markers_sorted[2, self.model_marker_names.index("LCAL"), :], '-b')
        # plt.plot(time_velocity, cal_velovity, '-k')
        # plt.xlim(0, 1.4)
        # plt.show()


    def detect_heel_touch(self):
        """
        Detect the heel touch event when the antero-posterior GRF reaches a certain threshold after the swing phase
        """
        maximal_forward_force_threshold = 0.005
        grf_left_y_filtered = moving_average(self.grf_sorted[0, 1, :], 21)
        grf_right_y_filtered = moving_average(self.grf_sorted[1, 1, :], 21)

        # Left
        swing_timings = np.where(self.phases_left_leg["swing"])[0]
        left_swing_sequence = np.array_split(
            swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1
        )
        for i_swing, swing_phase in enumerate(left_swing_sequence):
            idx = swing_phase[-1] - 5
            while idx < self.nb_analog_frames - 1 and np.abs(grf_left_y_filtered[idx]) < maximal_forward_force_threshold:
                idx += 1
            if idx <= self.nb_analog_frames - 1:
                idx -= 1
                self.events["left_leg_heel_touch"] += [int(((swing_phase[-1] + idx) / 2))]

        # Right
        swing_timings = np.where(self.phases_right_leg["swing"])[0]
        right_swing_sequence = np.array_split(
            swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1
        )
        for i_swing, swing_phase in enumerate(right_swing_sequence):
            idx = swing_phase[-1] - 5
            while idx < self.nb_analog_frames - 1 and np.abs(grf_right_y_filtered[idx]) < maximal_forward_force_threshold:
                idx += 1
            if idx <= self.nb_analog_frames - 1:
                idx -= 1
                self.events["right_leg_heel_touch"] += [int(((swing_phase[-1] + idx) / 2))]


    def detect_toes_touch(self):
        """
        Detect the toes touch event when the vertical GRF is maximal
        """
        grf_left_z_filtered = moving_average(self.grf_sorted[0, 2, :], 35)
        grf_right_z_filtered = moving_average(self.grf_sorted[1, 2, :], 35)

        swing_timings = np.where(self.phases_left_leg["swing"])[0]
        left_swing_sequence = np.array_split(
            swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1
        )
        for i_swing, swing_phase in enumerate(left_swing_sequence):
            end_swing_idx = swing_phase[-1]
            if end_swing_idx == self.nb_analog_frames - 1:
                continue
            if i_swing == len(left_swing_sequence) - 1:
                beginning_next_swing_idx = self.nb_analog_frames - 1
            else:
                beginning_next_swing_idx = left_swing_sequence[i_swing+1][0]
            mid_stance = int((end_swing_idx + beginning_next_swing_idx) / 2)
            partial_idx_first_peak_z = np.argmax(np.abs(grf_left_z_filtered[end_swing_idx:mid_stance]))
            self.events["left_leg_toes_touch"] += [int(end_swing_idx + partial_idx_first_peak_z)]

        # Right
        swing_timings = np.where(self.phases_right_leg["swing"])[0]
        right_swing_sequence = np.array_split(
            swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1
        )
        for i_swing, swing_phase in enumerate(right_swing_sequence):
            end_swing_idx = swing_phase[-1]
            if end_swing_idx == self.nb_analog_frames - 1:
                continue
            if i_swing == len(right_swing_sequence) - 1:
                beginning_next_swing_idx = self.nb_analog_frames - 1
            else:
                beginning_next_swing_idx = right_swing_sequence[i_swing+1][0]
            mid_stance = int((end_swing_idx + beginning_next_swing_idx) / 2)
            partial_idx_first_peak_z = np.argmax(np.abs(grf_right_z_filtered[end_swing_idx:mid_stance]))
            self.events["right_leg_toes_touch"] += [int(end_swing_idx + partial_idx_first_peak_z)]


    def detect_heel_off(self):
        """
        Detect hell off events when the heel marker moves faster than 0.1 m/s
        """
        # Left
        left_cal_velocity = np.diff(self.markers_sorted[2, self.model_marker_names.index("LCAL"), :]) / self.markers_dt
        left_heel_moving = np.where(left_cal_velocity > 0.1)[0]
        left_heel_moving_sequence = np.array_split(
            left_heel_moving, np.flatnonzero(np.diff(left_heel_moving) > 1) + 1
        )
        for sequence in left_heel_moving_sequence:
            self.events["left_leg_heel_off"] += [int(sequence[0]*self.markers_dt/self.analogs_dt)]
        # Right
        right_cal_velocity = np.diff(self.markers_sorted[2, self.model_marker_names.index("RCAL"), :]) / self.markers_dt
        right_heel_moving = np.where(right_cal_velocity > 0.1)[0]
        right_heel_moving_sequence = np.array_split(
            right_heel_moving, np.flatnonzero(np.diff(right_heel_moving) > 1) + 1
        )
        for sequence in right_heel_moving_sequence:
            self.events["right_leg_heel_off"] += [int(sequence[0]*self.markers_dt/self.analogs_dt)]


    def detect_toes_off(self):
        """
        Detect the toes off event when the vertical GRF is lower than a threshold
        """
        # Left
        swing_timings = np.where(self.phases_left_leg["swing"])[0]
        left_swing_sequence = np.array_split(
            swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1
        )
        for i_swing, swing_phase in enumerate(left_swing_sequence):
            beginning_swing_idx = swing_phase[0]
            if beginning_swing_idx == self.nb_analog_frames - 1 or beginning_swing_idx == 0:
                continue
            self.events["left_leg_toes_off"] += [int(beginning_swing_idx)]

        # Right
        swing_timings = np.where(self.phases_right_leg["swing"])[0]
        right_swing_sequence = np.array_split(
            swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1
        )
        for i_swing, swing_phase in enumerate(right_swing_sequence):
            beginning_swing_idx = swing_phase[0]
            if beginning_swing_idx == self.nb_analog_frames - 1 or beginning_swing_idx == 0:
                continue
            self.events["right_leg_toes_off"] += [int(beginning_swing_idx)]


    def detect_swing_phases_temporary(self):
        """
        Detect the swing phase when the vertical GRF is lower than a threshold
        """
        minimal_vertical_force_threshold = 0.007
        grf_right_z_filtered = moving_average(self.grf_sorted[0, 2, :], 21)
        grf_left_z_filtered = moving_average(self.grf_sorted[1, 2, :], 21)
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
                self.phases_left_leg[phase_name][init_idx:next_closing_idx + 1] = 1

        # Right
        for init_idx in self.events["right_leg_" + init_event_name]:
            list_index_idx = np.where(np.array(self.events["right_leg_" + closing_event_name]) - init_idx > 0)[0]
            if list_index_idx.shape != (0,):
                list_index_idx = list_index_idx[0]
                next_closing_idx = self.events["right_leg_" + closing_event_name][list_index_idx]
                self.phases_right_leg[phase_name][init_idx:next_closing_idx + 1] = 1
        return


    def detect_phases_both_legs(self, phase_name, left_leg_phase_name, right_leg_phase_name):
        self.phases[phase_name] = np.logical_and(self.phases_left_leg[left_leg_phase_name],
                                                 self.phases_right_leg[right_leg_phase_name])
        return


    def find_event_timestamps(self):
        # Detect when each foot is in the air as a preliminary step for the detection of events
        # Please not that anything happening before the first temporary swing phase is not considered
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

        # Plot the phase detection results
        self.plot_events()


    def plot_events(self):
        # time_velocity = (self.marker_time_vector[1:] + self.marker_time_vector[:-1]) / 2
        # grf_left_y_filtered = moving_average(self.grf_sorted[0, 1, :], 21)
        # grf_left_z_filtered = moving_average(self.grf_sorted[0, 2, :], 21)
        # grf_left_z_filtered = moving_average(self.grf_sorted[0, 2, :], 35)
        # axs[0].plot(grf_left_y_filtered, '--g', label='Y filtered')
        # axs[0].plot(grf_left_z_filtered, '--b', label='Z filtered')
        # axs[0].plot(np.where(self.phases_left_leg["swing"])[0], self.grf_sorted[0, 2, np.where(self.phases_left_leg["swing"])[0]], '.k')
        # for heel_touch_index in self.events["left_leg_heel_touch"]:
        #     axs[0].axvline(heel_touch_index, color='k')
        # for toes_touch_index in self.events["left_leg_toes_touch"]:
        #     axs[0].axvline(toes_touch_index, color='r')
        # # axs[0].plot(np.array(self.events["left_leg_heel_touch"]), self.grf_sorted[0, 2, np.array(self.events["left_leg_heel_touch"])], 'om')
        # # axs[0].plot(np.array(self.events["test"]), self.grf_sorted[0, 1, np.array(self.events["test"])], 'og')


        fig, axs = plt.subplots(3, 1, figsize=(15, 7))

        axs[0].plot(self.analogs_time_vector, self.grf_sorted[0, 0, :], '-r', label='Medio-lateral')
        axs[0].plot(self.analogs_time_vector, self.grf_sorted[0, 1, :], '-g', label='Antero-posterior')
        axs[0].plot(self.analogs_time_vector, self.grf_sorted[0, 2, :], '-b', label='Vertical')

        axs[1].plot(self.analogs_time_vector, self.grf_sorted[1, 0, :], '-r', label='Medio-lateral')
        axs[1].plot(self.analogs_time_vector, self.grf_sorted[1, 1, :], '-g', label='Antero-posterior')
        axs[1].plot(self.analogs_time_vector, self.grf_sorted[1, 2, :], '-b', label='Vertical')

        color = colormaps["magma"]
        for i_phase, key in enumerate(self.phases_left_leg):
            # Left
            # axs[0].plot(self.analogs_time_vector, self.phases_left_leg[key], '.', color=color(i_phase/4), label=key)
            start_idx = 0
            is_label = False
            for idx in range(1, self.phases_left_leg[key].shape[0]):
                if self.phases_left_leg[key][idx] == True:
                    if self.phases_left_leg[key][idx-1] == False:
                        start_idx = idx
                if self.phases_left_leg[key][idx] == False:
                    if self.phases_left_leg[key][idx-1] == True:
                        end_idx = idx
                        if not is_label:
                            axs[0].axvspan(self.analogs_time_vector[start_idx], self.analogs_time_vector[end_idx],
                                           alpha=0.2, color=color(i_phase / 4), label=key)
                            is_label = True
                        else:
                            axs[0].axvspan(self.analogs_time_vector[start_idx], self.analogs_time_vector[end_idx], alpha=0.2, color=color(i_phase/4))
            # Right
            # axs[1].plot(self.analogs_time_vector, self.phases_right_leg[key], '.', color=color(i_phase/4), label=key)
            start_idx = 0
            is_label = False
            for idx in range(1, self.phases_right_leg[key].shape[0]):
                if self.phases_right_leg[key][idx] == True:
                    if self.phases_right_leg[key][idx-1] == False:
                        start_idx = idx
                if self.phases_right_leg[key][idx] == False:
                    if self.phases_right_leg[key][idx-1] == True:
                        end_idx = idx
                        if not is_label:
                            axs[1].axvspan(self.analogs_time_vector[start_idx], self.analogs_time_vector[end_idx],
                                           alpha=0.2, color=color(i_phase/4), label=key)
                            is_label = True
                        else:
                            axs[1].axvspan(self.analogs_time_vector[start_idx], self.analogs_time_vector[end_idx],
                                           alpha=0.2, color=color(i_phase / 4))

        for i_phase, key in enumerate(self.phases):
            # axs[2].plot(self.analogs_time_vector, self.phases[key], '.', color=color(i_phase/8), label=key)
            start_idx = 0
            is_label = False
            for idx in range(1, self.phases[key].shape[0]):
                if self.phases[key][idx] == True:
                    if self.phases[key][idx-1] == False:
                        start_idx = idx
                if self.phases[key][idx] == False:
                    if self.phases[key][idx-1] == True:
                        end_idx = idx
                        if not is_label:
                            axs[2].axvspan(self.analogs_time_vector[start_idx], self.analogs_time_vector[end_idx], alpha=0.2, color=color(i_phase/8), label=key)
                            is_label = True
                        else:
                            axs[2].axvspan(self.analogs_time_vector[start_idx], self.analogs_time_vector[end_idx], alpha=0.2, color=color(i_phase/8))

        axs[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        axs[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        axs[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        axs[0].set_ylabel('Left leg GRF')
        axs[0].set_xlim(self.analogs_time_vector[20000], self.analogs_time_vector[30000])
        axs[1].set_ylabel('Right leg GRF')
        axs[1].set_xlim(self.analogs_time_vector[20000], self.analogs_time_vector[30000])
        axs[2].set_ylabel('Phases both legs')
        axs[2].set_xlim(self.analogs_time_vector[20000], self.analogs_time_vector[30000])
        plt.savefig("GRF.png")
        plt.show()
        print('Here')
    
