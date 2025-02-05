import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps

from gait_analyzer.plots.plot_utils import split_cycles, mean_cycles, LegToPlot, PlotType


class PlotLegData:
    def __init__(self, result_folder: str, leg_to_plot: LegToPlot, plot_type: PlotType, conditions_to_compare: list[str]):
        # Checks
        if not isinstance(result_folder, str):
            raise ValueError("result_folder must be a string")
        if not os.path.isdir(result_folder):
            raise ValueError(f"The result_folder specified {result_folder} does not exist.")
        if not isinstance(leg_to_plot, LegToPlot):
            raise ValueError("leg_to_plot must be LegToPlot type")
        if not isinstance(plot_type, PlotType):
            raise ValueError("plot_type must be PlotType type")
        if not isinstance(conditions_to_compare, list):
            raise ValueError("conditions_to_compare must be a list")

        # Initial attributes
        self.result_folder = result_folder
        self.leg_to_plot = leg_to_plot
        self.plot_type = plot_type
        self.conditions_to_compare = conditions_to_compare

        # Extended attributes
        self.cycles_data = None
        self.plot_idx = None
        self.plot_labels = None
        self.fig = None

        # Prepare the plot
        self.prepare_plot()


    def prepare_plot(self):
        """
        This function prepares the data to plot.
        """
        # TODO: ThomasAout/FloEthv -> please decide if you want to compare mean of all participants
        cycles_data = {cond: [] for cond in self.conditions_to_compare}
        # Load the treated data to plot
        for result_file in os.listdir(self.result_folder):
            if os.path.isdir(os.path.join(self.result_folder, result_file)):
                for file_in_sub_folder in os.listdir(os.path.join(self.result_folder, result_file)):
                    file_in_sub_folder = os.path.join(self.result_folder, result_file, file_in_sub_folder)
                    if file_in_sub_folder.endswith(".pkl"):
                        with open(file_in_sub_folder, "rb") as file:
                            data = pickle.load(file)
                        subject_name = data["subject_name"]
                        cond = file_in_sub_folder.replace(f"{self.result_folder}/{result_file}/", "").replace(subject_name, "").replace("_results.pkl", "")
                        event_idx = from_analog_frame_to_marker_frame(data["analogs_time_vector"],
                                                                      data["markers_time_vector"],
                                                                      data["events"]["right_leg_heel_touch"])
                        if cond in self.conditions_to_compare:
                            cycles_data[cond] += split_cycles(data[self.plot_type.value], event_idx)
            else:
                if result_file.endswith(".pkl"):
                    with open(result_file, "rb") as file:
                        data = pickle.load(file)
                    subject_name = data["subject_name"]
                    cond = result_file.replace(subject_name, "").replace(".pkl", "")
                    event_idx = from_analog_frame_to_marker_frame(data["analogs_time_vector"],
                                                                  data["markers_time_vector"],
                                                                  data["events"]["right_leg_heel_touch"])
                    if cond in self.conditions_to_compare:
                        cycles_data[cond] += split_cycles(data[self.plot_type.value], event_idx)

        # TODO: remove ------------------------
        plt.figure()
        data_tempo = cycles_data["_ManipStim_L400_F40_I40"]
        for i in range(len(data_tempo)):
            print(data_tempo[i].shape)
            plt.plot(data_tempo[i][3, :])
        plt.savefig("plottttt.png")
        plt.show()

        # Prepare the plot
        if self.leg_to_plot == LegToPlot.RIGHT:
            plot_idx = [20, 3, 6, 9, 11]
            plot_labels = ["Torso", "Pelvis", "Femur", "Tibia", "Calcaneus"]
            # plot_labels = ["Hip", "Knee", "Ankle"]  # TODO
        else:
            raise NotImplementedError("Only the right leg is implemented for now.")

        # Store the output
        self.cycles_data = cycles_data
        self.plot_idx = plot_idx
        self.plot_labels = plot_labels


    def draw_plot(self):
        # TODO: Charbie -> combine plots in one figure (Q and Power for example side by side)

        # Initialize the plot
        if self.leg_to_plot in [LegToPlot.RIGHT, LegToPlot.LEFT]:
            n_cols = 1
            fig_width = 5
        else:
            n_cols = 2
            fig_width = 10
        n_rows = len(self.plot_idx) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, 10))
        n_data_to_plot = len(self.cycles_data)
        colors = [colormaps["magma"](i/n_data_to_plot) for i in range(n_data_to_plot)]
        nb_frames_interp = 101
        normalized_time = np.linspace(0, 100, nb_frames_interp)

        # Store the mean ans std for further analysis
        all_mean_data = np.zeros((n_data_to_plot, len(self.plot_idx), nb_frames_interp))
        all_std_data = np.zeros((n_data_to_plot, len(self.plot_idx), nb_frames_interp))
        all_labels = []

        # Plot the data
        for i_cycle, key in enumerate(self.cycles_data):
            cycles = self.cycles_data[key]
            # Compute the mean over cycles
            mean_data, std_data = mean_cycles(cycles, index_to_keep=self.plot_idx, nb_frames_interp=nb_frames_interp)
            mean_data = np.rad2deg(mean_data)
            std_data = np.rad2deg(std_data)
            all_mean_data[i_cycle, :, :] = mean_data
            all_std_data[i_cycle, :, :] = std_data
            all_labels.append(self.conditions_to_compare[i_cycle])
            for i_ax, ax in enumerate(axs):
                ax.fill_between(normalized_time,
                                mean_data[i_ax, :] - std_data[i_ax, :],
                                mean_data[i_ax, :] + std_data[i_ax, :],
                                color=colors[i_cycle], alpha=0.3)
                ax.plot(normalized_time,
                        mean_data[i_ax, :], label=f"{self.conditions_to_compare[i_cycle]}",
                        color=colors[i_cycle])
                if i_cycle == 0:
                    if i_ax == len(axs) - 1:
                        ax.set_xlabel("Normalized time [%]")
                    ax.set_ylabel(f"{self.plot_labels[i_ax]} angle " + r"[$^\circ$]")

        # Legend
        for i_cycle, key in enumerate(self.cycles_data):
            axs[0].fill_between([], [], [], color=colors[i_cycle], alpha=0.3, label=f"{self.conditions_to_compare[i_cycle]}")
        axs[0].legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', borderaxespad=0.)
        fig.tight_layout()
        fig.savefig("plot_Q_temporary.png")
        self.fig = fig


    def save(self, file_name: str):
        self.fig.savefig(file_name)


    def show(self):
        self.fig.show()
