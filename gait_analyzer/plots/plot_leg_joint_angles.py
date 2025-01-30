import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from gait_analyzer.plots.plot_utils import split_cycles, mean_cycles, LegToPlot, PlotType, from_analog_frame_to_marker_frame


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
                        cond = file_in_sub_folder.replace(subject_name, "").replace(".pkl", "")
                        event_idx = from_analog_frame_to_marker_frame(data["analogs_time_vector"],
                                                                      data["markers_time_vector"],
                                                                      data["events"]["right_leg_heel_touch"])
                        if cond in self.conditions_to_compare:
                            cycles_data[cond] += split_cycles(data[self.plot_type.value], event_idx)
            else:
                with open(result_file, "rb") as file:
                    data = pickle.load(file)
                subject_name = data["subject_name"]
                cond = result_file.replace(subject_name, "").replace(".pkl", "")
                event_idx = from_analog_frame_to_marker_frame(data["analogs_time_vector"],
                                                              data["markers_time_vector"],
                                                              data["events"]["right_leg_heel_touch"])
                if cond in self.conditions_to_compare:
                    cycles_data[cond] += split_cycles(data[self.plot_type.value], event_idx)

        # Prepare the plot
        if self.leg_to_plot == LegToPlot.RIGHT:
            plot_idx = [0] # TODO

        # Store the output
        self.cycles_data = cycles_data
        self.plot_idx = plot_idx


    def draw_plot(self):
        axs, fig = plt.subplots()

        for cycles in self.cycles_data:
            # Compute the mean over cycles
            mean_data, std_data = mean_cycles(cycles, nb_frames_interp=100)


        plt.xlabel("Frame")
        plt.ylabel("Joint Angle (deg)")
        plt.legend()
        self.fig = fig


    def save(self, file_name: str):
        self.fig.savefig(file_name)


    def show(self):
        self.fig.show()
