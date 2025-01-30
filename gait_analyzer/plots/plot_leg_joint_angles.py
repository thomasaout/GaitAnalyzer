import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from gait_analyzer.plots.plot_utils import split_cycles, mean_cycles, LegToPlot, PlotType, from_analog_frame_to_marker_frame



def plot_leg_data(result_folder:str, leg_to_plot: LegToPlot, plot_type: PlotType, conditions_to_compare: list[str]):
    """
    This function plots the joint angles of the legs.
    """
    # TODO: ThomasAout/FloEthv -> please decide if you want to compare mean of all participants
    cycles_data = {cond: [] for cond in conditions_to_compare}
    # Load the treated data to plot
    for result_file in os.listdir(result_folder):
        if os.path.isdir(result_file):
            for file_in_sub_folder in os.listdir(result_file):
                if file_in_sub_folder.endswith(".pkl"):
                    with open(file_in_sub_folder, "rb") as file:
                        data = pickle.load(file)
                    subject_name = data["subject_name"]
                    cond = file_in_sub_folder.replace(subject_name, "").replace(".pkl", "")
                    event_idx = from_analog_frame_to_marker_frame(data["analogs_time_vector"],
                                                                  data["markers_time_vector"],
                                                                  data["events"]["right_leg_heel_touch"])
                    if cond in conditions_to_compare:
                        cycles_data[cond] += split_cycles(data[plot_type.value], event_idx)
        else:
            with open(result_file, "rb") as file:
                data = pickle.load(file)
            subject_name = data["subject_name"]
            cond = result_file.replace(subject_name, "").replace(".pkl", "")
            event_idx = from_analog_frame_to_marker_frame(data["analogs_time_vector"],
                                                          data["markers_time_vector"],
                                                          data["events"]["right_leg_heel_touch"])
            if cond in conditions_to_compare:
                cycles_data[cond] += split_cycles(data[plot_type.value], event_idx)

    # Prepare the plot
    if leg_to_plot == LegToPlot.RIGHT:
        plot_idx = [0] # TODO

    draw_plot(cycles_data, plot_idx)


def draw_plot(cycles_data: dict, plot_idx: list[int]):
    axs, fig = plt.subplots()

    for cycles in cycles_data:
        # Compute the mean over cycles
        mean_data, std_data = mean_cycles(cycles, nb_frames_interp=100)


    plt.xlabel("Frame")
    plt.ylabel("Joint Angle (deg)")
    plt.legend()
    plt.show()
