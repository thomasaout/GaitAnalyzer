from gait_analyzer import (
    helper,
    ResultManager,
    OsimModels,
    Operator,
    AnalysisPerformer,
    PlotLegData,
    LegToPlot,
    PlotType,
)


def analysis_to_perform(subject_name: str, subject_mass: float, c3d_file_name: str, result_folder: str):

    # --- Example of analysis --- #
    results = ResultManager(subject_name=subject_name, subject_mass=subject_mass, result_folder=result_folder)
    # Please note that the OpenSim model should already be scaled in the OpenSim GUI
    results.create_biorbd_model(osim_model_type=OsimModels.WholeBody(), skip_if_existing=True)
    results.add_experimental_data(c3d_file_name=c3d_file_name, animate_c3d_flag=False)

    results.add_events(plot_phases_flag=False)
    results.reconstruct_kinematics(animate_kinematics_flag=False, plot_kinematics_flag=True, skip_if_existing=False)
    results.perform_inverse_dynamics(animate_dynamics_flag=True)
    # results.estimate_optimally()

    return results


def parameters_to_extract_for_statistical_analysis():
    # TODO: Add the parameters you want to extract for statistical analysis
    pass


if __name__ == "__main__":

    # --- Example of how to get help on a GaitAnalyzer class --- #
    helper(Operator)

    # --- Steps to perform before running the analysis --- #
    # 1. Placing the data in the folder "data/[subject_name]/"
    # 2. Using the code in the folder c3d_to_trc to convert the static file (MATLAB: main.m, Python: convert_c3d_files.py)
    # 3. Generating a scaled model in the OpenSim GUI (using the .trc file) and placing it in the folder "models/OpenSim_models/[model_name]_[subject_name].osim"

    # --- Example of how to run the analysis --- #
    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze=["AOT_01"],
        result_folder="results",
        trails_to_analyze=["_ManipStim_L200_F30_I20"],
        skip_if_existing=False
    )

    # --- Example of how to plot the joint angles --- #
    plot = PlotLegData(result_folder="results",
                       leg_to_plot=LegToPlot.RIGHT,
                       plot_type=PlotType.Q,
                       conditions_to_compare=["_ManipStim_L200_F30_I20"])
                       # conditions_to_compare=["_ManipStim_L200_F30_I20",
                       #                        "_ManipStim_L300_F30_I60",
                       #                        "_ManipStim_L400_F30_I40",
                       #                        "_ManipStim_L400_F40_I40",
                       #                        "_ManipStim_L400_F50_I40"])
    plot.draw_plot()
    plot.save("results/AOT_01_Q_plot_temporary.png")
    plot.show()

    # --- Example of how to plot the joint torques --- #
    plot = PlotLegData(result_folder="results",
                       leg_to_plot=LegToPlot.RIGHT,
                       plot_type=PlotType.TAU,
                       conditions_to_compare=["_ManipStim_L200_F30_I20"])
                       # conditions_to_compare=["_ManipStim_L200_F30_I20",
                       #                        "_ManipStim_L300_F30_I60",
                       #                        "_ManipStim_L400_F30_I40",
                       #                        "_ManipStim_L400_F40_I40",
                       #                        "_ManipStim_L400_F50_I40"])
    plot.draw_plot()
    plot.save("results/AOT_01_Q_plot_temporary.png")
    plot.show()
