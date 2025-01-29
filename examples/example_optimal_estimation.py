import os
from gait_analyzer import (
    helper,
    ResultManager,
    OsimModels,
    Operator,
    AnalysisPerformer,
)


def analysis_to_perform(subject_name: str, subject_mass: float, c3d_file_name: str):
    # --- Example of analysis --- #
    results = ResultManager(subject_name=subject_name, subject_mass=subject_mass)
    # Please note that the OpenSim model should already be scaled in the OpenSim GUI
    results.create_biorbd_model(osim_model_type=OsimModels.WholeBody())
    results.add_experimental_data(c3d_file_name=c3d_file_name, animate_c3d_flag=False)

    results.add_events(plot_phases_flag=False)
    # results.reconstruct_kinematics(animate_kinematics_flag=False)
    # results.estimate_optimally()

    return results


def parameters_to_extract_for_statistical_analysis():
    # TODO: Add the parameters you want to extract for statistical analysis
    pass


if __name__ == "__main__":

    #  --- Example of how to get help on a GaitAnalyzer class --- #
    helper(Operator)

    # --- Example of how to run the analysis --- #
    AnalysisPerformer(analysis_to_perform, subjects_to_analyze=["VIF_04"], result_folder="results")



