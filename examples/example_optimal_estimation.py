
from gait_analyzer import (
    helper,
    ResultManager,
    OsimModels,
    Operator,
)


if __name__ == "__main__":

    c3d_file_name = "VIF_04_Cond0007.c3d"

    #  --- Example of how to get help on a GaitAnalyzer class --- #
    helper(Operator)

    # --- Example of analysis --- #
    results = ResultManager(subject_name="VIF_04", subject_mass=71)
    # Please note that the OpenSim model should already be scaled in the OpenSim GUI
    results.create_biorbd_model(osim_model_type=OsimModels.WholeBody())
    results.add_experimental_data(c3d_file_name=c3d_file_name, animate_c3d_flag=False)
    results.add_events(plot_phases_flag=True)
    results.reconstruct_kinematics(animate_kinematics_flag=False)
    results.estimate_optimally()

    # TODO: Guys -> Which kind of stats do we want to perform ?
