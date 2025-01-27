
import numpy as np
import ezc3d
import biorbd


from gait_analyzer import (
    helper,
    ExperimentalData,
    ResultManager,
    OsimModels,
    Operator,
)


def extract_data_for_OCP(exp_data, file_path):

    model = biorbd.Model(exp_data.biorbd_model_path)
    # q_names = [m.to_string() for m in model.nameDof()]
    q_names_model = ['Pelvis_TransX', 'Pelvis_TransY', 'Pelvis_TransZ', 'Pelvis_RotX', 'Pelvis_RotY', 'Pelvis_RotZ',
               'Thorax_RotX', 'Thorax_RotY', 'Thorax_RotZ',
               'RHumerus_RotX', 'RHumerus_RotY', 'RHumerus_RotZ',
               'RRadius_RotX', 'RRadius_RotY', 'RRadius_RotZ',
               'RHand_RotX', 'RHand_RotY', 'RHand_RotZ',
               'LHumerus_RotX', 'LHumerus_RotY', 'LHumerus_RotZ',
               'LRadius_RotX', 'LRadius_RotY', 'LRadius_RotZ',
               'LHand_RotX', 'LHand_RotY', 'LHand_RotZ',
               'RFemur_RotX', 'RFemur_RotY', 'RFemur_RotZ',
               'RTibia_RotX', 'RTibia_RotY', 'RTibia_RotZ',
               'RFoot_RotX', 'RFoot_RotY', 'RFoot_RotZ',
               'LFemur_RotX', 'LFemur_RotY', 'LFemur_RotZ',
               'LTibia_RotX', 'LTibia_RotY', 'LTibia_RotZ',
               'LFoot_RotX', 'LFoot_RotY', 'LFoot_RotZ']
    q_names_c3d = ['LHipAngles',
                   'LKneeAngles',
                   'LAnkleAngles',
                   'RHipAngles',
                   'RKneeAngles',
                   'RAnkleAngles',
                   'LShoulderAngles',
                   'LElbowAngles',
                   'LWristAngles',
                   'RShoulderAngles',
                   'RElbowAngles',
                   'RWristAngles',
                   'LNeckAngles',
                   'RNeckAngles',
                   'LSpineAngles',
                   'RSpineAngles',
                   'LHeadAngles',
                   'RHeadAngles',
                   'LThoraxAngles',
                   'RThoraxAngles',
                   'LPelvisAngles',
                   'RPelvisAngles',


                   'LHipVitesse', 'LKneeVitesse', 'LAnkleVitesse', 'LAbsAnkleVitesse', 'LFootProgressVitesse', 'RHipVitesse', 'RKneeVitesse', 'RAnkleVitesse', 'RAbsAnkleVitesse', 'RFootProgressVitesse', 'LShoulderVitesse', 'LElbowVitesse', 'LWristVitesse', 'RShoulderVitesse', 'RElbowVitesse', 'RWristVitesse', 'LNeckVitesse', 'RNeckVitesse', 'LSpineVitesse', 'RSpineVitesse', 'LHeadVitesse', 'RHeadVitesse', 'LThoraxVitesse', 'RThoraxVitesse', 'LPelvisVitesse', 'RPelvisVitesse']

    c3d = ezc3d.c3d(file_path)
    exp_marker_names = c3d["parameters"]["POINT"]["LABELS"]["value"]


    c3d["data"]["points"][:3, exp_marker_names.index("LHipAngles"), :]
    q_exp = np.zeros((len(q_names), c3d["data"]["points"].shape[2]))
    q_exp[0, :] = c3d["data"]["points"][:3, exp_marker_names.index(q_names[i_dof]), :]







if __name__ == "__main__":

    # biorbd_model_path = "data/VIF_04.bioMod"
    # static_file_path = "data/VIF_04_Statique.c3d"
    c3d_file_name = "VIF_04_Cond0007.c3d"

    #  --- Example of how to get help on a GaitAnalyzer class --- #
    helper(Operator)

    # --- Example of analysis --- #
    results = ResultManager(subject_name="VIF_04", subject_mass=71)
    # Please note that the OpenSim model should already be scaled in the OpenSim GUI
    results.create_biorbd_model(osim_model_type=OsimModels.WholeBody())
    results.add_experimental_data(c3d_file_name=c3d_file_name, animate_c3d_flag=False)
    # results.add_events(plot_phases=True)
    results.reconstruct_kinematics(animate_kinematics_flag=False)

    extract_data_for_OCP(exp_data, file_path)

    ocp = prepare_ocp(biorbd_model_path=biorbd_model_path,
                      )

    # --- Solve the ocp --- #
    solver = Solver.IPOPT(show_online_optim=True)
    sol = ocp.solve(solver=solver)

    # --- Show results --- #
    # sol.graphs()
    sol.animate()
