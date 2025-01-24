
import biorbd
import osim_to_biomod as otb


class OsimModels:
    # TODO: Charbie -> Do we have the right to add the OpenSim models to a public repository?
    # TODO: Charbie -> Otherwise, can Florian give the link to the OpenSim model?
    @property
    def osim_model_name(self):
        raise RuntimeError("This method is implemented in the child class. You should call OsimModels.[mode type name].osim_model_name.")

    @property
    def osim_model_full_path(self):
        raise RuntimeError("This method is implemented in the child class. You should call OsimModels.[mode type name].osim_model_full_path.")

    @property
    def muscles_to_ignore(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].muscles_to_ignore.")

    # Child classes acting as an enum
    class WholeBody:
        """This is a hole body model that consists of 23 bodies, 42 degrees of freedom and 30 muscles.
        The whole-body geometric model and the lower limbs, pelvis and upper limbs anthropometry are based on the running model of Hammer et al. 2010 which consists of 12 segments and 29 degrees of freedom.
        Extra segments and degrees of freedom were later added based on Dumas et al. 2007.
        Each lower extremity had seven degrees-of-freedom; the hip was modeled as a ball-and-socket joint (3 DoFs), the knee was modeled as a revolute joint with 1 dof, the ankle was modeled as 2 revolute joints and feet toes with one revolute joint.
        The pelvis joint was model as a free flyer joint (6 DoFs) to allow the model to translate and rotate in the 3D space, the lumbar motion was modeled as a ball-and-socket joint (3 DoFs) (Anderson and Pandy, 1999) and the neck joint was modeled as a ball-and-socket joint (3 DoFs).
        Mass properties of the torso and head (including the neck) segments were estimated from Dumas et al., 2007. Each arm consisted of 8 degrees-of-freedom; the shoulder was modeled as a ball-and-socket joint (3 DoFs), the elbow and forearm rotation were each modeled with revolute joints (1 dof) (Holzbaur et al., 2005), the wrist flexion and deviation were each modeled with revolute joints and the hand fingers were modeled with 1 revolute joint.
        Mass properties for the arms were estimated from 1 and de Leva, 1996. The model also include 30 superficial muscles of the whole body.
        [Charbie -> Link ?]
        """

        @property
        def osim_model_name(self):
            return "wholebody"

        @property
        def osim_model_full_path(self):
            return "models/OpenSim_models/wholebody.osim"

        @property
        def muscles_to_ignore(self):
            return ["ant_delt_r",
                             "ant_delt_l",
                             "medial_delt_l",
                             "post_delt_r",
                             "post_delt_l",
                             "medial_delt_r",
                             "ercspn_r",
                             "ercspn_l",
                             "rect_abd_r",
                             "rect_abd_l",
                             "r_stern_mast",
                             "l_stern_mast",
                             "r_trap_acr",
                             "l_trap_acr",
                             "TRIlong",
                             "TRIlong_l",
                             "TRIlat",
                             "TRIlat_l",
                             "BIClong",
                             "BIClong_l",
                             "BRD",
                             "BRD_l",
                             "FCR",
                             "FCR_l",
                             "ECRL",
                             "ECRL_l",
                             "PT",
                             "PT_l",
                             "LAT2",
                             "LAT2_l",
                             "PECM2",
                             "PECM2_l",
                             ] + ["glut_med1_r",
                                     "semiten_r",
                                     "bifemlh_r",
                                     "sar_r",
                                     "tfl_r",
                                     "vas_med_r",
                                     "vas_lat_r",
                                     "glut_med1_l",
                                     "semiten_l",
                                     "bifemlh_l",
                                     "sar_l",
                                     "tfl_l",
                                     "vas_med_l",
                                     "vas_lat_l"]


class BiomodModelCreator:
    def __init__(self, subject_name: str, osim_model_type):
        self.subject_name = subject_name
        self.osim_model = osim_model_type

        osim_model_path = "../models/OpenSim_models"
        biorbd_model_path = "../models/biorbd_models"
        vtp_geometry_path = "Geometry"  # TODO: Charbie -> can we point to the Opensim folder where all opensim's vtp files are stored
        self.osim_model_full_path = osim_model_path + '/' + osim_model_type.osim_model_name + '_' + subject_name + ".osim"
        self.biorbd_model_full_path = biorbd_model_path + '/' + osim_model_type.osim_model_name + '_' + subject_name + ".bioMod"

        converter = otb.Converter(
            self.biorbd_model_full_path,  # .bioMod file to export to
            self.osim_model_full_path,  # .osim file to convert from
            ignore_muscle_applied_tag=False,
            ignore_fixed_dof_tag=False,
            ignore_clamped_dof_tag=False,
            mesh_dir=vtp_geometry_path,
            muscle_type=otb.MuscleType.HILL,
            state_type=otb.MuscleStateType.DEGROOTE,
            print_warnings=True,
            print_general_informations=True,
            vtp_polygons_to_triangles=True,
            muscles_to_ignore=osim_model_type.muscles_to_ignore,
        )
        converter.convert_file()

        self.biorbd_model = biorbd.Model(self.biorbd_model_full_path)

    def inputs(self):
        return {
            "subject_name": self.subject_name,
            "osim_model_type": self.osim_model,
            "osim_model_full_path": self.osim_model_full_path,
        }

    def outputs(self):
        return {
            "biorbd_model_full_path": self.biorbd_model_full_path,
            "biorbd_model": self.biorbd_model,
        }
