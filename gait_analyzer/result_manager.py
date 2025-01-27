
from gait_analyzer.biomod_model_creator import BiomodModelCreator
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.events import Events
from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor


class ResultManager:
    """
    This class contains all the results from the gait analysis and is the main class handling all types of analysis to perform on the experimental data.
    """
    def __init__(self, subject_name: str, subject_mass: float):
        """
        Initialize the ResultManager.
        .
        Parameters
        ----------
        subject_name: str
            The name of the subject
        subject_mass: float
            The mass of the subject
        """
        # Checks:
        if not isinstance(subject_name, str):
            raise ValueError("subject_name must be a string")
        if not isinstance(subject_mass, int):
            raise ValueError("subject_mass must be an float")

        # Attributes
        self.subject_name = subject_name
        self.subject_mass = subject_mass
        self.experimental_data = None
        self.biorbd_model_creator = None
        self.gait_parameters = None
        self.events = None
        self.kinematics_reconstructor = None
        self.optimal_estimation = None


    def create_biorbd_model(self, osim_model_type):
        """
        Create and add the biorbd model to the ResultManager
        """

        # Checks
        if self.biorbd_model_creator is not None:
            raise Exception("Biorbd model already added")

        # Add BiomodModelCreator
        self.biorbd_model_creator = BiomodModelCreator(self.subject_name, osim_model_type)


    def add_experimental_data(self, c3d_file_name: str, animate_c3d_flag: bool = False):

        # Checks
        if self.experimental_data is not None:
            raise Exception("Experimental data already added")
        if self.biorbd_model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_biorbd_model()")

        # Add experimental data
        self.experimental_data = ExperimentalData(c3d_file_name=c3d_file_name,
                                                  biorbd_model=self.biorbd_model_creator.biorbd_model,
                                                  animate_c3d_flag=animate_c3d_flag)


    def add_events(self, plot_phases_flag):

        # Checks
        if self.biorbd_model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_biorbd_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is not None:
            raise Exception("Events already added")

        # Add events
        self.events = Events(experimental_data = self.experimental_data,
                             plot_phases_flag=plot_phases_flag)

    def reconstruct_kinematics(self, animate_kinematics_flag: bool = False):
        # Checks
        if self.biorbd_model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_biorbd_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.kinematics_reconstructor is not None:
            raise Exception("kinematics_reconstructor already added")

        # Reconstruct kinematics
        self.kinematics_reconstructor = KinematicsReconstructor(self.experimental_data, self.biorbd_model_creator.biorbd_model, animate_kinematics_flag=animate_kinematics_flag)
