
from gait_analyzer.biomod_model_creator import BiomodModelCreator
from gait_analyzer.experimental_data import ExperimentalData


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
        self.kinematics = None
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


    def add_experimental_data(self, file_path: str):

        # Checks
        if self. experimental_data is not None:
            raise Exception("Experimental data already added")

        # Add experimental data
        self.experimental_data = ExperimentalData(file_path=file_path, biorbd_model_path=biorbd_model_path)

