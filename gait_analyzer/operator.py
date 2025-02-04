import os
import pickle
from scipy.io import savemat
import pandas as pd
import numpy as np
import git
from datetime import date
import subprocess
import json


class Operator:

    @staticmethod
    def moving_average(x: np.array, window_size: int):
        """
        Compute the moving average of a signal
        .
        Parameters
        ----------
        x: np.array
            The signal to be averaged (for now, this can only be a vector)
        window_size: int
            The size of the window to compute the average on
        .
        Returns
        -------
        x_averaged: np.array
            The signal processed using the moving average
        """
        # Checks
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd number")
        if len(x.shape) != 1:
            if len(x.shape) == 2 and x.shape[1] == 1:
                x = x.flatten()
            else:
                raise ValueError("x must be a vector")
        if x.shape[0] / 2 < window_size:
            raise ValueError("window_size must be smaller than half of the length of the signal")

        # Compute the moving average
        x_averaged = np.zeros_like(x)
        for i in range(len(x)):
            if i < window_size // 2:
                x_averaged[i] = np.mean(x[: i + window_size // 2 + 1])
            elif i >= len(x) - window_size // 2:
                x_averaged[i] = np.mean(x[i - window_size // 2 :])
            else:
                x_averaged[i] = np.mean(x[i - window_size // 2 : i + window_size // 2 + 1])
        return x_averaged

    @staticmethod
    def from_analog_frame_to_marker_frame():
        # TODO: Charbie
        pass

    @staticmethod
    def from_marker_frame_to_analog_frame():
        # TODO: Charbie
        pass


class AnalysisPerformer:
    def __init__(
        self,
        analysis_to_perform: callable,
        subjects_to_analyze: list[str],
        result_folder: str = "../results/",
        skip_if_existing: bool = False,
    ):
        """
        Initialize the AnalysisPerformer.
        .
        Parameters
        ----------
        analysis_to_perform: callable(subject_name: str, subject_mass: float, c3d_file_name: str)
            The analysis to perform
        subjects_to_analyze: list[str]
            The list of subjects to analyze
        result_folder: str
            The folder where the results will be saved. It will look like result_folder/subject_name.
        skip_if_existing: bool
            If True, the analysis will not be performed if the results already exist.
        """
        # Checks:
        if not callable(analysis_to_perform):
            raise ValueError("analysis_to_perform must be a callable")
        if not isinstance(subjects_to_analyze, list):
            raise ValueError("subjects_to_analyze must be a list")
        for subject in subjects_to_analyze:
            if not isinstance(subject, str):
                raise ValueError("All elements of subjects_to_analyze must be strings")
        if not isinstance(result_folder, str):
            raise ValueError("result_folder must be a string")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
            print(f"Result folder did not exist, I have created it here {os.path.abspath(result_folder)}")

        # Initial attributes
        self.analysis_to_perform = analysis_to_perform
        self.subjects_to_analyze = subjects_to_analyze
        self.result_folder = result_folder
        self.skip_if_existing = skip_if_existing

        # Run the analysis
        self.run_analysis()

    @staticmethod
    def get_version():
        """
        Save the version of the code and the date of the analysis for future reference
        """

        # Packages installed in the env
        # Running 'conda list' command and parse it as JSON
        result = subprocess.run(["conda", "list", "--json"], capture_output=True, text=True)
        packages = json.loads(result.stdout)
        packages_versions = {elt["name"]: elt["version"] for elt in packages}

        # Get the version of the current package
        repo = git.Repo(search_parent_directories=True)
        commit_id = str(repo.commit())
        branch = str(repo.active_branch)
        try:
            tag = repo.git.describe("--tags")
        except git.exc.GitCommandError:
            tag = "No tag"
        gait_analyzer_version = repo.git.version_info
        git_date = repo.git.log("-1", "--format=%cd")
        version_dic = {
            "commit_id": commit_id,
            "git_date": git_date,
            "branch": branch,
            "tag": tag,
            "gait_analyzer_version": gait_analyzer_version,
            "date_of_the_analysis": date.today().strftime("%b-%d-%Y-%H-%M-%S"),
            "biorbd_version": packages_versions["biorbd"],
            "pyomeca_version": packages_versions["pyomeca"] if "pyomeca" in packages_versions else "Not installed",
            "ezc3d_version": packages_versions["ezc3d"],
            "bioptim_version": (
                packages_versions["bioptim"] if "bioptim" in packages_versions else "Not installed through conda-forge"
            ),
        }
        return version_dic

    def save_subject_results(self, results, result_file_name: str):
        """
        Save the results of the analysis in a pickle file and a matlab file.
        .
        Parameters
        ----------
        results: ResultManager
            The ResultManager containing the results of the analysis performed by analysis_to_perform
        result_file_name: str
            The name of the file where the results will be saved. The file will be saved as result_file_name.pkl and result_file_name.mat
        """

        result_dict = self.get_version()
        for attr_name in dir(results):
            attr = getattr(results, attr_name)
            if not callable(attr) and not attr_name.startswith("__"):
                if hasattr(attr, "outputs") and callable(getattr(attr, "outputs")):
                    this_output_dict = attr.outputs()
                    for key, value in this_output_dict.items():
                        if key in result_dict:
                            raise ValueError(
                                f"Key {key} from class {attr_name} already exists in the result dictionary, please change the key to differentiate them."
                            )
                        elif key == "biorbd_model":
                            pass  # biorbd models are not picklable
                        else:
                            result_dict[key] = value

        # Save the results
        # For python analysis
        with open(result_file_name + ".pkl", "wb") as f:
            pickle.dump(result_dict, f)
        # For matlab analysis
        savemat(result_file_name + ".mat", result_dict)

    def run_analysis(self):
        """
        Loops over the data files and perform the analysis specified by the user (on the subjects specified by the user).
        """
        # Loop over all subjects
        for subject_name in self.subjects_to_analyze:
            # Checks
            if not os.path.exists(f"../data/{subject_name}"):
                os.makedirs(f"../data/{subject_name}")
                tempo_subject_path = os.path.abspath(f"../data/{subject_name}")
                raise RuntimeError(
                    f"Data folder for subject {subject_name} does not exist. I have created it here {tempo_subject_path}, please put the data files in here."
                )
            if not os.path.exists(f"../data/{subject_name}/Sujet_{subject_name}.xlsx"):
                tempo_subject_path = os.path.abspath(f"../data/{subject_name}/Sujet_{subject_name}.xlsx")
                raise FileNotFoundError(f"Please put the participant information file here {tempo_subject_path}")

            # Get the subject's information
            subject_path = f"../data/{subject_name}/Sujet_{subject_name}.xlsx"
            dfs = pd.read_excel(subject_path, sheet_name=None)
            mass_column_idx = list(dfs["Subject presentation"]["Information Sujet "]).index("Masse")
            subject_mass = float(dfs["Subject presentation"]["Value"][mass_column_idx])

            # Loop over all data files
            for data_file in os.listdir(f"../data/{subject_name}"):
                if data_file.endswith("Statique.c3d") or not data_file.endswith(
                    ".c3d"
                ):  # TODO: Charbie -> add other "conditions_to_exclude" here
                    continue
                c3d_file_name = f"../data/{subject_name}/{data_file}"
                result_folder = f"{self.result_folder}/{subject_name}"
                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)
                result_file_name = f"{result_folder}/{data_file.replace('.c3d', '_results')}"

                if self.skip_if_existing and os.path.exists(result_file_name + ".pkl"):
                    print(f"Skipping {subject_name} - {data_file} because it already exists.")
                    continue

                results = self.analysis_to_perform(subject_name, subject_mass, c3d_file_name)
                self.save_subject_results(results, result_file_name)
