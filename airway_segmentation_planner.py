from batchgenerators.utilities.file_and_folder_operations import join
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
import numpy as np


class AirwaySegPlanner(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "AirwaySegPlanner"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "AirwaySegPlanner_plans_3D.pkl")

    def get_target_spacing(self):
        return np.array([i * 2 for i in (10, 5.159, 5.159)])
