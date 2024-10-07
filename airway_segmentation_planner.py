import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import (
    ExperimentPlanner3D_v21,
)


class AirwaySegPlanner(ExperimentPlanner3D_v21):
    """
    Custom experiment planner class for airway segmentation using a modified 3D U-Net architecture.
    
    This class extends the ExperimentPlanner3D_v21 to set specific data identifiers and plan filenames for airway
    segmentation tasks, as well as to customize the target spacing used in resampling operations.
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        """
        Initializes the AirwaySegPlanner with specific folder paths and settings.

        Args:
            folder_with_cropped_data (str): Path to the folder containing cropped data.
            preprocessed_output_folder (str): Path to the folder where preprocessed data will be stored.
        """
        super(ExperimentPlanner3D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "AirwaySegPlanner"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "AirwaySegPlanner_plans_3D.pkl")

    def get_target_spacing(self):
        """
        Returns the target spacing to be used for resampling the image data.
        
        This function overrides the default target spacing and returns a specific spacing that is
        customized for airway segmentation tasks.

        Returns:
            np.ndarray: An array representing the target spacing for the image data.
        """
        return np.array([i * 2 for i in (10, 5.159, 5.159)])
