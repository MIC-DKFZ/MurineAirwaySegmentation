import os
from pathlib import Path

from data_conversion import convert_to_nifti, get_spacing_from_tif_airwaySeg

if __name__ == "__main__":
    """
    Main script to convert a single lung TIFF file to NIfTI format with appropriate spacing information.
    This script prepares the NIfTI file for use in nnUNet or other medical imaging applications.
    """

    # Specify the path to the TIFF file you want to convert
    path_to_lung_file = "path/to/the/lung/tiffile/you/want/to/convert"

    img_path = Path(path_to_lung_file) # Convert the file path to a Path object for easier manipulation
    out_folder = img_path.parent.parent / "lung_nnunet_format" # Set the output folder for the converted file

    # Create the output folder if it doesn't already exist
    os.makedirs(out_folder, exist_ok=True)

    # Define the output file path by replacing the TIFF file extension with the nnUNet naming convention
    output_path = out_folder / img_path.name.replace(".tif", "_0000.nii.gz")

    # Extract the spacing information from the TIFF file
    spacing = get_spacing_from_tif_airwaySeg(str(img_path))
    
    # Convert the TIFF file to NIfTI format using the extracted spacing
    convert_to_nifti(str(img_path), str(output_path), spacing)
