from data_conversion import convert_to_nifti, get_spacing_from_tif_airwaySeg
from pathlib import Path
import os


if __name__ == "__main__":

    path_to_lung_file = "path/to/the/lung/tiffile/you/want/to/convert"

    img_path = Path(path_to_lung_file)
    out_folder = img_path.parent.parent / "lung_nnunet_format"
    os.makedirs(out_folder, exist_ok=True)

    output_path = out_folder / img_path.name.replace(".tif", "_0000.nii.gz")

    spacing = get_spacing_from_tif_airwaySeg(str(img_path))
    convert_to_nifti(str(img_path), str(output_path), spacing)
