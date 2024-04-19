import shutil
from pathlib import Path
from tqdm import tqdm
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data
from tifffile import TiffFile
import SimpleITK
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


def get_spacing_from_tif_airwaySeg(filename: str):

    with TiffFile(filename) as tif:
        try:
            imagej_metadata = tif.imagej_metadata
            x_spacing = (
                tif.pages[0].tags["XResolution"].value[1]
                / tif.pages[0].tags["XResolution"].value[0]
            )
            y_spacing = (
                tif.pages[0].tags["YResolution"].value[1]
                / tif.pages[0].tags["YResolution"].value[0]
            )
            z_spacing = imagej_metadata["spacing"]
            print(filename.split("/")[-1], z_spacing, y_spacing, x_spacing)
            return (z_spacing, y_spacing, x_spacing)
        except:
            print(filename.split("/")[-1], 10, 5.159, 5.159)
            return (10, 5.159, 5.159)


def convert_to_nifti(input_filename, output_filename, spacing, is_seg=False):
    with TiffFile(input_filename) as tif:
        npy_image = tif.asarray()
    if is_seg:
        print(input_filename.split("/")[-1], np.unique(npy_image))
        npy_image[npy_image > 0] = 1
    sitk_image = SimpleITK.GetImageFromArray(npy_image)
    sitk_image.SetSpacing(list(spacing)[::-1])
    SimpleITK.WriteImage(sitk_image, output_filename)


if __name__ == "__main__":
    # Download the data from https://zenodo.org/records/7413818
    download_path = Path("<path/to/your/downloaded/zenodo/data>")

    train_set = [
        "ITLI_002",
        "ITLI_003",
        "ITLI_011",
        "NOAI_001",
        "VAAD_002",
        "VAAD_004",
        "VAAD_010",
        "VAAD_015",
        "VAAD_018",
        "Lung_003",
        "Lung_004",
        "Lung_005",
        "Lung_006",
        "Lung_007",
        "Lung_008",
        "Lung_009",
        "Lung_010",
    ]

    zip_files = list(download_path.glob("*.zip"))
    # only unzip the train files for reproducing the training
    for zf in tqdm(zip_files):
        if zf.name.replace(".zip", "") in train_set:
            shutil.unpack_archive(zf, download_path / "extracted", "zip")

    task_name = "Task145_LungAirway"
    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)

    # glob in the extracted dir and only consider files that have GT (not "AI result")
    mcai_files1 = list((download_path / "extracted").glob("*/*/*MCAI*.tif"))
    mcai_files2 = list((download_path / "extracted").glob("*/*MCAI*.tif"))
    mcai_files = mcai_files1 + mcai_files2

    # iter over the MCAIs (manually corrected AI results -> last Active Learning Iteration)
    for mcai in sorted(mcai_files):
        # get img list
        if mcai.parent.name.startswith("02"):
            # there are subdirectories, filter NP files
            img_files = [
                i
                for i in list(mcai.parent.parent.glob("01*/*.tif"))
                if not "NP" in str(i)
            ]
        else:
            # there are no subdirectories and no NP files
            img_files = [
                i for i in list(mcai.parent.glob("*.tif")) if not "MCAI" in str(i)
            ]

        # convert raw and GT to nifti and save with _0000 in nnunet dataset
        # first GT, infer spacing from this file
        spacing = get_spacing_from_tif_airwaySeg(str(mcai))

        # iter over img_files, use same spacing as GT
        for img_file in img_files:
            output_file = join(
                target_imagesTr, img_file.name.replace(".tif", "_0000.nii.gz")
            )
            convert_to_nifti(str(img_file), output_file, spacing, is_seg=False)
            output_file_label = join(
                target_labelsTr, img_file.name.replace(".tif", ".nii.gz")
            )
            convert_to_nifti(str(mcai), output_file_label, spacing, is_seg=True)

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        None,
        (["LSmicroscopy"]),
        labels={0: "background", 1: "airway"},
        dataset_name=task_name,
    )
