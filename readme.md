<img src="images/lung_example.gif" height="400px" />

# Introduction

This repository contains the code for training the final nnU-Net based segmentation model for the publication 
"Multimodal imaging and deep learning unveil pulmonary delivery profiles and acinar migration of tissue-resident 
macrophages in the lung" which is currently under review.

The nnU-Net model used for this publication makes changes to the target spacing for the 3d_fulles configuration and 
contains several changes to the data augmentation pipeline. The latter greatly increase the robustness of the network 
with respect to the commonly observed image artifacts in the dataset.

# Installation
Our installation instructions make use of concrete git commit hashes for batchgenerators and nnU-Net in order to 
ensure reproducibility. Both repositories are under constant development and we cannot guarantee compatibility looking 
forward.

Some files from this repository must be copied to the nnU-Net source code! Follow the instructions and everything will 
work out just fine!

**This repository ONLY works with Linux operating systems and requires a GPU!**

1) Setup a new virtual environment. Pip or conda doesn't matter. If you don't know how, google will help. Or chatGPT. You decide.
2) Install [PyTorch](https://pytorch.org/get-started/locally/) for your setup. This repository was tested with version 2.0.0.
3) Install batchgenerators: `pip install git+https://github.com/MIC-DKFZ/batchgenerators#6859efd8cd59000896c0bcb6313e2b8e12bbb031`
4) Clone the nnU-Net code and install it:
   - Creates a local copy of the code: `git clone https://github.com/MIC-DKFZ/nnUNet.git`
   - Move into this repository: `cd nnUNet`
   - Set the repository to exactly the commit we need: `git checkout 3855c6e5eb27f69edbc656733e35c9c456c3a926`
   - Install with: `pip install -e .`
5) Now install the new components of this repository into nnU-Net:
   - copy `airway_segmentation_planner.py` into `nnunet/experiment_planning/`
   - copy `nnUNetTrainerV2_airwayAug.py` into `nnunet/training/network_training/`
6) Set the necessary nnU-Net environment variables according to [these instructions](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/setting_up_paths.md)

**Done!** Typically this installation process takes 5-10 minutes.

# Demo 
## Download the demo lung
Download the demo lung ITLI_007 from [here](https://zenodo.org/records/7413818/files/ITLI_007.zip?download=1) (590.3GB) and unzip the file.

## Convert the file to nnU-Net format
The raw lung data comes in tif format. Transform the raw lung ```ITLI_007/01_Raw imaging data_compressed/ITLI_007_raw_AF520nm_09-45-10.tif``` to nnU-Net format as described [here](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/dataset_conversion.md). You can use the provided `demo_convert_file.py` script by setting the path to your downloaded `ITLI_007` tif file. 

```commandline
python demo_convert_file.py
```

## Run the inference
Download the [Pretrained models](#pretrained-models) and follow the instructions in [Inference/Prediction](#inferenceprediction) to predict the airways of the lung.
The `FOLDER_WITH_INPUT_IMAGES` is the folder containing your converted lung file.
Running the inference requires one or multiple GPUs and takes more than an hour.
Expected output is the predicted airway segmentation that matches the airway segmentation in the folder ```ITLI_007/02_AI results```.


# Reproducing the Training
If you prefer to use our already trained models, see [Pretrained models](#pretrained-models)
## Data conversion

The data is published and can be downloaded here: https://zenodo.org/records/7413818.
To reproduce the training the data needs to be converted to nnU-Net format as described [here](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/dataset_conversion.md). For that you can execute the provided `data_conversion.py` script. Set the path in the script to the folder where you downloaded the zenodo data. It will automatically unzip the train cases, transform them to nifti and save them as an nnU-Net dataset in the respective directory for raw images that you have set (as described in the [installation](#installation) step 6).


## nnU-Net planning and preprocessing
Run the following:
```commandline
nnUNet_plan_and_preprocess -t 145 -pl2d None -pl3d AirwaySegPlanner -tl 2 -tf 1
```
This will take the raw data, perform nnU-Net's planning and preprocessing steps and save the preprocessed training data

Now we can run the training. Run the following to train the 5-fold cross-validation:
```commandline
nnUNet_train 3d_fullres nnUNetTrainerV2_airwayAug 145 0 -p AirwaySegPlanner
nnUNet_train 3d_fullres nnUNetTrainerV2_airwayAug 145 1 -p AirwaySegPlanner
nnUNet_train 3d_fullres nnUNetTrainerV2_airwayAug 145 2 -p AirwaySegPlanner
nnUNet_train 3d_fullres nnUNetTrainerV2_airwayAug 145 3 -p AirwaySegPlanner
nnUNet_train 3d_fullres nnUNetTrainerV2_airwayAug 145 4 -p AirwaySegPlanner
```
You can also run these simultaneously if you have multiple GPUS:
```commandline
CUDA_VISIBLE_DEVICES=0 nnUNet_train 3d_fullres nnUNetTrainerV2_airwayAug 145 0 -p AirwaySegPlanner &
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2_airwayAug 145 1 -p AirwaySegPlanner &
CUDA_VISIBLE_DEVICES=2 nnUNet_train 3d_fullres nnUNetTrainerV2_airwayAug 145 2 -p AirwaySegPlanner &
CUDA_VISIBLE_DEVICES=3 nnUNet_train 3d_fullres nnUNetTrainerV2_airwayAug 145 3 -p AirwaySegPlanner &
CUDA_VISIBLE_DEVICES=4 nnUNet_train 3d_fullres nnUNetTrainerV2_airwayAug 145 4 -p AirwaySegPlanner &
wait
```
IMPORTANT: Wait with starting the last 4 folds (1, 2, 3, 4) until the training of the first fold has started using the 
GPU. This is related to unpacking the training data which can only be done by one nnU-Net process (it would result in 
read/write conflicts if several were to do this simultaneously).

# Inference/Prediction
Once training for all 5 folds is completed, you can use the trained models to perform predictions on new images. For that, 
the images must again be converted into nnU-Net format. See the [demo](#demo) for an example of converting a single file.
After conversion, prediction can be run like that:

```commandline
nnUNet_predict -i FOLDER_WITH_INPUT_IMAGES -o OUTPUT_FOLDER -t 145 -f 0 1 2 3 4 -m 3d_fullres -p AirwaySegPlanner -tr nnUNetTrainerV2_airwayAug --num_threads_preprocessing 1
```
The images are large. This will take a while. If you have multiple GPUs at your disposal, use the following (example assumes you have 4 GPUs):
```commandline
CUDA_VISIBLE_DEVICES=0 nnUNet_predict -i FOLDER_WITH_INPUT_IMAGES -o OUTPUT_FOLDER -t 145 -f 0 1 2 3 4 -m 3d_fullres -p AirwaySegPlanner -tr nnUNetTrainerV2_airwayAug --num_threads_preprocessing 1 --num_parts 4 --part_id 0 & 
CUDA_VISIBLE_DEVICES=1 nnUNet_predict -i FOLDER_WITH_INPUT_IMAGES -o OUTPUT_FOLDER -t 145 -f 0 1 2 3 4 -m 3d_fullres -p AirwaySegPlanner -tr nnUNetTrainerV2_airwayAug --num_threads_preprocessing 1 --num_parts 4 --part_id 1 & 
CUDA_VISIBLE_DEVICES=2 nnUNet_predict -i FOLDER_WITH_INPUT_IMAGES -o OUTPUT_FOLDER -t 145 -f 0 1 2 3 4 -m 3d_fullres -p AirwaySegPlanner -tr nnUNetTrainerV2_airwayAug --num_threads_preprocessing 1 --num_parts 4 --part_id 2 & 
CUDA_VISIBLE_DEVICES=3 nnUNet_predict -i FOLDER_WITH_INPUT_IMAGES -o OUTPUT_FOLDER -t 145 -f 0 1 2 3 4 -m 3d_fullres -p AirwaySegPlanner -tr nnUNetTrainerV2_airwayAug --num_threads_preprocessing 1 --num_parts 4 --part_id 3 &
wait 
```

Inference will need a lot of RAM! >=128GB is a must!

# Pretrained models
Pretrained models are available [here](https://zenodo.org/record/7892040). To use them, simply download the .zip file 
and install it with 
`nnUNet_install_pretrained_model_from_zip` (use -h for usage instructions). Then follow the instructions in 
[Inference/Prediction](#inferenceprediction).

# Acknowledgements
This is a joint project between [Helmholtz Imaging](http://helmholtz-imaging.de) 
(located at [DKFZ](https://www.dkfz.de/en/mic/index.php)) and [Helmholtz Munich](https://www.helmholtz-munich.de/en).

<img src="images/HI_Logo.png" height="100px" />
<img src="images/dkfz_logo.png" height="100px" />
<img src="images/Helmholtz_Munich_Logo.jpg" height="100px" />

