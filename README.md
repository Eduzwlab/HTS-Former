# HTS-Former
## A hybrid two-stage weakly supervised learning framework for molecular subtypes of prostate cancer using histopathological slides
The study presents a hybrid two-stage approach to predict molecular subtypes of prostate cancer from H&E histology slides.


!["HTS-Former"](./assets/HTS-Former.png)
## Contents
- [Pre-requisites and Environmen](#Pre-requisites-and-Environmen)
- [Data Preparation](#Data-Preparation)
- [Available Models](#Available Models)
- [K-fold Cross Validation](#K-fold-Cross-Validation)
## Pre-requisites and Environment
### Pre-requisites
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 3090) 
* Python (3.8.19), OpenCV (4.9.0.80), Openslide-python (1.3.1) and Pytorch (2.2.2)

### Environment Configuration
1. Create a virtual environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).
   ```
   $ conda create -n  python=3.8.19
   $ conda activate htsfomer
   $ conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.4 -c pytorch -c nvidia
   ```

2. To try out the Python code and set up environment, please activate the `htsfomer` environment first:

    ``` shell
    $ conda activate htsfomer
    $ cd htsfomer/
    ```
3. For ease of use, you can just set up the environment and run the following:
   ``` shell
   $ pip install -r requirements.txt
   ```

## Data Preparation

we follow the CLAM's WSI processing solution (https://github.com/mahmoodlab/CLAM)

```bash
# WSI Segmentation and Patching
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --preset bwh_biopsy.csv --seg --patch --stitch

# Feature Extraction
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs
```

## Available Models
- MaxPooling
- MeanPooling
- ABMIL
- DSMIL
- ATTMIL
- CLAM
- DTFD
- TRANSMIL
- S4MIL
- MAMBAMIL
- HTS-Former

```
## K-fold Cross Validation
After data preparation, HTS-Former can be trained and tested in a k-fold cross-validation by calling:
``` shell
$ CUDA_VISIBLE_DEVICES=0 python main.py
```
