# UkraineAP10

## Image preprocessing
Downloading and preprocessing Sentinel-2 imagery data.
- **`download_s2_images_ukr.js`**: Google Earth Engine (GEE) code for downloading and preprocessing Sentinel-2 imagery over Ukraine in 2023

## Sample preparation
Preparing samples for the model.
- **`label_to_mask.py`**: Converts agricultural parcel vector labels into binary masks.  
- **`clip_small_image_mask.py`**: Crops both images and masks into 256×256 patches with a 50% overlap.  
- **`get_boundary_dist.py`**: Generates two auxiliary task targets from the mask: boundary maps and distance transforms.

## HBGNet 2.0
Code for training the HBGNet 2.0 model.
- **Pretrained weights**:  
  The PVTv2 backbone pretrained on ImageNet can be downloaded from:  
  [https://drive.google.com/file/d/1uzeVfA4gEQ772vzLntnkqvWePSw84F6y/view?usp=sharing](https://drive.google.com/file/d/1uzeVfA4gEQ772vzLntnkqvWePSw84F6y/view?usp=sharing)
- **`dataset.py`**: Loads training data for the model.  
- **`HBGNet.py`**: Model architecture (see also: [https://github.com/NanNanmei/HBGNet](https://github.com/NanNanmei/HBGNet)).  
- **`losses.py`**: Implements an adaptive weighted loss function that accounts for task uncertainty.  
- **`train_HBGNet_improve.py`**: Main script for training HBGNet 2.0.  
- **`utils.py`**: Utility functions required during training.

## Model Inference
Code for sliding-window prediction on Sentinel-2 imagery over Ukraine.
- **`predict_a_big_image_HBGNet.py`**: Conducts block-wise inference on Ukrainian Sentinel-2 imagery to generate a nationwide agricultural parcel prediction map.

## Result Optimization
Post-processing scripts for refining model inference outputs.
- **`parcel_mask_water_glad.py`**: GEE script that applies a water mask using the GLAD global surface water data to remove water bodies from parcel predictions.  
- **`remove_small_parcels.py`**: Performs morphological opening on the water-masked results to eliminate small, spurious parcel segments.
