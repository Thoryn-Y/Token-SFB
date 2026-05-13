# TokenSFB: Token-Level Spatial-Frequency Fusion Block for Transformer-based Medical Image Segmentation

**TokenSFB: A Token-Level Spatial-Frequency Fusion Feature Enhancement Module for Transformer-based Medical Image Segmentation**

## 1. Introduction

TokenSFB is a plug-and-play, token-level spatial-frequency fusion feature enhancement module, natively designed to align with the token sequence computation paradigm of Transformer architectures (e.g., Swin-Unet, TransUNet). It addresses the incompatibility issues between existing spatial-frequency blocks (SFB) and Transformer backbones.

- **Spatial Branch**: Utilizes depthwise convolution to reinforce fine-grained local boundary details.
- **Frequency Branch**: Captures global context through FFT and frequency-domain attention, compensating for the shallow-layer limitations of window-based attention.
- **Adaptive Fusion**: A learnable gating mechanism dynamically balances the contribution of the two branches, while DropPath and LayerNorm ensure training stability.

On the Synapse multi-organ abdominal CT dataset and the ACDC cardiac MRI dataset, TokenSFB consistently improves segmentation accuracy and boundary quality for both pure Transformer and hybrid architectures, revealing an adaptation principle that spatial-frequency modules must align with the underlying computation paradigm.

## 2. Download Pre-trained Models

### 2.1 Swin Transformer Backbone for Swin-Unet (Swin-T)

- [Get the pre-trained model here](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Place the pre-trained Swin-T checkpoint into the folder `pretrained_ckpt/`.

### 2.2 Google Pre-trained ViT Model for TransUNet

- Available in the TransUNet [project directory](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing). Once obtained, locate the file at: `../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz`

## 3. Environment Setup

- Prepare a Python 3.10 environment, then install the required dependencies using the command `pip install -r requirements.txt`.

## 4. Dataset Preparation

- We utilize the datasets provided by the authors of TransUNet.

### 4.1 Synapse Multi-Organ Abdominal CT Dataset

- **Description**: Contrast-enhanced abdominal CT scans with annotations for eight abdominal organs (aorta, gallbladder, left kidney, right kidney, liver, pancreas, spleen, stomach). Commonly used to evaluate multi-organ segmentation performance under low-contrast and ambiguous boundary conditions.
- **Access**: [Download the processed dataset here](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)

### 4.2 ACDC Cardiac MRI Dataset

- **Description**: Cardiac MRI scans with annotations for the left ventricle, right ventricle, and myocardium. The distinct imaging modality compared to CT serves to validate cross-modality generalization.
- **Access**: [Download the processed dataset here](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4)

## 5. Train/Test

### Swin-Unet

- Train

```bash
sh train.sh 
# or
python train.py --output_dir your_output_dir --dataset Synapse --img_size 224 --batch_size 16 --max_epochs 150 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your_data_dir --n_class 9 --num_classes 9 --num_workers 0
```

- Test

```bash
sh test.sh 
# or
python test.py --output_dir your_output_dir --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your_data_dir --img_size 224 --batch_size 1 --num_classes 9 --n_class 9 --is_savenii
```

### TransUNet

- Train

```bash
python train.py --root_path your_data_dir --dataset Synapse --list_dir ./lists/lists_Synapse --vit_name R50-ViT-B_16 --num_classes 9 --batch_size 12 --base_lr 0.005 --max_epochs 150 --img_size 224 --n_gpu 1
```

- Test

```bash
python test.py --volume_path ../data/Synapse/test_vol_h5 --dataset Synapse --list_dir ./lists/lists_Synapse --vit_name R50-ViT-B_16 --num_classes 9 --batch_size 12 --base_lr 0.005 --max_epochs 150 --img_size 224 --is_savenii
```

## References

- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [TransUNet](https://github.com/Beckschen/TransUNet)

