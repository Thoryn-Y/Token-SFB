# TokenSFB: Token-Level Spatial-Frequency Fusion Block for Transformer-based Medical Image Segmentation

**"TokenSFB：面向Transformer医学影像分割的Token级空间-频域融合特征增强模块"**

## 1. 项目简介

TokenSFB 是一个**即插即用**的 Token 级空间-频域融合特征增强模块，旨在原生适配 Transformer 架构（如 Swin-Unet, TransUNet）的 Token 序列计算范式，解决现有空间-频域模块（SFB）与 Transformer 不兼容的问题。

- **空间分支**：利用深度卷积强化局部边界细节。
- **频域分支**：通过 FFT 与频域注意力捕获全局上下文，弥补窗口注意力的浅层局限。
- **自适应融合**：可学习门控参数动态平衡双分支贡献，配合 DropPath 与 LayerNorm 保障训练稳定。

在 Synapse 腹部多器官 CT 和 ACDC 心脏 MRI 数据集上，TokenSFB 可稳定提升纯 Transformer 及混合架构的分割精度与边界质量，并揭示了“空间-频域模块需匹配底层计算范式”的适配规律。

## 2. 下载预训练模型

### 2.1 Swin-Unet 使用的 swin 变换器模型（Swin-T）

- [在此链接获取预训练模型](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing)：把预训练的 Swin-T 放进文件夹 “pretrained\_ckpt/”

### 2.2 TransUNet 使用的谷歌预训练的ViT模型

- 可以在 TransUNet 的[项目文件夹](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing)中获取获取。提取后，在以下位置找到该文件：../model/vit\_checkpoint/imagenet21k/R50+ViT-B\_16.npz

## 3. 环境配置

- 请准备一个带有 python=3.10 的环境，然后使用命令“pip install -r requirements.txt”来处理依赖。

## 4. 数据集准备

- 我们使用的数据集由 TransUnet 的作者提供。

### 4.1 Synapse 多器官腹部 CT 数据集

- **描述**：腹部增强 CT 扫描，标注了 8 个腹部器官（主动脉、胆囊、左肾、右肾、肝脏、胰腺、脾脏、胃）。常用于评估多器官分割在低对比度、边界模糊场景下的性能。
- **获取方式**：[在此链接获取处理数据](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)

### 4.2 ACDC 心脏 MRI 数据集

- **描述**：心脏 MRI 扫描，标注了左心室、右心室和心肌。成像模态与 CT 差异显著，用于验证跨模态泛化性。
- **获取方式**：[在此链接获取处理数据](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4)

## 5. 训练/测试

### Swin-Unet

- 训练

```bash
sh train.sh 
# or
python train.py --output_dir your_output_dir --dataset Synapse --img_size 224 --batch_size 16 --max_epochs 150 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your_data_dir --n_class 9 --num_classes 9 --num_workers 0
```

- 测试

```bash
sh test.sh 
# or
python test.py --output_dir your_output_dir --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your_data_dir --img_size 224 --batch_size 1 --num_classes 9 --n_class 9 --is_savenii
```

### TransUNet

- 训练

```bash
python train.py --root_path your_data_dir --dataset Synapse --list_dir ./lists/lists_Synapse --vit_name R50-ViT-B_16 --num_classes 9 --batch_size 12 --base_lr 0.005 --max_epochs 150 --img_size 224 --n_gpu 1
```

- 测试

```bash
python test.py --volume_path ../data/Synapse/test_vol_h5 --dataset Synapse --list_dir ./lists/lists_Synapse --vit_name R50-ViT-B_16 --num_classes 9 --batch_size 12 --base_lr 0.005 --max_epochs 150 --img_size 224 --is_savenii
```

## 参考文献

- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [TransUNet](https://github.com/Beckschen/TransUNet)

