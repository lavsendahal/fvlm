# Fine-grained Vision-language Pre-training for Enhanced CT Image Understanding

## Data processing

- Download the [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) dataset into the data folder.

- Download ImageNet pre-trained ViT weights from [link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth), and BiomedVLP-CXR-BERT-specialized text encoder from [link](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized), as used by CT-CLIP.

- Download the decomposed anatomy-wise descriptions from our provided supplementary materials [link](https://openreview.net/forum?id=nYpPAT4L3D&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)), and process the CT volume with the following commands.

  ```bash
  cd data
  python fix_data.py --split [train/valid]
  python generate_mask.py --split [train/valid]
  python resize.py --split [train/valid]
  python preprocess.py --split [train/valid]
  ```

  The processed results.

  ```bash
  |-- BiomedVLP-CXR-BERT
  |-- data
  |   |-- train
  |   |-- valid
  |   |-- train_fixed
  |   |-- valid_fixed
  |   |-- train_mask
  |   |-- valid_mask
  |   |-- resized_train_images
  |   |-- resized_train_masks
  |   |-- resized_valid_images
  |   |-- resized_valid_masks
  |   |-- processed_train_images
  |   |-- processed_train_masks
  |   |-- processed_valid_images
  |   |-- processed_valid_masks
  |   |-- multi_abnormality_labels
  |   |-- desc_info.json
  |   |-- conc_info.json
  |-- mae_pretrain_vit_base.pth
  ```



## Training

```shell
torchrun --nproc_per_node=4 train.py
```



## Evaluation

```bash
torchrun --nproc_per_node=4 eval.py
```

Then, you can calculate the metrics using the generated CSV file.

```bash
python calc_metrics.py --csv_file res/xxx.csv
```

## Citation
If you find this repository useful, please cite:

- **Bootstrapping Chest CT Image Understanding by Distilling Knowledge from X-ray Expert Models (CVPR 2024)**  
Weiwei Cao, Jianpeng Zhang, Yingda Xia, Tony CW Mok, Zi Li, Xianghua Ye, Le Lu, Jian Zheng, Yuxing Tang, Ling Zhang
