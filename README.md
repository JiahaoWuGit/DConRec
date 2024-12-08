# DConRec (Dataset Condensation for Recommendation)

Also, named "Condensing Pre-augmented Recommendation Data via Lightweight Policy Gradient Estimation"

This is the official PyTorch implementation for the [paper](https://arxiv.org/abs/2310.01038):
> Jiahao Wu, Wenqi Fan,Jingfan Chen, Shengcai Liu, Qijiong Liu, Rui He, Qing Li, Ke Tang. Dataset Condensation for Recommendation. TKDE

## Overview

We propose a lightweight condensation framework tailored for recommendation (DConRec), focusing on condensing user-item historical interaction sets. Specifically, we model the discrete user-item interactions via a probabilistic approach and design a pre-augmentation module to incorporate the potential preferences of users into the condensed datasets. While the substantial size of datasets leads to costly optimization, we propose a lightweight policy gradient estimation to accelerate the data synthesis.


## Requirements

```
recbole==1.0.0
python==3.7.15
pytorch==1.12.1
faiss-gpu==1.7.3
cudatoolkit==10.1
```

## Quick Start

```bash
python main.py --dataset ml-1m
```

You can replace `ml-1m` to `dianping`, `ciao`, `gowalla-merged`,  or `ml-20m` to reproduce the results reported in our paper. For more advanced options, you can refer to the commands recorded in `run.sh`.

## Datasets

Datasets can be downloaded [here](https://drive.google.com/file/d/1haTPUFh7xfaWZfnP4crh6VfYTLErlrSy/view?usp=sharing).

## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

Please cite the following paper as the references if you use our codes.

```
@article{DConRec2024wu,
  author={Wu, Jiahao and Fan, Wenqi and Chen, Jingfan and Liu, Shengcai and Liu, Qijiong and He, Rui and Li, Qing and Tang, Ke},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Condensing Pre-augmented Recommendation Data via Lightweight Policy Gradient Estimation}, 
  year={2024},
  pages={1-11}}
```
