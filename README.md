# DConRec (Dataset Condensation for Recommendation)

Also, named "Condensing Pre-augmented Recommendation Data via Lightweight Policy Gradient Estimation"

This is the official PyTorch implementation for the [paper](https://arxiv.org/abs/2310.01038):
> Jiahao Wu, Wenqi Fan,Jingfan Chen, Shengcai Liu, Qijiong Liu, Rui He, Qing Li, Ke Tang. Dataset Condensation for Recommendation.

## Overview

We propose a lightweight condensation framework tailored for recommendation (DConRec), focusing on condensing user-item historical interaction sets. Specifically, we model the discrete user-item interactions via a probabilistic approach and design a pre-augmentation module to incorporate the potential preferences of users into the condensed datasets. While the substantial size of datasets leads to costly optimization, we propose a lightweight policy gradient estimation to accelerate the data synthesis.

<div  align="center"> 
<img src="asset/intro.png" style="width: 75%"/>
</div>

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

Please cite the following papers as the references if you use our codes.

```
@article{DConRec2024wu,
    author={Jiahao Wu, Wenqi Fan, Jingfan Chen, Shengcai Liu, Qijiong Liu, Rui He, Qing Li, Ke Tang},
    title={Dataset Condensation for Recommendation},
    journal={{TKDE}},
    year={2024},
}
```
