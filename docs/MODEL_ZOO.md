# Model Zoo

In general,
- results are presented in the format of *<Rank-1 (mAP)>*, unless specified otherwise.
- when computing FLOPs, only layers that are used at test time are considered (see `torchreid.utils.compute_model_complexity`).
- asterisk (\*) means the model is trained from scratch.
- model name denotes the key to build the model in `torchreid.models.build_model`.
- `combineall=True` means all images in the dataset are used for model training.
- for the cuhk03 dataset, we use the 767/700 split by [Zhong et al. CVPR'17](https://arxiv.org/abs/1701.08398).
- [label smoothing regularizer](https://arxiv.org/abs/1512.00567) is used in the softmax loss.

## ImageNet pretrained weights

| Model | Download |
| :--- | :---: |
| shufflenet | |
| mobilenetv2_x1_0 | |
| mobilenetv2_x1_4 | |
| mlfn | |
| osnet | |
| osnet_medium | |
| osnet_tiny | |
| osnet_verytiny | |
| osnet_ibn | |

## Same-domain ReID

| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance | market1501  | dukemtmcreid | msmt17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| resnet50 | 23.5 | 2.7 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [87.9 (70.4)](https://mega.nz/#!FKZjVKaZ!4v_FR8pTvuHoMQIKdstJ_YCsRrtZW2hwWxc-T0JIlHE) | [78.3 (58.9)](https://mega.nz/#!JPZjCYhK!YVJbE_4vTc8DX19Rt_FB77YY4BaEA1P6Xb5sNJGep2M) | [63.2 (33.9)](https://mega.nz/#!APAxDY4Z!Iou9x8s3ATdYS2SlK2oiJbHrhvlzH7F1gE2qjM-GJGw) |
| resnet50_fc512 | 24.6 | 4.1 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [90.8 (75.3)](https://mega.nz/#!EaZjhKyS!lBvD3vAJ4DOmElZkNa7gyPM1RE661GUd2v9kK84gSZE) | [81.0 (64.0)](https://mega.nz/#!lXYDSKZa!lumiXkY2H5Sm8gEgTWPBdWKv3ujy4zjrffjERaXkc9I) | [69.6 (38.4)](https://mega.nz/#!9PQTXIpL!iI5wgieTCn0Jm-pyg9RCu0RkH43pV3ntHhr1PeqSyT4) |
| mlfn | 32.5 | 2.8 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [90.1 (74.3)](https://mega.nz/#!kHQ3ESLT!NoGc8eHEBZOJZM19THh3DFfRBXIPXzM-sdLmF1mvTXA) | [81.1 (63.2)](https://mega.nz/#!8PQXUCaI!mJO1vD9tI739hkNBj2QWUt0VPcZ-s89fSMMGPPP1msc) | [66.4 (37.2)](https://mega.nz/#!paIXFQCS!W3ZGkxyF1idwvQzTRDE2p0DhNDki2SBJRfp7S_Cwphk) |
| hacnn<sup>*</sup> | 4.5 | 0.5 | softmax | (160, 64) | `random_flip`, `random_crop` | `euclidean` | [90.9 (75.6)](https://mega.nz/#!ULQXUQBK!S-8v_pR2xBD3ZpuY0I7Bqift-eX_V84gajHMDG6zUac) | [80.1 (63.2)](https://mega.nz/#!wPJTkAQR!XkKd39lsmBZMrCh3JjF6vnNafBZkouVIVdeBqQKdSzA) | [64.7 (37.2)](https://mega.nz/#!AXAziKjL!JtMwHz2UYy58gDMQLGakSmF3JOr72o8zmkqlQA-LIpQ) |
| mobilenetv2_x1_0 | 2.2 | 0.2 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [85.6 (67.3)](https://mega.nz/#!8KYTFAIB!3dL35WQLxSoTSClDTv0kxa81k3fh5hXmAWA4_a3qiOI) | [74.2 (54.7)](https://mega.nz/#!hbRXDSCL!YYgqJ6PVUf4clgtUuK2s5FRhYJdU3yTibLscwOTNnDk) | [57.4 (29.3)](https://mega.nz/#!5SJTmCYb!ZQ8O2MN9JF4-WDAeX04Xex1KyuBYQ_o2aoMIsTgQ748) |
| mobilenetv2_x1_4 | 4.3 | 0.4 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [87.0 (68.5)](https://mega.nz/#!4XZhEKCS!6lTuTRbHIWU5nzJzTPDGykA7sPME8_1ISGsUYFJXZWA) | [76.2 (55.8)](https://mega.nz/#!JbQVDIYQ!-7pnjIfpIDt1EoQOvpvuIEcTj3Qg8SE6o_3ZPGWrIcw) | [60.1 (31.5)](https://mega.nz/#!gOYDAQrK!sMJO7c_X4iIxoVfV_tXYdzeDJByPo5XkUjEN7Z2JTmM) |
| osnet | 2.2 | 0.98 | softmax | (256, 128) | `random_flip` | `euclidean` | 94.2 (82.6) | 87.0 (70.2) | 74.9 (43.8) |
| osnet_medium | 1.3 | 0.57 | softmax | (256, 128) | `random_flip` | `euclidean` | 93.7 (81.2) | 85.8 (69.8) | 72.8 (41.4) |
| osnet_tiny | 0.6 | 0.27 | softmax | (256, 128) | `random_flip` | `euclidean` | 92.5 (79.8) | 85.1 (67.4) | 69.7 (37.5) |
| osnet_verytiny | 0.2 | 0.08 | softmax | (256, 128) | `random_flip` | `euclidean` | 91.2 (75.0) | 82.0 (61.4) | 61.4 (29.5) |


## Cross-domain ReID

#### Market1501 -> DukeMTMC-reID


| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance  | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | Download |
| :--- | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| osnet_ibn | 2.2 | 0.98  | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 48.5 | 62.3 | 67.4 | 72.2 | 26.7 | model |


#### DukeMTMC-reID -> Market1501


| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance  | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | Download |
| :--- | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| osnet_ibn | 2.2 | 0.98  | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 57.7 | 73.7 | 80.0 | 84.8 | 26.1 | model |


#### MSMT17 -> Market1501, DukeMTMC-reID & CUHK03


| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance | msmt17 -> market1501 | msmt17 -> dukemtmcreid | msmt17 -> cuhk03 | Download |
| :--- | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| osnet_ibn | 2.2 | 0.98  | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 60.2 (29.9) | 59.9 (37.4) | 15.1 (14.1) | model |

#### MSMT17 (`combineall=True`) -> Market1501, DukeMTMC-reID & CUHK03

| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance | msmt17 -> market1501 | msmt17 -> dukemtmcreid | msmt17 -> cuhk03 | Download |
| :--- | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| resnet50 | 23.5 | 2.7 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 46.3 (22.8) | 52.3 (32.1) | 11.7 (13.1) | model |
| osnet | 2.2 | 0.98 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 66.6 (37.5) | 66.0 (45.3) | 21.0 (19.9) | model |
| osnet_medium | 1.3 | 0.57 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 63.6 (35.5) | 65.3 (44.5) | 20.0 (19.4) | model |
| osnet_tiny | 0.6 | 0.27 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 64.3 (34.9) | 65.2 (43.3) | 19.6 (19.2) | model |
| osnet_verytiny | 0.2 | 0.08 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 59.9 (31.0) | 61.5 (39.6) | 14.5 (14.5) | model |
| osnet_ibn | 2.2 | 0.98 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 66.5 (37.2) | 67.4 (45.6) | 22.0 (20.8) | model |
