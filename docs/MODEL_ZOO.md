# Model Zoo

- Results are presented in the format of *<Rank-1 (mAP)>*.
- When computing model size and FLOPs, only layers that are used at test time are considered (see `torchreid.utils.compute_model_complexity`).
- Asterisk (\*) means the model is trained from scratch.
- `combineall=True` means all images in the dataset are used for model training.


## ImageNet pretrained models


| Model | Download |
| :--- | :---: |
| shufflenet | [model](https://mega.nz/#!RDpUlQCY!tr_5xBEkelzDjveIYBBcGcovNCOrgfiJO9kiidz9fZM) |
| mobilenetv2_x1_0 | [model](https://mega.nz/#!NKp2wAIA!1NH1pbNzY_M2hVk_hdsxNM1NUOWvvGPHhaNr-fASF6c) |
| mobilenetv2_x1_4 | [model](https://mega.nz/#!RGhgEIwS!xN2s2ZdyqI6vQ3EwgmRXLEW3khr9tpXg96G9SUJugGk) |
| mlfn | [model](https://mega.nz/#!YHxAhaxC!yu9E6zWl0x5zscSouTdbZu8gdFFytDdl-RAdD2DEfpk) |
| osnet_x1_0 | [model](https://mega.nz/#!YK5GRARL!F90NsNB2XHjXGZFC3Lrw1GMic0oMw4fnfuDUnSrPAYM) |
| osnet_x0_75 | [model](https://mega.nz/#!NPxilYBA!Se414Wtts__7eY6J5FIrowynvjUUG7a8Z5zUPfJN33s) |
| osnet_x0_5 | [model](https://mega.nz/#!NO4ihQSJ!oMIRSZ0HlJF_8FKUbXT8Ei0vzH0xUYs5tWaf_KLrODg) |
| osnet_x0_25 | [model](https://mega.nz/#!IDwQwaxT!TbQ_33gPK-ZchPFTf43UMc45rlNKWiWMqH4rTXB1T7k) |
| osnet_ibn_x1_0 | [model](https://mega.nz/#!8Wo2kSDR!bNvgu4V0VkCQp_L2ZUDaudYKYRCkkSNdzcA1CcZGZTE) |
| osnet_ain_x1_0 | [model](https://drive.google.com/open?id=1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo) |


## Same-domain ReID


| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance | market1501  | dukemtmcreid | msmt17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| resnet50 | 23.5 | 2.7 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [87.9 (70.4)](https://mega.nz/#!FKZjVKaZ!4v_FR8pTvuHoMQIKdstJ_YCsRrtZW2hwWxc-T0JIlHE) | [78.3 (58.9)](https://mega.nz/#!JPZjCYhK!YVJbE_4vTc8DX19Rt_FB77YY4BaEA1P6Xb5sNJGep2M) | [63.2 (33.9)](https://mega.nz/#!APAxDY4Z!Iou9x8s3ATdYS2SlK2oiJbHrhvlzH7F1gE2qjM-GJGw) |
| resnet50_fc512 | 24.6 | 4.1 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [90.8 (75.3)](https://mega.nz/#!EaZjhKyS!lBvD3vAJ4DOmElZkNa7gyPM1RE661GUd2v9kK84gSZE) | [81.0 (64.0)](https://mega.nz/#!lXYDSKZa!lumiXkY2H5Sm8gEgTWPBdWKv3ujy4zjrffjERaXkc9I) | [69.6 (38.4)](https://mega.nz/#!9PQTXIpL!iI5wgieTCn0Jm-pyg9RCu0RkH43pV3ntHhr1PeqSyT4) |
| mlfn | 32.5 | 2.8 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [90.1 (74.3)](https://mega.nz/#!kHQ3ESLT!NoGc8eHEBZOJZM19THh3DFfRBXIPXzM-sdLmF1mvTXA) | [81.1 (63.2)](https://mega.nz/#!8PQXUCaI!mJO1vD9tI739hkNBj2QWUt0VPcZ-s89fSMMGPPP1msc) | [66.4 (37.2)](https://mega.nz/#!paIXFQCS!W3ZGkxyF1idwvQzTRDE2p0DhNDki2SBJRfp7S_Cwphk) |
| hacnn<sup>*</sup> | 4.5 | 0.5 | softmax | (160, 64) | `random_flip`, `random_crop` | `euclidean` | [90.9 (75.6)](https://mega.nz/#!ULQXUQBK!S-8v_pR2xBD3ZpuY0I7Bqift-eX_V84gajHMDG6zUac) | [80.1 (63.2)](https://mega.nz/#!wPJTkAQR!XkKd39lsmBZMrCh3JjF6vnNafBZkouVIVdeBqQKdSzA) | [64.7 (37.2)](https://mega.nz/#!AXAziKjL!JtMwHz2UYy58gDMQLGakSmF3JOr72o8zmkqlQA-LIpQ) |
| mobilenetv2_x1_0 | 2.2 | 0.2 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [85.6 (67.3)](https://mega.nz/#!8KYTFAIB!3dL35WQLxSoTSClDTv0kxa81k3fh5hXmAWA4_a3qiOI) | [74.2 (54.7)](https://mega.nz/#!hbRXDSCL!YYgqJ6PVUf4clgtUuK2s5FRhYJdU3yTibLscwOTNnDk) | [57.4 (29.3)](https://mega.nz/#!5SJTmCYb!ZQ8O2MN9JF4-WDAeX04Xex1KyuBYQ_o2aoMIsTgQ748) |
| mobilenetv2_x1_4 | 4.3 | 0.4 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [87.0 (68.5)](https://mega.nz/#!4XZhEKCS!6lTuTRbHIWU5nzJzTPDGykA7sPME8_1ISGsUYFJXZWA) | [76.2 (55.8)](https://mega.nz/#!JbQVDIYQ!-7pnjIfpIDt1EoQOvpvuIEcTj3Qg8SE6o_3ZPGWrIcw) | [60.1 (31.5)](https://mega.nz/#!gOYDAQrK!sMJO7c_X4iIxoVfV_tXYdzeDJByPo5XkUjEN7Z2JTmM) |
| osnet_x1_0 | 2.2 | 0.98 | softmax | (256, 128) | `random_flip` | `euclidean` | [94.2 (82.6)](https://mega.nz/#!hLoyTSba!fqt7GcKrHJhwe9BtuK0ozgVAQcrlMG8Pm6JsSfr5HEI) | [87.0 (70.2)](https://mega.nz/#!ETwGhQYB!h2gHN-H3J4X4WqcJXy2b0pPKl28paydkiS-PDHsEgPM) | [74.9 (43.8)](https://mega.nz/#!hWxE2aJA!NGcxu5uYH1qI6DfBTu0KFoi_NfoA0TJcBFW-g43pC0I) |
| osnet_x0_75 | 1.3 | 0.57 | softmax | (256, 128) | `random_flip` | `euclidean` | [93.7 (81.2)](https://mega.nz/#!JO4WAaJa!nQuoqZnYfy0xu7vs2mp28AFceya-ZhrXTry837jvoDQ) | [85.8 (69.8)](https://mega.nz/#!lOgkEIoI!fQ5vuYIABIOcRxF-OK-6YxtEufWhyVkYkGB4qPoRYJ4) | [72.8 (41.4)](https://mega.nz/#!0exGXI5a!rxtzBayyRK0on0HFq9XO0UtWEBhbV86dFitljhjeWcs) |
| osnet_x0_5 | 0.6 | 0.27 | softmax | (256, 128) | `random_flip` | `euclidean` | [92.5 (79.8)](https://mega.nz/#!QCx0RArD!hqz3Mh0Iif5d8PpQW0frxa-Tepn2a2g24aei7du4MFs) | [85.1 (67.4)](https://mega.nz/#!QTxCDIbT!eOZxj4dHl0uFnjKEB-J3YBY98blXZvppgWGA3CGa-tk) | [69.7 (37.5)](https://mega.nz/#!ETpiECDa!CCkq4JryztHqgw7spL5zDw0usJpAfEsSd5gPlkMufCc) |
| osnet_x0_25 | 0.2 | 0.08 | softmax | (256, 128) | `random_flip` | `euclidean` | [91.2 (75.0)](https://mega.nz/#!VWxCgSqY!Q4WaQ3j9D7HMhK3jsbvMuwaZ7yBY80T2Zj5V8JAlAKU) | [82.0 (61.4)](https://mega.nz/#!5TpwnATK!UvU_Asdy_aJ9SNzuvqhEFoemxSSB8vm_Gm8Xe03jqiA) | [61.4 (29.5)](https://mega.nz/#!AWgE3SzD!DngUaNyA7VIqOd2gq10Aty_-ER0CmG0xTJLHLj6_36g) |


## Cross-domain ReID

#### Market1501 -> DukeMTMC-reID


| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance  | Rank-1 | Rank-5 | Rank-10 | mAP | Download |
| :--- | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| osnet_ibn_x1_0 | 2.2 | 0.98  | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 48.5 | 62.3 | 67.4 | 26.7 | [model](https://mega.nz/#!wXwGxKxK!f8EMk8hBt6AjxU3JIPGMFSMvX7j-Nt5Lp1Gpbqso1Ts) |
| osnet_ain_x1_0 | 2.2 | 0.98  | softmax | (256, 128) | `random_flip`, `color_jitter` | `cosine` | 52.4 | 66.1 | 71.2 | 30.5 | [model](https://mega.nz/#!QLJE2CRI!FXYc3Vm6Y5Scwx0xvRwBJxId56kf06fIXNLwA_b_1FE) |


#### DukeMTMC-reID -> Market1501


| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance  | Rank-1 | Rank-5 | Rank-10 | mAP | Download |
| :--- | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| osnet_ibn_x1_0 | 2.2 | 0.98  | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 57.7 | 73.7 | 80.0 | 26.1 | [model](https://mega.nz/#!FD4WEKJS!ZGgI-2IwVuX6re09xylChR03o6Dkjpi6KSebrbS0fAA) |
| osnet_ain_x1_0 | 2.2 | 0.98  | softmax | (256, 128) | `random_flip`, `color_jitter` | `cosine` | 61.0 | 77.0 | 82.5 | 30.6 | [model](https://mega.nz/#!4PBQlCCL!9yMHu1WyyBVxqssubLAEyoEfHUiNP4Ggg5On0nCX2S4) |


#### MSMT17 (`combineall=True`) -> Market1501 & DukeMTMC-reID


| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance | msmt17 -> market1501 | msmt17 -> dukemtmcreid | Download |
| :--- | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: |
| resnet50 | 23.5 | 2.7 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 46.3 (22.8) | 52.3 (32.1) | [model](https://mega.nz/#!VTpkWSbS!Y8gDnmg7u-sPwnZDhWXrtZNYOj7UYL4QzZkhDf1qWW4) |
| osnet_x1_0 | 2.2 | 0.98 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 66.6 (37.5) | 66.0 (45.3) | [model](https://mega.nz/#!MepG3QRC!Lb-C9d7rdS_YJjGSoJ5cRlzjYcP28P_1Cm5S5WSslW0) |
| osnet_x0_75 | 1.3 | 0.57 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 63.6 (35.5) | 65.3 (44.5) | [model](https://mega.nz/#!tO4WDagL!8Tl6kdJWRXRHQb16GeUHR008tJqW3N7_3fyVMu-LcKM) |
| osnet_x0_5 | 0.6 | 0.27 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 64.3 (34.9) | 65.2 (43.3) | [model](https://mega.nz/#!papSWQhY!IId-QfcHj7nXQ_muUubgv9_n0SsnZzarmb5mQgcMv74) |
| osnet_x0_25 | 0.2 | 0.08 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 59.9 (31.0) | 61.5 (39.6) | [model](https://mega.nz/#!QCoE0Kpa!BITLANumgjiR68TUFteL__N_RIoDKkL0M5Bl3Q8LC3U) |
| osnet_ibn_x1_0 | 2.2 | 0.98 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 66.5 (37.2) | 67.4 (45.6) | [model](https://mega.nz/#!dL4Q2K5B!ZdHQ_X_rs2T-xmggigM5YvzJhmT1orkr6aQ1_fHgunM) |
| osnet_ain_x1_0 | 2.2 | 0.98 | softmax | (256, 128) | `random_flip`, `color_jitter` | `cosine` | 70.1 (43.3) | 71.1 (52.7) | [model](https://mega.nz/#!YTZFnSJY!wlbo_5oa2TpDAGyWCTKTX1hh4d6DvJhh_RUA2z6i_so) |
