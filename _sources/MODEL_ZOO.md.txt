# Model Zoo
- Results are presented in the format of *<Rank-1 (mAP)>*.
- When computing FLOPs, only layers that are used at test time are considered (see `torchreid.utils.compute_model_complexity`).
- Unless specified otherwise, only `Random2DTranslation` and `RandomHorizontalFlip` are used for data augmentation.
- Asterisk (\*) means the model is trained from scratch.


| Model | # Param (10^6) | GFLOPs | Loss | Input | market1501  | dukemtmcreid | msmt17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| resnet50 | 23.5 | 2.7 | softmax | (256, 128) | [87.9 (70.4)](https://mega.nz/#!FKZjVKaZ!4v_FR8pTvuHoMQIKdstJ_YCsRrtZW2hwWxc-T0JIlHE) | [78.3 (58.9)](https://mega.nz/#!JPZjCYhK!YVJbE_4vTc8DX19Rt_FB77YY4BaEA1P6Xb5sNJGep2M) | [63.2 (33.9)](https://mega.nz/#!APAxDY4Z!Iou9x8s3ATdYS2SlK2oiJbHrhvlzH7F1gE2qjM-GJGw) |
| resnet50_fc512 | 24.6 | 4.1 | softmax | (256, 128) | [90.8 (75.3)](https://mega.nz/#!EaZjhKyS!lBvD3vAJ4DOmElZkNa7gyPM1RE661GUd2v9kK84gSZE) | [81.0 (64.0)](https://mega.nz/#!lXYDSKZa!lumiXkY2H5Sm8gEgTWPBdWKv3ujy4zjrffjERaXkc9I) | [69.6 (38.4)](https://mega.nz/#!9PQTXIpL!iI5wgieTCn0Jm-pyg9RCu0RkH43pV3ntHhr1PeqSyT4) |
| mlfn | 32.5 | 2.8 | softmax | (256, 128) | [90.1 (74.3)](https://mega.nz/#!kHQ3ESLT!NoGc8eHEBZOJZM19THh3DFfRBXIPXzM-sdLmF1mvTXA) | [81.1 (63.2)](https://mega.nz/#!8PQXUCaI!mJO1vD9tI739hkNBj2QWUt0VPcZ-s89fSMMGPPP1msc) | [66.4 (37.2)](https://mega.nz/#!paIXFQCS!W3ZGkxyF1idwvQzTRDE2p0DhNDki2SBJRfp7S_Cwphk) |
| hacnn<sup>*</sup> | 4.5 | 0.5 | softmax | (160, 64) | [90.9 (75.6)](https://mega.nz/#!ULQXUQBK!S-8v_pR2xBD3ZpuY0I7Bqift-eX_V84gajHMDG6zUac) | [80.1 (63.2)](https://mega.nz/#!wPJTkAQR!XkKd39lsmBZMrCh3JjF6vnNafBZkouVIVdeBqQKdSzA) | [64.7 (37.2)](https://mega.nz/#!AXAziKjL!JtMwHz2UYy58gDMQLGakSmF3JOr72o8zmkqlQA-LIpQ) |
| mobilenetv2_1dot0 | 2.2 | 0.2 | softmax | (256, 128) | [85.6 (67.3)](https://mega.nz/#!8KYTFAIB!3dL35WQLxSoTSClDTv0kxa81k3fh5hXmAWA4_a3qiOI) | [74.2 (54.7)](https://mega.nz/#!hbRXDSCL!YYgqJ6PVUf4clgtUuK2s5FRhYJdU3yTibLscwOTNnDk) | [57.4 (29.3)](https://mega.nz/#!5SJTmCYb!ZQ8O2MN9JF4-WDAeX04Xex1KyuBYQ_o2aoMIsTgQ748) |
| mobilenetv2_1dot4 | 4.3 | 0.4 | softmax | (256, 128) | [87.0 (68.5)](https://mega.nz/#!4XZhEKCS!6lTuTRbHIWU5nzJzTPDGykA7sPME8_1ISGsUYFJXZWA) | [76.2 (55.8)](https://mega.nz/#!JbQVDIYQ!-7pnjIfpIDt1EoQOvpvuIEcTj3Qg8SE6o_3ZPGWrIcw) | [60.1 (31.5)](https://mega.nz/#!gOYDAQrK!sMJO7c_X4iIxoVfV_tXYdzeDJByPo5XkUjEN7Z2JTmM) |
