## Benchmarks
- :dog: means that model is initialized with imagenet pretrained weights.
- Results are presented using the format of `Rank1 (mAP)` in the following tables.
- CUHK03: detected, [new protocol (767/700)](https://github.com/zhunzhong07/person-re-ranking#the-new-trainingtesting-protocol-for-cuhk03).
- Classification layer is ignored when computing model size.

### Image person reid
| Model | # param (M) | Loss | Market1501 | CUHK03 | DukeMTMC-reID | MSMT17 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50:dog: | 23.5 | xent | [88.5 (71.3)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_xent_market1501.pth.tar) | [47.9 (46.8)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_xent_cuhk03.pth.tar) | [77.7 (58.8)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_xent_dukemtmcreid.pth.tar) | [63.4 (34.2)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_xent_msmt17.pth.tar) |
| DenseNet121:dog: | 7.0 | xent | [88.2 (69.2)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/densenet121_xent_market1501.pth.tar) | [41.0 (40.1)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/densenet121_xent_cuhk03.pth.tar) | [78.6 (58.5)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/densenet121_xent_dukemtmcreid.pth.tar) | [66.0 (34.6)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/densenet121_xent_msmt17.pth.tar) |
| DenseNet121:dog: | 7.0 | htri | 86.6 (70.8) | - | - | - |
| DenseNet121:dog: | 7.0 | xent+htri | 90.1 (74.0) | - | - | - |
| ResNet50M:dog: | 27.7 | xent | 89.0 (74.6) | 55.4 (52.7) | 81.0 (64.1) | 64.6 (35.9) |
| NasnetMobile:dog: | 4.2 | xent | 83.8 (64.9) | 42.4 (42.4) | 74.0 (53.7) | 57.1 (30.2) |
| SqueezeNet | 1.1 | xent | 72.2 (47.9) | 26.9 (25.8) | 58.8 (37.8) | 30.6 (13.0) |
| MobileNetV2 | 2.2 | xent | 84.2 (65.8) | 41.0 (40.3) | 73.2 (52.5) | 44.9 (21.1) |
| ShuffleNet | 0.9 | xent | 80.0 (58.4) | 31.9 (31.7) | 69.3 (46.8) | 39.6 (17.8) |
| HACNN | 2.9 | xent | [90.6 (75.3)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/hacnn_xent_market1501.pth.tar) | [48.0 (47.6)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/hacnn_xent_cuhk03.pth.tar) | [80.7 (64.4)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/hacnn_xent_dukemtmcreid.pth.tar) | [61.8 (34.6)](http://eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/hacnn_xent_msmt17.pth.tar) |