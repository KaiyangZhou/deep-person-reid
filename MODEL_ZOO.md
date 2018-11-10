# Model Zoo
- :dog: means that the model is initialized with [imagenet](http://www.image-net.org/) pretrained weights.
- Results are presented in the format of **Rank1 (mAP)**.
- Classification layer is ignored when computing the model size.
- Unless specified otherwise, the following [data augmentation techniques](torchreid/transforms.py) are used: (1) Random2DTranslation, and (2) RandomHorizontalFlip.

## Image person reid
| Model | # param (M) | Loss | Input | market1501  | dukemtmcreid | msmt17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| resnet50<sup>:dog:</sup> |  | xent | (256, 128) | xx | xx | xx |
| resnet50_fc512<sup>:dog:</sup> | 24.6 | xent | (256, 128) | xx | xx | xx |
| densenet121_fc512<sup>:dog:</sup> | 7.5 | xent | (256, 128) | xx | xx | xx |
| se_resnet50_fc512<sup>:dog:</sup> | 27.1 | xent | (256, 128) | xx | xx | xx |
| squeezenet1_0_fc512<sup>:dog:</sup> | 1.0 | xent | (256, 128) | xx | xx | xx |
| resnet50mid<sup>:dog:</sup> | 27.7 | xent | (256, 128) | xx | xx | xx |
| mlfn<sup>:dog:</sup> | 32.5 | xent | (256, 128) | xx | xx | xx |
| hacnn<sup></sup> | 3.7 | xent | (160, 64) | xx | xx | xx |
| pcb_p6<sup>:dog:</sup> | 24.0 | xent | (384, 128) | xx | xx | xx |