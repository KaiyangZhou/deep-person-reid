# Model Zoo
- :dog: means that the model is initialized with imagenet pretrained weights.
- Results are presented in the format of **Rank1 (mAP)**.
- Classification layer is ignored when computing the model size.

## Image person reid
| Model | # param (M) | Loss | Input | market1501  | dukemtmcreid | msmt17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| resnet50_fc512<sup>:dog:</sup> | 24.6 | xent | (256, 128) | xx | xx | xx |
| mlfn<sup>:dog:</sup> | 32.5 | xent | (256, 128) | xx | xx | xx |
| hacnn | 3.7 | xent | (160, 64) | xx | xx | xx |
| squeezenet1_0_fc512 | 1.0 | xent | (256, 128) | xx | xx | xx |