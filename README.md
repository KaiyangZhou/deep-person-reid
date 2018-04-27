# deep-person-reid
This repo contains [PyTorch](http://pytorch.org/) implementations of deep person re-identification models.

We support
- multi-GPU training.
- both image-based and video-based reid.
- unified interface for different reid models.
- end-to-end training and evaluation.
- standard splits used by most papers.
- download of trained models.

## Updates
- Apr 2018: Added [Harmonious Attention CNN (CVPR'18)](https://arxiv.org/abs/1802.08122). We achieved Rank-1 42.4% (vs. 41.7% in the paper) on CUHK03 (Detected) by training from scratch. The result can be reproduced by `python train_img_model_xent.py -d cuhk03 -a hacnn --save-dir log/hacnn-xent-cuhk03 --height 160 --width 64 --max-epoch 500 --stepsize -1 --eval-step 50`.
- Apr 2018: Code upgraded to pytorch 0.4.0.
- Apr 2018: Added [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html). Models are [available](https://github.com/KaiyangZhou/deep-person-reid#cuhk03-detected-new-protocol-767700).
- Apr 2018: Added [iLIDS-VID](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html) and [PRID-2011](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/). Models are [available](https://github.com/KaiyangZhou/deep-person-reid#video-person-reid).
- Mar 2018: Added argument `--htri-only` to `train_img_model_xent_htri.py` and `train_vid_model_xent_htri.py`. If this argument is true, only `htri` [4] is used for training. See [here](https://github.com/KaiyangZhou/deep-person-reid/blob/master/train_img_model_xent_htri.py#L189) for detailed changes.
- Mar 2018: Added [Multi-scale Deep CNN (ICCV'17)](https://arxiv.org/abs/1709.05165) [10] with slight modifications: (a) Input size is (256, 128) instead of (160, 60); (b) We add an average pooling layer after the last conv feature maps. (c) We train the network with our strategy. Model trained from scratch on Market1501 is [available](https://github.com/KaiyangZhou/deep-person-reid#results).
- Mar 2018: Added [center loss (ECCV'16)](https://github.com/KaiyangZhou/pytorch-center-loss) [9] and the trained model weights.

## Dependencies
- [PyTorch](http://pytorch.org/) (0.4.0)
- [torchvision](https://github.com/pytorch/vision/)

## Install
1. `cd` to the folder where you want to download this repo.
2. run `git clone https://github.com/KaiyangZhou/deep-person-reid`.

## Prepare data
Create a directory to store reid datasets under this repo via
```bash
cd deep-person-reid/
mkdir data/
```

Please follow the instructions below to prepare each dataset.

**Market1501** [7]:
1. Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html.
2. Extract dataset and rename to `market1501`. The data structure would look like:
```
market1501/
    bounding_box_test/
    bounding_box_train/
    ...
```
3. Use `-d market1501` when running the training code.

**CUHK03** [13]:
1. Create a folder named `cuhk03/` under `data/`.
2. Download dataset to `data/cuhk03/` from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html and extract `cuhk03_release.zip`, so you will have `data/cuhk03/cuhk03_release`.
3. Download new split [14] from [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03). What you need are `cuhk03_new_protocol_config_detected.mat` and `cuhk03_new_protocol_config_labeled.mat`. Put these two mat files under `data/cuhk03`. Finally, the data structure would look like
```
cuhk03/
    cuhk03_release/
    cuhk03_new_protocol_config_detected.mat
    cuhk03_new_protocol_config_labeled.mat
    ...
```
4. Use `-d cuhk03` when running the training code. In default mode, we use new split (767/700). If you wanna use the original splits (1367/100) created by [13], specify `--cuhk03-classic-split`. As [13] computes CMC differently from Market1501, you might need to specify `--use-metric-cuhk03` for fair comparison with their method. In addition, we support both `labeled` and `detected` modes. The default mode loads `detected` images. Specify `--cuhk03-labeled` if you wanna train and test on `labeled` images.

**MARS** [8]:
1. Create a directory named `mars/` under `data/`.
2. Download dataset to `data/mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract `bbox_train.zip` and `bbox_test.zip`.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars` (we want to follow the standard split in [8]). The data structure would look like:
```
mars/
    bbox_test/
    bbox_train/
    info/
```
5. Use `-d mars` when running the training code.

**iLIDS-VID** [11]:
1. The code supports automatic download and formatting. Simple use `-d ilidsvid` when running the training code. The data structure would look like:
```
ilids-vid/
    i-LIDS-VID/
    train-test people splits/
    splits.json
```

**PRID** [12]:
1. Under `data/`, do `mkdir prid2011` to create a directory.
2. Download dataset from https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/ and extract it under `data/prid2011`.
3. Download the split created by [iLIDS-VID](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html) from [here](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/datasets/prid2011/splits_prid2011.json), and put it in `data/prid2011/`. We follow [11] and use 178 persons whose sequences are more than a threshold so that results on this dataset can be fairly compared with other approaches. The data structure would look like:
```
prid2011/
    splits_prid2011.json
    prid_2011/
        multi_shot/
        single_shot/
        readme.txt
```
4. Use `-d prid` when running the training code.

## Dataset loaders
These are implemented in `dataset_loader.py` where we have two main classes that subclass [torch.utils.data.Dataset](http://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html#Dataset):
* [ImageDataset](https://github.com/KaiyangZhou/deep-person-reid/blob/master/dataset_loader.py#L22): processes image-based person reid datasets.
* [VideoDataset](https://github.com/KaiyangZhou/deep-person-reid/blob/master/dataset_loader.py#L38): processes video-based person reid datasets.

These two classes are used for [torch.utils.data.DataLoader](http://pytorch.org/docs/master/_modules/torch/utils/data/dataloader.html#DataLoader) that can provide batched data. Data loader wich `ImageDataset` outputs batch data of `(batch, channel, height, width)`, while data loader with `VideoDataset` outputs batch data of `(batch, sequence, channel, height, width)`.

## Models
* `models/ResNet.py`: ResNet50 [1], ResNet50M [2].
* `models/DenseNet.py`: DenseNet121 [3].
* `models/MuDeep.py`: MuDeep [10]. 
* `models/HACNN.py`: HACNN [15]

## Loss functions
* `xent`: cross entropy + label smoothing regularizer [5].
* `htri`: triplet loss with hard positive/negative mining [4] .
* `cent`: center loss [9].

Optimizers are wrapped in `optimizers.py`, which supports `adam` (default) and `sgd`. Use `--optim string_name` to manage the optimizer.

## Train
Training codes are implemented mainly in
* `train_img_model_xent.py`: train image model with cross entropy loss.
* `train_img_model_xent_htri.py`: train image model with combination of cross entropy loss and hard triplet loss.
* `train_img_model_cent.py`: train image model with center loss.
* `train_vid_model_xent.py`: train video model with cross entropy loss.
* `train_vid_model_xent_htri.py`: train video model with combination of cross entropy loss and hard triplet loss.

For example, to train an image reid model using ResNet50 and cross entropy loss, run
```bash
python train_img_model_xent.py -d market1501 -a resnet50 --max-epoch 60 --train-batch 32 --test-batch 32 --stepsize 20 --eval-step 20 --save-dir log/resnet50-xent-market1501 --gpu-devices 0
```

To use multiple GPUs, you can set `--gpu-devices 0,1,2,3`.

Please run `python train_blah_blah.py -h` for more details regarding arguments.

## Results
### Image person reid
#### Market1501

| Model | Param Size (M) | Loss | Rank-1/5/10 (%) | mAP (%) | Model weights | Published Rank | Published mAP |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | 7.72 | xent | 86.5/93.6/95.7 | 67.8 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/densenet121_xent_market1501.pth.tar) | | |
| DenseNet121 | 7.72 | xent+htri | 89.5/96.3/97.5 | 72.6 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/densenet121_xent_htri_market1501.pth.tar) | | |
| Resnet50 | 25.05 | cent | 85.1/93.8/96.2 | 69.1 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_cent_market1501.pth.tar) | | |
| ResNet50 | 25.05 | xent | 85.4/94.1/95.9 | 68.8 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_xent_market1501.pth.tar) | | |
| ResNet50 | 25.05 | xent+htri | 87.5/95.3/97.3 | 72.3 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_xent_htri_market1501.pth.tar) | | |
| ResNet50M | 30.01 | xent | 89.4/95.9/97.4 | 75.0 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50m_xent_market1501.pth.tar) | 89.9/-/- | 75.6 |
| ResNet50M | 30.01 | xent+htri | 90.7/97.0/98.2 | 76.8 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50m_xent_htri_market1501.pth.tar) | | |
| MuDeep | 138.02 | xent+htri| 71.5/89.3/96.3 | 47.0 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/mudeep_xent_htri_market1501.pth.tar) | | |

#### CUHK03 (detected, [new protocol (767/700)](https://github.com/zhunzhong07/person-re-ranking#the-new-trainingtesting-protocol-for-cuhk03))

| Model | Param Size (M) | Loss | Rank-1/5/10 (%) | mAP (%) | Model weights | Published Rank | Published mAP |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | 7.74 | xent | 41.0/61.7/71.5 | 40.6 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/densenet121_xent_cuhk03.pth.tar) | | |
| ResNet50 | 25.08 | xent | 48.8/69.4/78.4 | 47.5 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_xent_cuhk03.pth.tar) | | |
| ResNet50M | 30.06 | xent | 57.5/75.4/82.5 | 55.2 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50m_xent_cuhk03.pth.tar) | 47.1/-/- | 43.5 |
| HACNN | 3.72 | xent | 42.4/60.9/70.5 | 40.9 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/hacnn_xent_cuhk03.pth.tar) | 41.7/-/- |38.6 |

### Video person reid
#### MARS

| Model | Param Size (M) | Loss | Rank-1/5/10 (%) | mAP (%) | Model weights | Published Rank | Published mAP |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | 7.59 | xent | 65.2/81.1/86.3 | 52.1 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/densenet121_xent_mars.pth.tar) | | |
| DenseNet121 | 7.59 | xent+htri | 82.6/93.2/95.4 | 74.6 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/densenet121_xent_htri_mars.pth.tar) | | |
| ResNet50 | 24.79 | xent | 74.5/88.8/91.8 | 64.0 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50_xent_mars.pth.tar) | | |
| ResNet50 | 24.79 | xent+htri | 80.8/92.1/94.3 | 74.0 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50_xent_htri_mars.pth.tar) | | |
| ResNet50M | 29.63 | xent | 77.8/89.8/92.8 | 67.5 | - | | |
| ResNet50M | 29.63 | xent+htri | 82.3/93.8/95.3 | 75.4 | - | | |

#### iLIDS-VID

| Model | Param Size (M) | Loss | Rank-1/5/10 (%) | mAP (%) | Model weights | Published Rank | Published mAP |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50 | 23.82 | xent | 62.7/82.7/90.7 | 72.6 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50_xent_ilidsvid.pth.tar) | | |
| ResNet50M | 28.17 | xent | 63.3/85.3/92.7 | 73.6 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50m_xent_ilidsvid.pth.tar) | | |

#### PRID-2011

| Model | Param Size (M) | Loss | Rank-1/5/10 (%) | mAP (%) | Model weights | Published Rank | Published mAP |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50 | 23.69 | xent | 75.3/96.6/97.8 | 84.3 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50_xent_prid.pth.tar) | | |
| ResNet50M | 27.98 | xent | 85.4/96.6/98.9 | 90.1 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50m_xent_prid.pth.tar) | | |

## Test
Say you have downloaded ResNet50 trained with `xent` on `market1501`. The path to this model is  `'saved-models/resnet50_xent_market1501.pth.tar'` (create a directory to store model weights `mkdir saved-models/`). Then, run the following command to test
```bash
python train_img_model_xent.py -d market1501 -a resnet50 --evaluate --resume saved-models/resnet50_xent_market1501.pth.tar --save-dir log/resnet50-xent-market1501 --test-batch 32
```

Likewise, to test video reid model, you should have a pretrained model saved under `saved-models/`, e.g. `saved-models/resnet50_xent_mars.pth.tar`, then run
```bash
python train_vid_model_xent.py -d mars -a resnet50 --evaluate --resume saved-models/resnet50_xent_mars.pth.tar --save-dir log/resnet50-xent-mars --test-batch 2
```
**Note** that `--test-batch` in video reid represents number of tracklets. If we set this argument to 2, and sample 15 images per tracklet, the resulting number of images per batch is 2*15=30. Adjust this argument according to your GPU memory.

## Q&A
1. **How do I set different learning rates to different components in my model?**

A: Instead of giving `model.parameters()` to optimizer, you could pass an iterable of `dict`s, as described [here](http://pytorch.org/docs/master/optim.html#per-parameter-options). Please see the example below
```python
# First comment the following code.
#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
param_groups = [
  {'params': model.base.parameters(), 'lr': 0},
  {'params': model.classifier.parameters()},
]
# Such that model.base will be frozen and model.classifier will be trained with
# the default leanring rate, i.e. args.lr. This example code only applies to model
# that has two components (base and classifier). Modify the code to adapt to your model.
optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
```
Of course, you can pass `model.classifier.parameters()` to optimizer if you only need to train the classifier (in this case, setting the `requires_grad`s wrt the base model params to false will be more efficient).


## References
[1] [He et al. Deep Residual Learning for Image Recognition. CVPR 2016.](https://arxiv.org/abs/1512.03385)<br />
[2] [Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching. arXiv:1711.08106.](https://arxiv.org/abs/1711.08106) <br />
[3] [Huang et al. Densely Connected Convolutional Networks. CVPR 2017.](https://arxiv.org/abs/1608.06993) <br />
[4] [Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.](https://arxiv.org/abs/1703.07737) <br />
[5] [Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.](https://arxiv.org/abs/1512.00567) <br />
[6] [Kingma and Ba. Adam: A Method for Stochastic Optimization. ICLR 2015.](https://arxiv.org/abs/1412.6980) <br />
[7] [Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) <br />
[8] [Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.](http://www.liangzheng.com.cn/Project/project_mars.html) <br />
[9] [Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016](https://ydwen.github.io/papers/WenECCV16.pdf) <br />
[10] [Qian et al. Multi-scale Deep Learning Architectures for Person Re-identification. ICCV 2017.](https://arxiv.org/abs/1709.05165) <br />
[11] [Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.](http://www.eecs.qmul.ac.uk/~xiatian/papers/ECCV14/WangEtAl_ECCV14.pdf) <br />
[12] [Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.](https://files.icg.tugraz.at/seafhttp/files/ba284964-6e03-4261-bb39-e85280707598/hirzer_scia_2011.pdf) <br />
[13] [Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf) <br />
[14] [Zhong et al. Re-ranking Person Re-identification with k-reciprocal Encoding. CVPR 2017](https://arxiv.org/abs/1701.08398) <br />
[15] [Li et al. Harmonious Attention Network for Person Re-identification. CVPR 2018.](https://arxiv.org/abs/1802.08122) <br />
