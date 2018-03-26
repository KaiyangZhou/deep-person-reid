# deep-person-reid
This repo contains [pytorch](http://pytorch.org/) implementations of deep person re-identification models.

Pretrained models are available.

We will actively maintain this repo to incorporate new models.

## Updates
- Mar 2018: Added [Multi-scale Deep CNN (ICCV'17)](https://arxiv.org/abs/1709.05165) [10] with slight modifications: (a) Input size is (256, 128) instead of (160, 60); (b) We add an average pooling layer after the last conv feature maps. (c) We train the network with our strategy. Model trained from scratch on Market1501 is [available](https://github.com/KaiyangZhou/deep-person-reid#results).
- Mar 2018: Added [center loss (ECCV'16)](https://github.com/KaiyangZhou/pytorch-center-loss) [9] and the trained model weights.

## Dependencies
- [Pytorch](http://pytorch.org/)
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

Market1501 [7]:
1. download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html.
2. extract dataset and rename to `market1501`.

MARS [8]:
1. create a directory named `mars/` under `data/`.
2. download dataset to `data/mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. extract `bbox_train.zip` and `bbox_test.zip`.
4. download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars`. (we want to follow the standard split in [8])

## Dataset loaders
These are implemented in `dataset_loader.py` where we have two main classes that subclass [torch.utils.data.Dataset](http://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html#Dataset):
* [ImageDataset](https://github.com/KaiyangZhou/deep-person-reid/blob/master/dataset_loader.py#L22): processes image-based person reid datasets.
* [VideoDataset](https://github.com/KaiyangZhou/deep-person-reid/blob/master/dataset_loader.py#L38): processes video-based person reid datasets.

These two classes are used for [torch.utils.data.DataLoader](http://pytorch.org/docs/master/_modules/torch/utils/data/dataloader.html#DataLoader) that can provide batched data. Data loader wich `ImageDataset` outputs batch data of `(batch, channel, height, width)`, while data loader with `VideoDataset` outputs batch data of `(batch, sequence, channel, height, width)`.

## Models
* `models/ResNet.py`: ResNet50 [1], ResNet50M [2].
* `models/DenseNet.py`: DenseNet121 [3].
* `models/MuDeep.py`: MuDeep [10]. 

## Loss functions
* `xent`: cross entropy + label smoothing regularizer [5].
* `htri`: triplet loss with hard positive/negative mining [4] .
* `cent`: center loss [9].

We use `Adam` [6] everywhere, which turned out to be the most effective optimizer in our experiments.

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

Then, you will see
```bash
==========
Args:Namespace(arch='resnet50', dataset='market1501', eval_step=20, evaluate=False, gamma=0.1,
gpu_devices='0', height=256, lr=0.0003, max_epoch=60, print_freq=10, resume='',
save_dir='log/resnet50/', seed=1, start_epoch=0, stepsize=20, test_batch=32, train_batch=32,
use_cpu=False, weight_decay=0.0005, width=128, workers=4)
==========
Currently using GPU 0
Initializing dataset market1501
=> Market1501 loaded
Dataset statistics:
  ------------------------------
  subset   | # ids | # images
  ------------------------------
  train    |   751 |    12936
  query    |   750 |     3368
  gallery  |   751 |    15913
  ------------------------------
  total    |  1501 |    32217
  ------------------------------
Initializing model: resnet50
Model size: 25.04683M
==> Epoch 1/60
Batch 10/404     Loss 6.665115 (6.781841)
Batch 20/404     Loss 6.792669 (6.837275)
Batch 30/404     Loss 6.592124 (6.806587)
... ...
==> Epoch 60/60
Batch 10/404     Loss 1.101616 (1.075387)
Batch 20/404     Loss 1.055073 (1.075455)
Batch 30/404     Loss 1.081339 (1.073036)
... ...
==> Test
Extracted features for query set, obtained 3368-by-2048 matrix
Extracted features for gallery set, obtained 15913-by-2048 matrix
Computing distance matrix
Computing CMC and mAP
Results ----------
mAP: 68.8%
CMC curve
Rank-1  : 85.4%
Rank-5  : 94.1%
Rank-10 : 95.9%
Rank-20 : 97.2%
------------------
Finished. Total elapsed time (h:m:s): 1:57:44
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
| ResNet50 | 25.05 | xent | 85.4/94.1/95.9 | 68.8 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_xent_market1501.pth.tar) | 87.3/-/- | 67.6 |
| ResNet50 | 25.05 | xent+htri | 87.5/95.3/97.3 | 72.3 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_xent_htri_market1501.pth.tar) | | |
| ResNet50M | 30.01 | xent | 89.0/95.5/97.3 | 75.0 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50m_xent_market1501.pth.tar) | 89.9/-/- | 75.6 |
| ResNet50M | 30.01 | xent+htri | 90.4/96.7/98.0 | 76.6 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50m_xent_htri_market1501.pth.tar) | | |
| MuDeep | 138.02 | xent+htri| 71.5/89.3/96.3 | 47.0 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/mudeep_xent_htri_market1501.pth.tar) | | |

### Video person reid
#### MARS

| Model | Param Size (M) | Loss | Rank-1/5/10 (%) | mAP (%) | Model weights | Published Rank | Published mAP |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | 7.59 | xent | 65.2/81.1/86.3 | 52.1 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/densenet121_xent_mars.pth.tar) | | |
| DenseNet121 | 7.59 | xent+htri | 82.6/93.2/95.4 | 74.6 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/densenet121_xent_htri_mars.pth.tar) | | |
| ResNet50 | 24.79 | xent | 74.5/88.8/91.8 | 64.0 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50_xent_mars.pth.tar) | | |
| ResNet50 | 24.79 | xent+htri | 80.8/92.1/94.3 | 74.0 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50_xent_htri_mars.pth.tar) | | |
| ResNet50M | 29.63 | xent | 77.8/89.8/92.8 | 67.5 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50m_xent_mars.pth.tar) | | |
| ResNet50M | 29.63 | xent+htri | 82.3/93.8/95.3 | 75.4 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50m_xent_htri_mars.pth.tar) | | |

## Test
Say you have downloaded ResNet50 trained with `xent` on `market1501`. The path to this model is  `'saved-models/resnet50_xent_market1501.pth.tar'` (create a directory to store model weights `mkdir saved-models/`). Then, run the following command to test
```bash
python train_img_model_xent.py -d market1501 -a resnet50 --evaluate --resume saved-models/resnet50_xent_market1501.pth.tar --save-dir log/resnet50-xent-market1501 --test-batch 32
```

Likewise, to test video reid model, you should have a pretrained model saved under `saved-models/`, e.g. `saved-models/resnet50_xent_mars.pth.tar`, then run
```bash
python train_vid_model_xent.py -d mars -a resnet50 --evaluate --resume saved-models/resnet50_xent_mars.pth.tar --save-dir log/resnet50-xent-mars --test-batch 2
```
Note that `--test-batch` in video reid represents number of tracklets. If we set this argument to 2, and sample 15 images per tracklet, the resulting number of images per batch is 2*15=30. Adjust this argument according to your GPU memory.

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