# deep-person-reid
This repo contains [pytorch](http://pytorch.org/) implementations of deep person re-identification approaches.

This repo will be actively maintained.

## Install
1. `cd` to the folder where you want to download this repo.
2. run `git clone https://github.com/KaiyangZhou/deep-person-reid`.

## Prepare data
Create a directory to store reid datasets under this repo via
```
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
* `ImageDataset`: processes image-based person reid datasets.
* `VideoDataset`: processes video-based person reid datasets.

These two classes are used for [torch.utils.data.DataLoader](http://pytorch.org/docs/master/_modules/torch/utils/data/dataloader.html#DataLoader) that can provide batched data.

## Models
* `models/ResNet.py`: ResNet50 [1], ResNet50M [2].
* `models/DenseNet.py`: DenseNet121 [3].

## Loss functions
* `xent`: cross entropy + label smoothing regularizer [5].
* `htri`: triplet loss with hard positive/negative mining [4] .

We use `Adam` [6] everywhere, which turned out to be the most effective optimizer in our experiments.

## Train
Training codes are implemented mainly in
* `train_img_model_xent.py`: train image model with cross entropy loss.
* `train_img_model_xent_htri.py`: train image model with combination of cross entropy loss and hard triplet loss.
* `train_vid_model_xent.py`: train video model with cross entropy loss.
* `train_vid_model_xent_htri.py`: train video model with combination of cross entropy loss and hard triplet loss.

For example, to train an image reid model using ResNet50 and cross entropy loss, run
```
python train_img_model_xent.py -d market1501 -a resnet50 --max-epoch 60 --train-batch 32 --test-batch 32 --stepsize 20 --eval-step 20 --save-dir log/resnet50-xent-market1501 --gpu-devices 0
```

Please run `python train_blah_blah.py -h` for more details regarding arguments.

## Results
### Image person reid
#### Market1501

| Model | Size (M) | Loss | Rank-1/5/10 (%) | mAP (%) | Model weights | Published Rank | Published mAP |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | 7.72 | xent | 86.5/93.6/95.7 | 67.8 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/densenet121_xent_market1501.pth.tar) | | |
| DenseNet121 | 7.72 | xent+htri | 89.5/96.3/97.5 | 72.6 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/densenet121_xent_htri_market1501.pth.tar) | | |
| ResNet50 | 25.05 | xent | 85.4/94.1/95.9 | 68.8 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_xent_market1501.pth.tar) | 87.3/-/- | 67.6 |
| ResNet50 | 25.05 | xent+htri | 87.5/95.3/97.3 | 72.3 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50_xent_htri_market1501.pth.tar) | | |
| ResNet50M | 30.01 | xent | 89.0/95.5/97.3 | 75.0 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50m_xent_market1501.pth.tar) | 89.9/-/- | 75.6 |
| ResNet50M | 30.01 | xent+htri | 90.4/96.7/98.0 | 76.6 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/image-models/resnet50m_xent_htri_market1501.pth.tar) | | |

### Video person reid
#### MARS

| Model | Size (M) | Loss | Rank-1/5/10 (%) | mAP (%) | Model weights | Published Rank | Published mAP |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50 | 24.79 | xent | 74.5/88.8/91.8 | 64.0 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50_xent_mars.pth.tar) | | |
| ResNet50 | 24.79 | xent+htri | 80.8/92.1/94.3 | 74.0 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50_xent_htri_mars.pth.tar) | | |
| ResNet50M | 29.63 | xent | 77.8/89.8/92.8 | 67.5 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50m_xent_mars.pth.tar) | | |
| ResNet50M | 29.63 | xent+htri | 82.3/93.8/95.3 | 75.4 | [download](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/model-zoo/video-models/resnet50m_xent_htri_mars.pth.tar) | | |

## Test
Say you have downloaded ResNet50 trained with `xent` on `market1501`. The path to this model is  `'saved-models/resnet50_xent_market1501.pth.tar'` (create a directory to store model weights `mkdir saved-models/`). Then, run the following command to test
```
python train_img_model_xent.py -d market1501 -a resnet50 --evaluate --resume saved-models/resnet50_xent_market1501.pth.tar --save-dir log/resnet50-xent-market1501 --test-batch 32
```

Likewise, to test video reid model, you should have a pretrained model saved under `saved-models/`, e.g. `saved-models/resnet50_xent_mars.pth.tar`, then run
```
python train_vid_model_xent.py -d mars -a resnet50 --evaluate --resume saved-models/resnet50_xent_mars.pth.tar --save-dir log/resnet50-xent-mars --test-batch 2
```
Note that `--test-batch` in video reid represents number of tracklets. If we set this argument to 2, and sample 15 images per tracklet, the resulting number of images per batch is 2*15=30. Adjust this argument according to your GPU memory.

## References
[1] [He et al. Deep Residual Learning for Image Recognition. CVPR 2016.](https://arxiv.org/abs/1512.03385)<br />
[2] [Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching. arXiv:1711.08106.](https://arxiv.org/abs/1711.08106) <br />
[3] [Huang et al. Densely Connected Convolutional Networks. CVPR 2017.](https://arxiv.org/abs/1608.06993) <br />
[4] [Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.](https://arxiv.org/abs/1703.07737) <br />
[5] [Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.](https://arxiv.org/abs/1512.00567) <br />
[6] [Kingma and Ba. Adam: A Method for Stochastic Optimization. ICLR 2015.](https://arxiv.org/abs/1412.6980) <br />
[7] [Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) <br />
[8] [Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.](http://www.liangzheng.com.cn/Project/project_mars.html) <br />