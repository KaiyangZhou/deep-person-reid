# deep-person-reid
[PyTorch](http://pytorch.org/) implementation of deep person re-identification models.

We support
- multi-GPU training.
- both image-based and video-based reid.
- standard dataset splits used by most papers.
- unified interface for different reid models.
- easy dataset preparation.
- end-to-end training and evaluation.
- fast cython-based evaluation.
- multi-dataset training.
- visualization of ranked results.
- state-of-the-art reid models.

## Updates
- xx-11-2018: xxx.

## Get started
1. `cd` to the folder where you want to download this repo.
2. Run `git clone https://github.com/KaiyangZhou/deep-person-reid`.
3. Install dependencies by `pip install -r requirements.txt` (if necessary).
4. To accelerate evaluation (10x faster), you can use cython-based evaluation code (developed by [luzai](https://github.com/luzai)). First `cd` to `eval_lib`, then do `make` or `python setup.py build_ext -i`. After that, run `python test_cython_eval.py` to test if the package is successfully installed.

## Datasets
Image reid datasets:
- Market1501
- CUHK03
- DukeMTMC-reID
- MSMT17
- VIPeR
- GRID
- CUHK01
- PRID450S
- SenseReID

Video reid datasets:
- MARS
- iLIDS-VID
- PRID2011
- DukeMTMC-VideoReID

Instructions regarding how to prepare (and do evaluation on) these datasets can be found in [DATASETS.md](DATASETS.md).


## Models
### ImageNet classification models
- [ResNet](https://arxiv.org/abs/1512.03385)
- [ResNeXt](https://arxiv.org/abs/1611.05431)
- [SENet](https://arxiv.org/abs/1709.01507)
- [DenseNet](https://arxiv.org/abs/1608.06993)
- [Inception-ResNet-V2](https://arxiv.org/abs/1602.07261)
- [Inception-V4](https://arxiv.org/abs/1602.07261)
- [Xception](https://arxiv.org/abs/1610.02357)

### Lightweight models
- [NASNet](https://arxiv.org/abs/1707.07012)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [ShuffleNet](https://arxiv.org/abs/1707.01083)
- [SqueezeNet](https://arxiv.org/abs/1602.07360)

### ReID-specific models
- [MuDeep](https://arxiv.org/abs/1709.05165)
- [HACNN](https://arxiv.org/abs/1802.08122)
- [PCB](https://arxiv.org/abs/1711.09349)
- [MLFN](https://arxiv.org/abs/1803.09132)

In the [MODEL_ZOO](MODEL_ZOO.md), we provide pretrained models and the training scripts to reproduce the results.

## Losses
- `xent`: cross entropy loss (with label smoothing regularizer).
- `htri`: [hard mining triplet loss](https://arxiv.org/abs/1703.07737).

## Tutorial
### Train
Training methods are implemented in
- `train_imgreid_xent.py`: train image-reid models with cross entropy loss.
- `train_imgreid_xent_htri.py`: train image-reid models with hard mining triplet loss or the combination of hard mining triplet loss and cross entropy loss.
- `train_imgreid_xent.py`: train video-reid models with cross entropy loss.
- `train_imgreid_xent_htri.py`: train video-reid models with hard mining triplet loss or the combination of hard mining triplet loss and cross entropy loss.

Input arguments for the above training scripts are unified in [args.py](args.py).

To train an image-reid model with cross entropy loss, you can do
```bash
python train_imgreid_xent.py \
-s market1501 \ # source dataset for training
-t market1501 \ # target dataset for test
--height 256 \ # image height
--width 128 \ # image width
--optim amsgrad \ # optimizer
--label-smooth \ # label smoothing regularizer
--lr 0.0003 \ # learning rate
--max-epoch 60 \ # maximum epoch to run
--stepsize 20 40 \ # stepsize for learning rate decay
--train-batch-size 32 \
--test-batch-size 100 \
-a resnet50 \ # network architecture
--save-dir log/resnet50-market-xent \ # where to save the log and models
--gpu-devices 0 \ # gpu device index
```

#### Multi-dataset training
`-s` and `-t` can take different strings of arbitrary length (delimited by space). For example, if you wanna train models on Market1501 + DukeMTMC-reID and test on both of them, you can use `-s market1501 dukemtmcreid` and `-t market1501 dukemtmcreid`. If say, you wanna test on a different dataset, e.g. MSMT17, then just do `-t msmt17`. Multi-dataset training is implemented for both image-reid and video-reid. Note that when `-t` takes multiple datasets, evaluation is performed on each dataset individually.

#### Two-stepped transfer learning
To finetune models pretrained on external large-scale datasets such as [ImageNet](http://www.image-net.org/), the [two-stepped training strategy](https://arxiv.org/abs/1611.05244) is useful. First, you freeze the base network and only train the randomly initialized layers (e.g. last linear classifier) for `--fixbase-epoch` epochs. Only the layers that are specified by `--open-layers` are set to the **train** mode and are allowed to be updated while other layers are frozen and set to the **eval** mode. Second, after the new layers are adapted to the old layers, all layers are opened to train for `--max-epoch` epochs.

For example, to train the [resnet50](torchreid/models/resnet.py) with a `classifier` being initialized randomly, you can set `--fixbase-epoch 5` and `--open-layers classifier`. The layer names must align with the attribute names in the model, i.e. `self.classifier` exists in the model. See `open_specified_layers(model, open_layers)` in [torchreid/utils/torchtools.py](torchreid/utils/torchtools.py) for more details.

#### Using hard mining triplet loss
`htri` requires adding `--train-sampler RandomIdentitySampler`.

#### Training video-reid models
For video reid, `test-batch-size` refers to the number of tracklets, so the real image batch size is `--test-batch-size * --seq-len`.

### Test

#### Evaluation mode
Use `--evaluate` to switch to the evaluation mode. In doing so, no model training is performed. For example, you wanna load model weights at `path_to/resnet50.pth.tar` for `resnet50` and do evaluation on Market1501, you can do
```bash
python train_imgreid_xent.py \
-s market1501 \ # this does not matter any more
-t market1501 \ # you can add more datasets in the test list
--height 256 \
--width 128 \
--test-batch-size 100 \
--evaluate \
-a resnet50 \
--load-weights path_to/resnet50.pth.tar \
--save-dir log/resnet50-eval
--gpu-devices 0 \
```

Note that `--load-weights` will discard layer weights that do not match the model layers in size.

#### Visualize ranked results
Ranked results can be visualized via `--visualize-ranks`, which works along with `--evaluate`. Ranked images will be saved in `save_dir/ranked_results` where `save_dir` is the directory you specify with `--save-dir`.

<div align="center">
  <img src="imgs/ranked_results.jpg" alt="train" width="70%">
</div>


## Misc
- [Related person ReID projects](RELATED_PROJECTS.md).


## Citation
Please link this project in your paper.

## License
This project is under the [MIT License](LICENSE).