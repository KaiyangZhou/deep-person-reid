# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Torchreid is a PyTorch library for deep learning person re-identification (re-ID). Person re-ID is the task of identifying people across multiple non-overlapping camera views. The library is based on the ICCV'19 paper "Omni-Scale Feature Learning for Person Re-Identification."

**Note: This library is no longer actively maintained.**

## Common Commands

### Installation
```bash
# Create conda environment
conda create --name torchreid python=3.7
conda activate torchreid

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (adjust cudatoolkit version as needed)
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# Install torchreid in development mode (allows code changes without rebuilding)
python setup.py develop
```

### Docker
```bash
make build-image  # Build Docker image
make run          # Run container with GPU support
```

### Linting
```bash
bash linter.sh  # Runs isort, yapf, then flake8
```

### Training
```bash
# Train OSNet on Market1501
python scripts/main.py \
  --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
  --transforms random_flip random_erase \
  --root $PATH_TO_DATA

# Cross-domain: train on DukeMTMC, test on Market1501
python scripts/main.py \
  --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml \
  -s dukemtmcreid -t market1501 \
  --transforms random_flip color_jitter \
  --root $PATH_TO_DATA
```

### Evaluation
```bash
python scripts/main.py \
  --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
  --root $PATH_TO_DATA \
  model.load_weights log/osnet_x1_0_market1501_softmax_cosinelr/model.pth.tar-250 \
  test.evaluate True
```

### Testing Cython Build
```bash
python torchreid/metrics/rank_cylib/test_cython.py
```

### Multi-Split Results Aggregation
```bash
python tools/parse_test_res.py log/eval_viper
```

## Architecture

### Core Package Structure (`torchreid/`)

- **`data/`** - Data loading and augmentation
  - `datamanager.py`: `ImageDataManager` and `VideoDataManager` classes
  - `sampler.py`: Custom samplers (`RandomIdentitySampler`, `RandomDomainSampler`, `RandomDatasetSampler`)
  - `transforms.py`: Augmentations (random_flip, random_erase, color_jitter, etc.)
  - `datasets/image/`: 13+ image re-ID datasets (market1501, cuhk03, dukemtmcreid, msmt17, etc.)
  - `datasets/video/`: 4 video re-ID datasets (mars, ilids, prid2011, dukemtmc-videoreid)

- **`models/`** - 50+ model architectures
  - `osnet.py`, `osnet_ain.py`: OSNet variants (state-of-the-art for re-ID)
  - `resnet.py`, `senet.py`, `densenet.py`: ImageNet backbones
  - `mobilenetv2.py`, `shufflenetv2.py`: Lightweight models
  - `pcb.py`, `hacnn.py`, `mlfn.py`: Re-ID specific architectures

- **`engine/`** - Training/evaluation logic
  - `engine.py`: Base `Engine` class with checkpoint saving, TensorBoard logging
  - `image/softmax.py`, `image/triplet.py`: Image-based training engines
  - `video/softmax.py`, `video/triplet.py`: Video-based training engines

- **`losses/`** - Loss functions
  - `cross_entropy_loss.py`: Softmax with optional label smoothing
  - `hard_mine_triplet_loss.py`: Hard-mining triplet loss

- **`metrics/`** - Evaluation (CMC, mAP)
  - `rank_cylib/`: Cython-optimized ranking for faster evaluation

- **`utils/`** - Utilities
  - `feature_extractor.py`: Simple API for inference
  - `rerank.py`: Re-ranking for improved retrieval

### Entry Points

- **`scripts/main.py`**: Unified CLI for training/testing with YACS config
- **`scripts/default_config.py`**: Configuration schema with all hyperparameters
- **`configs/`**: Example YAML configuration files

### Configuration System

Uses YACS for hierarchical YAML configs. Override via command-line:
```bash
python scripts/main.py --config-file config.yaml train.lr 0.001 train.max_epoch 100
```

Key config groups: `model`, `data`, `train`, `loss`, `test`, `sampler`

### Data Tuple Format

Dataset items are 4-tuples: `(impath, pid, camid, dsetid)` where `dsetid` identifies the source dataset for multi-dataset training.

### Model Factory Pattern

```python
import torchreid
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=751,
    loss='softmax',  # or 'triplet'
    pretrained=True
)
```

### Feature Extraction API

```python
from torchreid.utils import FeatureExtractor
extractor = FeatureExtractor(model_name='osnet_x1_0', model_path='checkpoint.pth')
features = extractor(image_list)
```

## Key Resources

- Documentation: https://kaiyangzhou.github.io/deep-person-reid/
- Model Zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
- Pretrained weights: https://huggingface.co/kaiyangzhou/osnet
- Tech report: https://arxiv.org/abs/1910.10093
