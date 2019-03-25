Torchreid
===========
Torchreid is a library built on `PyTorch <https://pytorch.org/>`_ for deep-learning person re-identification.

It features:

- multi-GPU training
- support both image- and video-reid
- end-to-end training and evaluation
- incredibly easy preparation of reid datasets
- multi-dataset training
- cross-dataset evaluation
- standard protocol used by most research papers
- highly extensible (easy to add models, datasets, training methods, etc.)
- implementations of state-of-the-art deep reid models
- access to pretrained reid models
- advanced training techniques
- visualization of ranking results


Documentation: https://kaiyangzhou.github.io/deep-person-reid/.


News
------
24-03-2019: `Torchreid documentation <https://kaiyangzhou.github.io/deep-person-reid/>`_ is out!


Installation
---------------

1. Install PyTorch and torchvision following the `official instructions <https://pytorch.org/>`_.
2. Install Cython.
3. Clone ``deep-person-reid`` to your preferred directory.

.. code-block:: bash
    
    $ git clone https://github.com/KaiyangZhou/deep-person-reid.git

4. Install ``torchreid`` via

.. code-block:: bash
    
    $ cd deep-person-reid/
    $ python setup.py install # or python3
    $ # If you wanna modify the source code without
    $ # the need to re-build it, you can do
    $ # python setup.py develop


Get started: 30 seconds to Torchreid
-------------------------------------
1. Import ``torchreid``

.. code-block:: python
    
    import torchreid

2. Load data manager

.. code-block:: python
    
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='market1501',
        height=256,
        width=128,
        batch_size=32,
        market1501_500k=False
    )

3 Build model, optimizer and lr_scheduler

.. code-block:: python
    
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )

4. Build engine

.. code-block:: python
    
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

5. Run training and test

.. code-block:: python
    
    engine.run(
        save_dir='log/resnet50',
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )


A unified interface
-----------------------
In "deep-person-reid/scripts/", we provide a unified interface including a default parser file ``default_parser.py`` and the main script ``main.py``. For example, to train an image reid model on Market1501 using softmax, you can do

.. code-block:: bash
    
    python main.py \
    --root path/to/reid-data \
    --app image \
    --loss softmax \
    --label-smooth \
    -s market1501 \
    -a resnet50 \
    --optim adam \
    --lr 0.0003 \
    --max-epoch 60 \
    --stepsize 20 40 \
    --batch-size 32 \
    --save-dir log/resnet50-market-softmax \
    --gpu-devices 0

Please refer to ``default_parser.py`` and ``main.py`` for more details.


Datasets
--------

Image-reid datasets
^^^^^^^^^^^^^^^^^^^^^
- `Market1501 <https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf>`_
- `CUHK03 <https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf>`_
- `DukeMTMC-reID <https://arxiv.org/abs/1701.07717>`_
- `MSMT17 <https://arxiv.org/abs/1711.08565>`_
- `VIPeR <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.7285&rep=rep1&type=pdf>`_
- `GRID <http://www.eecs.qmul.ac.uk/~txiang/publications/LoyXiangGong_cvpr_2009.pdf>`_
- `CUHK01 <http://www.ee.cuhk.edu.hk/~xgwang/papers/liZWaccv12.pdf>`_
- `SenseReID <http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Spindle_Net_Person_CVPR_2017_paper.pdf>`_
- `QMUL-iLIDS <http://www.eecs.qmul.ac.uk/~sgg/papers/ZhengGongXiang_BMVC09.pdf>`_
- `PRID <https://pdfs.semanticscholar.org/4c1b/f0592be3e535faf256c95e27982db9b3d3d3.pdf>`_

Video-reid datasets
^^^^^^^^^^^^^^^^^^^^^^^
- `MARS <http://www.liangzheng.org/1320.pdf>`_
- `iLIDS-VID <https://www.eecs.qmul.ac.uk/~sgg/papers/WangEtAl_ECCV14.pdf>`_
- `PRID2011 <https://pdfs.semanticscholar.org/4c1b/f0592be3e535faf256c95e27982db9b3d3d3.pdf>`_
- `DukeMTMC-VideoReID <http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Exploit_the_Unknown_CVPR_2018_paper.pdf>`_

Models
-------

ImageNet classification models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- `ResNet <https://arxiv.org/abs/1512.03385>`_
- `ResNeXt <https://arxiv.org/abs/1611.05431>`_
- `SENet <https://arxiv.org/abs/1709.01507>`_
- `DenseNet <https://arxiv.org/abs/1608.06993>`_
- `Inception-ResNet-V2 <https://arxiv.org/abs/1602.07261>`_
- `Inception-V4 <https://arxiv.org/abs/1602.07261>`_
- `Xception <https://arxiv.org/abs/1610.02357>`_

Lightweight models
^^^^^^^^^^^^^^^^^^^
- `NASNet <https://arxiv.org/abs/1707.07012>`_
- `MobileNetV2 <https://arxiv.org/abs/1801.04381>`_
- `ShuffleNet <https://arxiv.org/abs/1707.01083>`_
- `SqueezeNet <https://arxiv.org/abs/1602.07360>`_

ReID-specific models
^^^^^^^^^^^^^^^^^^^^^^
- `MuDeep <https://arxiv.org/abs/1709.05165>`_
- `ResNet-mid <https://arxiv.org/abs/1711.08106>`_
- `HACNN <https://arxiv.org/abs/1802.08122>`_
- `PCB <https://arxiv.org/abs/1711.09349>`_
- `MLFN <https://arxiv.org/abs/1803.09132>`_


Losses
------
- `Softmax (cross entropy loss with label smoothing) <https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf>`_
- `Triplet (hard example mining triplet loss) <https://arxiv.org/abs/1703.07737>`_


Citation
---------
Please link this project in your paper
