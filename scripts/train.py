from argparse import ArgumentParser

import torchreid

import json
import logging


def read_jsonl(in_file_path):
    with open(in_file_path) as f:
        data = map(json.loads, f)
        for datum in data:
            yield datum


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def main(data_dir, save_dir):
    """
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='market1501',
        targets='market1501',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop']
    )
    """
    datamanager = torchreid.data.ImageDataManager(
        root=data_dir,
        sources=['safex_carla_simulation'],
        height=50,
        width=50,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'color_jitter']
    )

    model = torchreid.models.build_model(
        name='osnet_ain_x1_0',
        num_classes=datamanager.num_train_pids,
        loss='triplet',
        pretrained=False
    )

    model = model.cpu()

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

    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    engine.run(
        save_dir=save_dir,
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="./reid_out/", required=False)
    parser.add_argument("--save_dir", default="./reid_out/", required=False)

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        save_dir=args.save_dir
    )
