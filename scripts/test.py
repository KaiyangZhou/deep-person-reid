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
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip'],
        split_id=1
    )

    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='triplet',
        pretrained=True
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

    start_epoch = torchreid.utils.resume_from_checkpoint(
        '/home/ubuntu/deep-person-reid/reid_out/resnet/model/model.pth.tar-15',
        model,
        optimizer
    )

    engine.run(
        start_epoch=start_epoch,
        save_dir=save_dir,
        max_epoch=60,
        eval_freq=1,
        print_freq=2,
        test_only=True,
        visrank=True
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
