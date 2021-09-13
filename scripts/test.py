from argparse import ArgumentParser
from build_engine import build_engine

import torchreid

import json
import logging
import os

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def main(targets: list, data_dir: str, model_dir: str, eval_dir: str, cuda: bool):
    param_file = os.path.join(model_dir, "params.json")

    with open(param_file) as json_file:
        params = json.load(json_file)
        model_path = os.path.join(model_dir, params["model"])

        if not targets:
            targets = params["sources"]

        if not eval_dir:
            eval_dir = os.path.join("eval", os.path.basename(os.path.normpath(model_dir)))

        datamanager = torchreid.data.ImageDataManager(
            root=data_dir,
            sources=params["sources"],
            targets=targets,
            height=params["height"],
            width=params["width"],
            batch_size_train=32,
            batch_size_test=100,
            transforms=['random_flip', 'random_crop']
        )

        engine, model, optimizer = build_engine(datamanager, params, cuda)

        start_epoch = torchreid.utils.resume_from_checkpoint(
            model_path,
            model,
            optimizer
        )

        engine.run(
            start_epoch=start_epoch,
            save_dir=eval_dir,
            max_epoch=params["max_epochs"],
            fixbase_epoch=2,
            dist_metric=params["scheduler_name"],
            eval_freq=1,
            print_freq=2,
            test_only=True,
            visrank=True
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--targets", nargs='+', type=str, default="", required=False)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--data_dir", default="data", required=False)
    parser.add_argument("--eval_dir", default="", required=False)

    args = parser.parse_args()
    main(
        args.targets,
        args.data_dir,
        args.model_dir,
        args.eval_dir,
        args.cuda
    )
