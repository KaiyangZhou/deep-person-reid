from argparse import ArgumentParser
from typing import List
from build_engine import build_engine, check_unique_path
import torchreid
import os

import json
import logging

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def main(params: dict,
         targets: List[str],
         data_dir: str,
         model_dir: str,
         cuda: bool):
    if not targets:
        targets = params["sources"]

    if not model_dir:
        model_folder = ""
        if type(params["sources"]) == list:
            for source in params["sources"]:
                if model_folder:
                    model_folder += "-"
                model_folder += source
        else:
            model_folder = params["sources"]

        model_folder += "-" + params["model_type"]
        model_dir = check_unique_path(os.path.join("output", model_folder))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

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

    params["num_classes"] = datamanager.num_train_pids

    params["model"] = os.path.join("model", "model.pth.tar-" + str(params["max_epochs"]))
    param_file = os.path.join(model_dir, "params.json")
    with open(param_file, "w") as outfile:
        json.dump(params, outfile)

    engine, _, _ = build_engine(datamanager, params, cuda)

    engine.run(
        save_dir=model_dir,
        max_epoch=params["max_epochs"],
        fixbase_epoch=2,
        dist_metric=params["scheduler_name"],
        eval_freq=1,
        print_freq=2,
        test_only=False
    )

    LOGGER.info(f"Model saved in {model_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sources", nargs='+', type=str, default="miim_simulation", required=False)
    parser.add_argument("--targets", nargs='+', type=str, default="", required=False)
    parser.add_argument("--model_type", type=str, default="osnet_ain_x1_0", required=False)
    parser.add_argument("--max_epochs", type=int, default=2, required=False)
    parser.add_argument("--optimizer_name", type=str, default="amsgrad", required=False)
    parser.add_argument("--optimizer_lr", type=float, default=0.0015, required=False)
    parser.add_argument("--scheduler_name", type=str, default="cosine", required=False)
    parser.add_argument("--scheduler_stepsize", type=int, default=20, required=False)
    parser.add_argument("--loss_type", type=str, default="triplet", required=False)
    parser.add_argument("--height", type=int, default=256, required=False)
    parser.add_argument("--width", type=int, default=128, required=False)
    parser.add_argument("--data_dir", default="data", required=False)
    parser.add_argument("--model_dir", default="", required=False)
    parser.add_argument("--cuda", action='store_true')
    args = parser.parse_args()

    param_dict: dict = dict()
    param_dict["sources"] = args.sources
    param_dict["model_type"] = args.model_type
    param_dict["optimizer_name"] = args.optimizer_name
    param_dict["optimizer_lr"] = args.optimizer_lr
    param_dict["scheduler_name"] = args.scheduler_name
    param_dict["scheduler_stepsize"] = args.scheduler_stepsize
    param_dict["loss_type"] = args.loss_type
    param_dict["optimizer_lr"] = args.optimizer_lr
    param_dict["max_epochs"] = args.max_epochs
    param_dict["height"] = args.height
    param_dict["width"] = args.width

    main(
        params=param_dict,
        targets=args.targets,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        cuda=args.cuda
    )
