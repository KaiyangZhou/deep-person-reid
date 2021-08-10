import json
import logging
import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm

from torchreid.utils import FeatureExtractor
from torchreid.utils.tools import read_jsonl

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def main(in_fp: str, model_fp: str, out_fp: str, device: str, max_objects: int):
    extractor = FeatureExtractor(
        model_name="resnet18",
        model_path=model_fp,
        device=device
    )

    manifest_entries = read_jsonl(in_fp)
    # Todo (Josh) speed this up by batching
    with open(out_fp, "w") as f:
        for ix, manifest_entry in tqdm(enumerate(manifest_entries), desc="objects"):

            if max_objects and ix > max_objects:
                continue

            path = manifest_entry["path"]
            features = extractor([path])
            features = features[0].cpu().detach().numpy()
            manifest_entry["features"] = features
            f.write(json.dumps(manifest_entry, cls=NumpyEncoder) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--manifest_fp", required=True, type=str)
    parser.add_argument("--model_fp", required=True, type=str)
    parser.add_argument("--out_fp", required=True, type=str)
    parser.add_argument("--max_objects", required=False, type=int, default=None)
    parser.add_argument("--device", required=False, type=str, default="cpu")

    args = parser.parse_args()
    main(
        in_fp=args.manifest_fp,
        model_fp=args.model_fp,
        out_fp=args.out_fp,
        device=args.device,
        max_objects=args.max_objects
    )
