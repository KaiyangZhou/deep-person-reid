import argparse
import os
import sys
import numpy as np
from pathlib import Path
import torch

from brocolli.converter.pytorch_caffe_parser import PytorchCaffeParser

from torchreid.utils.feature_extractor import FeatureExtractor


__model_types = [
    'resnet50', 'mlfn', 'hacnn', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4',
    'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25',
    'osnet_ibn_x1_0', 'osnet_ain_x1_0']


def get_model_name(model):
    model = str(model).rsplit('/', 1)[-1].split('.')[0]
    for x in __model_types:
        if x in model:
            return x
    return None

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caffe Model Converter")
    parser.add_argument(
        "-p",
        "--weights",
        type=Path,
        default="./osnet_x0_75_imagenet.pth",
        help="Path to weights",
    )
    parser.add_argument(
        '--imgsz', '--img', '--img-size',
        nargs='+',
        type=int,
        default=[256, 128],
        help='image (h, w)'
    )
    parser.add_argument('--include',
                        nargs='+',
                        default=['onnx', 'openvino', 'tflite'],
                        help='onnx, openvino, tflite')
    args = parser.parse_args()

    # Build model
    extractor = FeatureExtractor(
        # get rid of dataset information DeepSort model name
        model_name=get_model_name(args.weights),
        model_path=args.weights,
        device=str('cpu')
    )
    
    im = torch.zeros(1, 3, args.imgsz[0], args.imgsz[1]).to('cpu')  # image size(1,3,640,480) BCHW iDetection
 
    concrete_args = {"return_featuremaps": False}
    runner = PytorchCaffeParser(extractor.model, im, concrete_args=concrete_args)
    runner.convert()
    runner.save(get_model_name(args.weights))
    runner.check_result()


