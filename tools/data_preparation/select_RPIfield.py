import os
import shutil
import json
from argparse import ArgumentParser
from random import shuffle
from typing import Dict, List


class RPIFieldSelector:
    def __init__(self, output_dir: str, max_per_camera: int, min_to_output: int):
        self.output_dir = output_dir
        self.max_per_camera = max_per_camera
        self.min_to_output = min_to_output
        self.camera_dict = dict()
        self.person_dict = dict()
        self.person_images: Dict[str, Dict[str, str]] = dict()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_camera(self, camera_path: str, camera_id):
        path_lookup: Dict[str, List[str]] = dict()
        for person_folder in os.listdir(camera_path):
            person_path = os.path.join(camera_path, person_folder)
            if not os.path.isdir(person_path):
                continue
            person_name = person_folder.split('_')[0]
            if person_name not in path_lookup:
                path_lookup[person_name] = list()

            for image in os.listdir(person_path):
                image_path = os.path.join(person_path, image)
                path_lookup[person_name].append(image_path)

        for person_name in path_lookup.keys():
            if person_name not in self.person_images.keys():
                self.person_images[person_name] = dict()

            image_paths = path_lookup[person_name]
            shuffle(image_paths)
            image_paths = image_paths[:self.max_per_camera]
            for image_path in image_paths:
                self.person_images[person_name][image_path] = camera_id

    def process_folder(self, input_dir: str):
        for camera_name in os.listdir(input_dir):
            camera_path = os.path.join(input_dir, camera_name)
            if not os.path.isdir(camera_path):
                continue
            if camera_name not in self.camera_dict.keys():
                self.camera_dict[camera_name] = str(len(self.camera_dict.keys()))
            self.process_camera(camera_path, self.camera_dict[camera_name])

        for person_name in self.person_images:
            if len(self.person_images[person_name].keys()) < self.min_to_output:
                continue
            if person_name not in self.person_dict:
                self.person_dict[person_name] = str(len(self.person_dict.keys()))
            person_id = self.person_dict[person_name]
            image_count = 0
            for image_path in self.person_images[person_name]:
                camera_id = self.person_images[person_name][image_path]
                dst = os.path.join(self.output_dir, person_id + '_' + camera_id + '_' + str(image_count) + ".png")
                image_count += 1
                shutil.copyfile(image_path, dst)

        id_dict = dict()
        id_dict["people"] = self.person_dict
        id_dict["cameras"] = self.camera_dict
        dict_path = os.path.join(self.output_dir, "id_dict.json")
        with open(dict_path, 'w') as outfile:
            json.dump(id_dict, outfile)


def main(input_dir: str, output_dir: str, max_per_camera: int, min_to_output: int):
    selector = RPIFieldSelector(output_dir, max_per_camera, min_to_output)
    selector.process_folder(input_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", default="/Users/joshuajohanson/signature/data/RPIfield/RPIfield_v0/Data",
                        type=str, help="path to config config file path json")
    parser.add_argument("--output_dir", default="/Users/joshuajohanson/signature/data/selected_RPIfield",
                        type=str, help="path to base output dir")
    parser.add_argument("--max_per_camera", default=5,
                        type=int, help="max crops per camera")
    parser.add_argument("--min_to_output", default=10,
                        type=int, help="max crops per camera")

    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_per_camera=args.max_per_camera,
        min_to_output=args.min_to_output
    )
