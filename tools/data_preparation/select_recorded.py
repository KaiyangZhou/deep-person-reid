import json
import logging
import os
import shutil
from random import shuffle

from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

VIRTUAL_SIMULATION_LABEL_TYPE = "safeXsn/object-detection"
HUMAN_ANNOTATED_NO = "no"


def main(input_dir, output_dir, max_files: int = 20):
    person_dict = dict()
    camera_dict = dict()
    scenario_dict = dict()
    resolution_dict = dict()
    filename_dict = dict()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for person_name in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        if person_name not in person_dict.keys():
            person_dict[person_name] = len(person_dict.keys())
        for scenario_name in os.listdir(person_path):
            scenario_path = os.path.join(person_path, scenario_name)
            if not os.path.isdir(scenario_path):
                continue
            if scenario_name not in scenario_dict.keys():
                scenario_dict[scenario_name] = len(scenario_dict.keys())
            for resolution_name in os.listdir(scenario_path):
                resolution_path = os.path.join(scenario_path, resolution_name)
                if not os.path.isdir(resolution_path):
                    continue
                if resolution_name not in resolution_dict.keys():
                    resolution_dict[resolution_name] = len(resolution_dict.keys())
                for camera_name in os.listdir(resolution_path):
                    camera_path = os.path.join(resolution_path, camera_name)
                    if not os.path.isdir(camera_path):
                        continue
                    if camera_name not in camera_dict.keys():
                        camera_dict[camera_name] = len(camera_dict.keys())
                    files = os.listdir(camera_path)
                    shuffle(files)
                    crop_files = files[: max_files]
                    file_base = "{0}_{1}_{2}_{3}_".format(str(person_dict[person_name]), str(camera_dict[camera_name]),
                                                          str(scenario_dict[scenario_name]),
                                                          str(resolution_dict[resolution_name]))

                    for crop_file in crop_files:
                        if file_base in filename_dict:
                            file_id = filename_dict[file_base]
                            filename_dict[file_base] += 1
                        else:
                            file_id = 0
                            filename_dict[file_base] = 0

                        dst = os.path.join(output_dir, file_base + str(file_id) + ".jpg")
                        src = os.path.join(camera_path, crop_file)
                        shutil.copyfile(src, dst)

    id_dict = dict()
    id_dict["people"] = person_dict
    id_dict["cameras"] = camera_dict
    id_dict["scenarios"] = scenario_dict
    id_dict["resolutions"] = resolution_dict

    dict_path = os.path.join(output_dir, "id_dict.json")
    with open(dict_path, 'w') as outfile:
        json.dump(id_dict, outfile)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", default="/Users/joshuajohanson/signature/data/recorded_crop",
                        type=str, help="path to config config file path json")
    parser.add_argument("--output_dir", default="/Users/joshuajohanson/signature/data/selected_miim_recorded",
                        type=str, help="path to base output dir")

    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
