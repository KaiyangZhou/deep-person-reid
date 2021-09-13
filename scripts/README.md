## Deep Person REID
### Object Signature Development Library
Be sure to have pyenv and poetry setup.
### Dev Setup
You will want to run training/inference on a GPU
```shell script
poetry install -vvv 
```

### Prepare dataset

Pick from available datasets.  Currently, we have:
['sensereid', 'prid', 'ward', 'rpifield', 'miim_simulation', 'sarc3d', 'pes3d', 'miim_recorded']

Make sure that any dataset you use is commercially viable:
https://safexai.atlassian.net/wiki/spaces/MIME/pages/2518745112/Object+signature+data+sources

You can pick as many datasets as you would like, but each needs to be prepared before using.  
For PRID and SenseReID, follow the instructions here:

https://kaiyangzhou.github.io/deep-person-reid/datasets.html#datasets

#### Prepare Simulation Dataset (miim_simulation)
Follow the README https://github.com/AICradle/intel-data-converter/tree/develop/data_converter/simulation) to generate a
REID dataset from our simulation data.

Within [data](data) you will need to have a structure that looks like the following:
* crops
* guids.json
* representation_dataset.jsonl

Please note that the code is looking for those exact names (crops, guids.sjon, representation_dataset.jsonl).
For the time being those names are hardcoded, so please name accordingly.
 
You can find the logic for the dataset preparation [here](torchreid/data/datasets/image/miim_simulation.py)

Note that if the representation_dataset.jsonl file has relative paths to the crops then you will need
to configure PyCharm to set your working directory to the data folder. 

This can be done in the debug/run config

The training script will produce a model file and a TensorBoard file which you can view during training.

You can find an example dataset here (s3://simulated-data-sandbox/simulated-object-signature/simulated_test.tar.gz) though
the files need to be renamed.

#### Prepare recorded data from shop  (miim_recorded)
Create the data structure as follows:
parent_directory
└── person_name
    └── scenario_name
        └── resolution_name
            ├── camera_name
            │   ├── cropped_image.jpg


run the following command
```shell script
poetry run python tools/data_preparation/select_recorded.py --input_dir input_dir --output_dir selected_miim_recorded
```

This will create a directory with crops with following structure:

<person_id>_<camera_id>_<scenario_id>_<resolution_id>_<image_id>.jpg

There will also be id_dict which will have the mapping to the ids.

## Train

Make sure that all training and testing data is appropriately prepared and in DATA_DIR (data by default)
To train run the following:

```shell script
poetry run python scripts/train.py 
```

By default, this will train on miim_simulation and test on miim_recorded. You can change most of the training options
at the command line.  If you are running on GPU, you will need the cuda option.

The options are as follows:

train.py [-h] [--sources SOURCES [SOURCES ...]] [--targets TARGETS [TARGETS ...]] [--model_type MODEL_TYPE] [--max_epochs MAX_EPOCHS] [--optimizer_name OPTIMIZER_NAME]
                [--optimizer_lr OPTIMIZER_LR] [--scheduler_name SCHEDULER_NAME] [--scheduler_stepsize SCHEDULER_STEPSIZE] [--loss_type LOSS_TYPE] [--height HEIGHT] [--width WIDTH]
                [--data_dir DATA_DIR] [--model_dir MODEL_DIR] [--cuda]

We have had the most success with this command:

```shell script
poetry run python scripts/train.py --cuda --max_epochs 30 --sources miim_simulation prid ward pes3d rpifield > log.out
```

The training script will produce a model file and a TensorBoard file which you can view during training in the model_dir,
which by default is output/<sources>-<model_type>.  By default, it will run with the last model trained.  To change
which model to use, update param.json.  To see which epoch had the best mean average precision, run this command on the output:

grep mAP: log.out

#### Testing Model
You can test the model on a dataset via the [test.py](scripts/test.py) script.

This script is very similar to the training scripts. You need to provide the directory that contains the param.json file.
By default, it will test on the testing portion of the datasets used to train.

```shell script
poetry run scripts/test.py --model_dir $PATH_TO_MODEL_FILE
```

usage: test.py [-h] --model_dir MODEL_DIR [--targets TARGETS [TARGETS ...]] [--cuda] [--data_dir DATA_DIR] [--eval_dir EVAL_DIR]

#### Extract Feature Vectors
You can extract feature vectors via the  [extract_features.py](scripts/extract_features.py.py) script.

This script takes in a manfiest file path generated from the data_converter tools (https://github.com/AICradle/intel-data-converter/tree/develop/data_converter/simulation).


```shell script
poetry run scripts/extract_features.py --manfiest_file_path $PATH_TO_MANIFEST.JSONL --out_fp $PATH_TO_OUT_FILE.JSONL --model_fp $PATH_TOM_MODEL_FILE
```

TODO FINISH DOCUMENTING
