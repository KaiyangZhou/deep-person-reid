## Deep Person REID
### Object Signature Development Library
Be sure to have pyenv and poetry setup.
### Dev Setup
You will want to run training/inference on a GPU
```shell script
poetry install -vvv 
```


## Run
#### Training on REID Dataset
Pick from available datasets: https://kaiyangzhou.github.io/deep-person-reid/pkg/data.html#module-torchreid.data.datasets.image.market1501

Change the source and targets on the DataManager in [train.py](scripts/train.py)
By default we use the market1501 dataset. 

You can also use multiple datasets for training.

Within the [train.py](scripts/train.py) you can change the usual Deep Learning config including:
* epochs 
* image size
* batch size
* transforms
* optimizer
* loss function
* learning rate
* etc

The defaults are what we have had the most success with. 

To train run the following:

```shell script
poetry run scripts/train.py 
```

The training script will produce a model file and a TensorBoard file which you can view during training.

#### Training on Simulation Dataset
Follow the README https://github.com/AICradle/intel-data-converter/tree/develop/data_converter/simulation) to generate a
REID dataset from our simulation data.

Within [data](data) you will need to have a structure that looks like the following:
* crops
* guids.json
* representation_dataset.jsonl

Please note that the code is looking for those exact names (crops, guids.sjon, representation_dataset.jsonl).
For the time being those names are hardcoded, so please name accordingly.
 
You can find the logic for the dataset preparation [here](torchreid/data/datasets/image/miim_simulation.py)

The training script is at (scripts/train_simulation.py)

Within the [train.py](scripts/train_simulation.py) you can change the usual Deep Learning config including:
* epochs 
* image size
* batch size
* transforms
* optimizer
* loss function
* learning rate
* etc

The defaults are what we have had the most success with.

Note that you can also train/test on both the simulation data and one of the REID datasets. 

To train run the following:
```shell script
poetry run scripts/train.py --data_dir $PATH_TO_DATA_DIRECTORY
```

Note that if the representation_dataset.jsonl file has relative paths to the crops then you will need
to configure PyCharm to set your working directory to the data folder. 

This can be done in the debug/run config

The training script will produce a model file and a TensorBoard file which you can view during training.

You can find an example dataset here (s3://simulated-data-sandbox/simulated-object-signature/simulated_test.tar.gz) though
the files need to be renamed.

#### Training on Shop Dataset
TODO FINISH DOCUMENTING


#### Testing Model
You also can test the model on a dataset via the [test.py](scripts/test.py) script.

This script is very similar to the training scripts. The only difference is that you also need to supply a model file path,
which is the output of a training run. This should be a PyTorch model file

```shell script
poetry run scripts/test.py --model_fp $PATH_TO_MODEL_FILE --data_dir $DATA_DIR --save_dir ./test_out
```

#### Extract Feature Vectors
You can extract feature vectors via the  [extract_features.py](scripts/extract_features.py.py) script.

This script takes in a manfiest file path generated from the data_converter tools (https://github.com/AICradle/intel-data-converter/tree/develop/data_converter/simulation).


```shell script
poetry run scripts/extract_features.py --manfiest_file_path $PATH_TO_MANIFEST.JSONL --out_fp $PATH_TO_OUT_FILE.JSONL --model_fp $PATH_TOM_MODEL_FILE
```

TODO FINISH DOCUMENTING