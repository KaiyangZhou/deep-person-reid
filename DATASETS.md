## How to prepare data

Create a directory to store reid datasets under this repo via
```bash
cd deep-person-reid/
mkdir data/
```

If you wanna store datasets in another directory, you need to specify `--root path_to_your/data` when running the training code. Please follow the instructions below to prepare each dataset. After that, you can simply do `-d the_dataset` when running the training code. 

Please do not call image dataset when running video reid scripts, otherwise error would occur, and vice versa.

### Image ReID

**Market1501** [7]:
1. Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html.
2. Extract dataset and rename to `market1501`. The data structure would look like:
```
market1501/
    bounding_box_test/
    bounding_box_train/
    ...
```
3. Use `-d market1501` when running the training code.

**CUHK03** [13]:
1. Create a folder named `cuhk03/` under `data/`.
2. Download dataset to `data/cuhk03/` from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html and extract `cuhk03_release.zip`, so you will have `data/cuhk03/cuhk03_release`.
3. Download new split [14] from [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03). What you need are `cuhk03_new_protocol_config_detected.mat` and `cuhk03_new_protocol_config_labeled.mat`. Put these two mat files under `data/cuhk03`. Finally, the data structure would look like
```
cuhk03/
    cuhk03_release/
    cuhk03_new_protocol_config_detected.mat
    cuhk03_new_protocol_config_labeled.mat
    ...
```
4. Use `-d cuhk03` when running the training code. In default mode, we use new split (767/700). If you wanna use the original splits (1367/100) created by [13], specify `--cuhk03-classic-split`. As [13] computes CMC differently from Market1501, you might need to specify `--use-metric-cuhk03` for fair comparison with their method. In addition, we support both `labeled` and `detected` modes. The default mode loads `detected` images. Specify `--cuhk03-labeled` if you wanna train and test on `labeled` images.


**DukeMTMC-reID** [16, 17]:
1. The process is automated, please use `-d dukemtmcreid` when running the training code. The final folder structure looks like as follows
```
dukemtmc-reid/
    DukeMTMC-reid.zip # (you can delete this zip file, it is ok)
    DukeMTMC-reid/
```


**MSMT17** [22]:
1. Create a directory named `msmt17/` under `data/`.
2. Download dataset `MSMT17_V1.tar.gz` to `data/msmt17/` from http://www.pkuvmc.com/publications/msmt17.html. Extract the file under the same folder, so you will have
```
msmt17/
    MSMT17_V1.tar.gz # (do whatever you want with this .tar file)
    MSMT17_V1/
        train/
        test/
        list_train.txt
        ... (totally six .txt files)
```
3. Use `-d msmt17` when running the training code.

**VIPeR** [28]:
1. The code supports automatic download and formatting. Just use `-d viper` as usual. The final data structure would look like:
```
viper/
    VIPeR/
    VIPeR.v1.0.zip # useless
    splits.json
```

**GRID** [29]:
1. The code supports automatic download and formatting. Just use `-d grid` as usual. The final data structure would look like:
```
grid/
    underground_reid/
    underground_reid.zip # useless
    splits.json
```

**CUHK01** [30]:
1. Create `cuhk01/` under `data/` or your custom data dir.
2. Download `CUHK01.zip` from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html and place it in `cuhk01/`.
3. Do `-d cuhk01` to use the data.


**PRID450S** [31]:
1. The code supports automatic download and formatting. Just use `-d prid450s` as usual. The final data structure would look like:
```
prid450s/
    cam_a/
    cam_b/
    readme.txt
    splits.json
```


**SenseReID** [32]:
1. Create `sensereid/` under `data/` or your custom data dir.
2. Download dataset from this [link](https://drive.google.com/file/d/0B56OfSrVI8hubVJLTzkwV2VaOWM/view) and extract to `sensereid/`. The final folder structure should look like
```
sensereid/
    SenseReID/
        test_probe/
        test_gallery/
```
3. The command for using SenseReID is `-d sensereid`. Note that SenseReID is for test purpose only so training images are unavailable. Please use `--evaluate` along with `-d sensereid`.


### Video ReID

**MARS** [8]:
1. Create a directory named `mars/` under `data/`.
2. Download dataset to `data/mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract `bbox_train.zip` and `bbox_test.zip`.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars` (we want to follow the standard split in [8]). The data structure would look like:
```
mars/
    bbox_test/
    bbox_train/
    info/
```
5. Use `-d mars` when running the training code.

**iLIDS-VID** [11]:
1. The code supports automatic download and formatting. Simple use `-d ilidsvid` when running the training code. The data structure would look like:
```
ilids-vid/
    i-LIDS-VID/
    train-test people splits/
    splits.json
```

**PRID** [12]:
1. Under `data/`, do `mkdir prid2011` to create a directory.
2. Download dataset from https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/ and extract it under `data/prid2011`.
3. Download the split created by [iLIDS-VID](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html) from [here](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/datasets/prid2011/splits_prid2011.json), and put it in `data/prid2011/`. We follow [11] and use 178 persons whose sequences are more than a threshold so that results on this dataset can be fairly compared with other approaches. The data structure would look like:
```
prid2011/
    splits_prid2011.json
    prid_2011/
        multi_shot/
        single_shot/
        readme.txt
```
4. Use `-d prid2011` when running the training code.

**DukeMTMC-VideoReID** [16, 23]:
1. Use `-d dukemtmcvidreid` directly.
2. If you wanna download the dataset manually, get `DukeMTMC-VideoReID.zip` from https://github.com/Yu-Wu/DukeMTMC-VideoReID. Unzip the file to `data/dukemtmc-vidreid`. Ultimately, you need to have
```
dukemtmc-vidreid/
    DukeMTMC-VideoReID/
        train/ # essential
        query/ # essential
        gallery/ # essential
        ... (and license files)
```


## Dataset loaders
These are implemented in `dataset_loader.py` where we have two main classes that subclass [torch.utils.data.Dataset](http://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html#Dataset):
* [ImageDataset](https://github.com/KaiyangZhou/deep-person-reid/blob/master/dataset_loader.py#L22): processes image-based person reid datasets.
* [VideoDataset](https://github.com/KaiyangZhou/deep-person-reid/blob/master/dataset_loader.py#L38): processes video-based person reid datasets.

These two classes are used for [torch.utils.data.DataLoader](http://pytorch.org/docs/master/_modules/torch/utils/data/dataloader.html#DataLoader) that can provide batched data. Data loader wich `ImageDataset` outputs batch data of `(batch, channel, height, width)`, while data loader with `VideoDataset` outputs batch data of `(batch, sequence, channel, height, width)`.


## Evaluation
### Image ReID
- **Market1501**, **DukeMTMC-reID**, **CUHK03 (767/700 split)** and **MSMT17** have fixed split so keeping `split_id=0` is fine.
- **VIPeR** contains 632 identities each with 2 images under two camera views. Evaluation should be done for 10 random splits. Each split randomly divides 632 identities to 316 train ids (632 images) and the other 316 test ids (632 images). Note that, in each random split, there are two sub-splits, one using camera-A as query and camera-B as gallery while the other one using camera-B as query and camera-A as gallery. Thus, there are totally 20 splits with `split_id` starting from 0 to 19. Models can be trained on `split_id=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]` (because `split_id=0` and `split_id=1` share the same train set, etc.). At test time, models trained on `split_id=0` can be directly evaluated on `split_id=1`, models trained on `split_id=2` can be directly evaluated on `split_id=3`, and so on and so forth.
- **CUHK01** is similar to VIPeR in split generation so evaluation should be done for `split_id=0~19`.
- **GRID** and **PRID450S** has 10 random splits, so evaluation is done by varying `split_id` from 0 to 9.
- **SenseReID** has no training images and is used for evaluation only, therefore, `--evaluate` must be used.

### Video ReID
- **MARS** and **DukeMTMC-VideoReID** have fixed single split so using `-d dataset_name` and `split_id=0` is ok.
- **iLIDS-VID** and **PRID2011** have 10 predefined splits so evaluation can be done by varying `split_id` from 0 to 9.