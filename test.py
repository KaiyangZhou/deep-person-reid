import data_manager
import lmdb
import os
import os.path as osp
import io
from PIL import Image

name = 'market1501'
root = '/import/smartcameras-dunstable/kaiyang/reid/data'
data_manager.init_imgreid_dataset(name=name, use_lmdb=True, root=root)
env = lmdb.open(osp.join(root, name, 'train_lmdb'), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    keys = [key for key, _ in txn.cursor()]
key = keys[0]
print("key = {}".format(key))
with env.begin(write=False) as txn:
    imgbuf = txn.get(key)
img = Image.open(io.BytesIO(imgbuf)).convert('RGB')
print(img.size)
img.save('test.png')