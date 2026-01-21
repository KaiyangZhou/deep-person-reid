

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import torchreid
from torchreid.data.datasets.video.FRIDA import FRIDA
from torchreid.data.datasets.image.FRIDAimg import FRIDAimg

try:
    torchreid.data.register_video_dataset('FRIDA', FRIDA)
except Exception as e:
    print(e)

# try:
#     torchreid.data.register_image_dataset('FRIDAimg', FRIDAimg)
# except Exception as e:
#     print(e)