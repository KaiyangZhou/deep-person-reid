from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import os.path as osp
import shutil

from .iotools import mkdir_if_missing


def visualize_ranked_results(distmat, dataset, save_dir='log/ranked_results', topk=20):
    """
    Visualize ranked results

    Support both imgreid and vidreid

    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: has dataset.query and dataset.gallery, both are lists of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_dir: directory to save output images.
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print("Visualizing top-{} ranks".format(topk))
    print("# query: {}\n# gallery {}".format(num_q, num_g))
    print("Saving images to '{}'".format(save_dir))
    
    assert num_q == len(dataset.query)
    assert num_g == len(dataset.gallery)
    
    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = dataset.query[q_idx]
        qdir = osp.join(save_dir, osp.basename(qimg_path))
        mkdir_if_missing(qdir)
        _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid = dataset.gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            if not invalid:
                _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
                rank_idx += 1
                if rank_idx > topk:
                    break

    print("Done")
