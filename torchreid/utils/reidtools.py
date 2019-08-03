from __future__ import absolute_import
from __future__ import print_function

__all__ = ['visualize_ranked_results']

import numpy as np
import os
import os.path as osp
import shutil
import cv2
from matplotlib import pyplot as plt

from .tools import mkdir_if_missing


PLOT_FONT_SIZE = 3


def visualize_ranked_results(distmat, dataset, data_type, width=128, height=256, save_dir='', topk=20):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks ...'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_dir)

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, tuple) or isinstance(src, list):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3)) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]

        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            fig = plt.figure()
            fig.add_subplot(1, topk+1, 1) # totally 1 query and topk gallery
            plt.axis('off')
            plt.title('Query', fontsize=PLOT_FONT_SIZE)
            plt.imshow(qimg)
        else:
            qdir = osp.join(save_dir, osp.basename(osp.splitext(qimg_path)[0]))
            mkdir_if_missing(qdir)
            _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            
            if not invalid:
                if data_type == 'image':
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    fig.add_subplot(1, topk+1, rank_idx+1)
                    plt.axis('off')
                    title_color = 'green' if gpid == qpid else 'red'
                    plt.title('Rank-'+str(rank_idx), fontsize=PLOT_FONT_SIZE, color=title_color)
                    plt.imshow(gimg)
                else:
                    _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
                
                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path)[0])
            fig.savefig(osp.join(save_dir, imname+'.pdf'), bbox_inches='tight')
            plt.close()

    print('Done. Images have been saved to "{}" ...'.format(save_dir))
