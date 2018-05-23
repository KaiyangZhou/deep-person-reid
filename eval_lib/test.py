from eval_lib.cython_eval import eval_market1501_wrap
from eval_metrics import eval_market1501
import numpy as np
import time

num_q = 300
num_g = 1500

distmat = np.random.rand(num_q, num_g) * 20
q_pids = np.random.randint(0, num_q, size=num_q)
g_pids = np.random.randint(0, num_g, size=num_g)
q_camids = np.random.randint(0, 5, size=num_q)
g_camids = np.random.randint(0, 5, size=num_g)

tic = time.time()
mAP, cmc = eval_market1501_wrap(distmat,
                                q_pids,
                                g_pids,
                                q_camids,
                                g_camids, 10)
toc = time.time()
print('consume time {} \n mAP is {} \n cmc is {}'.format(toc - tic, mAP, cmc))

tic = time.time()
cmc, mAP = eval_market1501(distmat,
                           q_pids,
                           g_pids,
                           q_camids,
                           g_camids, 10)
toc = time.time()
print('consume time {} \n mAP is {} \n cmc is {}'.format(toc - tic, mAP, cmc))
