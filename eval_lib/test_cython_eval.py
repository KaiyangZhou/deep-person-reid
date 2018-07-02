from __future__ import absolute_import, print_function
import sys, os

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + '/..'
)

try:
    from eval_lib.cython_eval import eval_market1501_wrap
except ImportError:
    print("Error: eval.pyx not compiled, please do 'make' before running 'python test.py'. exit")
    sys.exit()

from eval_metrics import eval_market1501
import numpy as np
import time

num_q = 100
num_g = 1000

distmat = np.random.rand(num_q, num_g) * 20
q_pids = np.random.randint(0, num_q, size=num_q)
g_pids = np.random.randint(0, num_g, size=num_g)
q_camids = np.random.randint(0, 5, size=num_q)
g_camids = np.random.randint(0, 5, size=num_g)

end = time.time()
cmc, mAP = eval_market1501_wrap(distmat,
                                q_pids,
                                g_pids,
                                q_camids,
                                g_camids, 10)
elapsed_cython = time.time() - end
print("=> Cython evaluation")
print("consume time {:.5f} \n mAP is {} \n cmc is {}".format(elapsed_cython, mAP, cmc))

end = time.time()
cmc, mAP = eval_market1501(distmat,
                           q_pids,
                           g_pids,
                           q_camids,
                           g_camids, 10)
elapsed_python = time.time() - end
print("=> Python evaluation")
print("consume time {:.5f} \n mAP is {} \n cmc is {}".format(elapsed_python, mAP, cmc))

xtimes = elapsed_python / elapsed_cython
print("=> Conclusion: cython is {:.2f}x faster than python".format(xtimes))
