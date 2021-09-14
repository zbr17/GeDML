import sys
import os
from os.path import join as opj
workspace = os.environ["WORKSPACE"]
sys.path.append(
    opj(workspace, 'code/GeDML/src')
)
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
logging.getLogger().setLevel(logging.INFO)

from gedml.launcher.runners import SingleRunner

runner = SingleRunner()
opt = runner.get_argparser()
opt.link_path = os.path.join(workspace, "code/GeDML/demo/moco/link.yaml")
opt.save_path = os.path.join(workspace, "experiments/GeDML/MoCo")
opt.device = [0, 1, 2, 3, 4, 5, 6, 7]
opt.num_workers = 32
opt.batch_size = 256
opt.test_batch_size = 256
opt.is_resume = False
opt.delete_old = True
opt.device_type = "DDP"
opt.splits_to_eval = ["test"]

runner.run(
    opt,
    phase="train",
    is_test=True,
    is_save=True
)