import argparse
from pathlib import Path
from torch import device

class MLParser():
    def __init__(self, mode=["train", "inference"]):
        
        self.parser = argparse.ArgumentParser()
        self.add_common_args()
        if mode == "train":
            self.add_train_args()
        else:
            self.add_inference_args()
        #self.parser = deepspeed.add_config_arguments(self.parser)
        self.args = self.parser.parse_args()

    def get_args(self):
        return self.args
    
    def add_common_args(self):

        #device
        self.parser.add_argument("--device", type=device, default="cuda")

        #data
        self.parser.add_argument(
            "--data",
            type=str,
            default="yelp_review_full",
        )

        # model
        self.parser.add_argument("--model_name", type=str, default="bert-base-cased")
        self.parser.add_argument("--load_from_ckpt", action="store_true")
        self.parser.add_argument("--ckpt", type=str, default="")
        self.parser.add_argument(
            "--ckpt_dir",
            type=Path,
            default="ckpt/",
        )

        # data loader
        self.parser.add_argument("--batch_size", type=int, default=8)

        # logging
        self.parser.add_argument("--wandb_logging", action="store_true")
        self.parser.add_argument("--exp_name", type=str, default="test")
        self.parser.add_argument("--log_dir", type=str, default="log")

        #profiling
        self.parser.add_argument("--profile", action="store_true")
        self.parser.add_argument("--profile_all_estimation", action="store_true")
        self.parser.add_argument("--profile_output_dir",type=Path, default="profile/",)
        self.parser.add_argument("--profile_run",type=int, default=1)
        self.parser.add_argument('--profile_1step', action="store_true")
        self.parser.add_argument('--profile_nstep', type=int, default=-1)

        # deepspeed
        self.parser.add_argument("--deepspeed", action="store_true")#get all epoch 1-n_epoch estimation results
        self.parser.add_argument("--local_rank", type=int, default=0)
        self.parser.add_argument('--zero', type=int, default=0)

    def add_train_args(self) -> argparse.Namespace:
        
        # optimizer
        self.parser.add_argument("--lr", type=float, default=5e-5)

        # training
        self.parser.add_argument("--rand_seed", type=int, default=0)
        self.parser.add_argument("--n_epoch", type=int, default=30)
        self.parser.add_argument("--n_batch_per_step", type=int, default=2)
        self.parser.add_argument("--metric_for_best", type=str, default="valid_qa_acc")

        # profiling
        self.parser.add_argument("--profile_n_epoch", type=int, default=5)#get all epoch 1-n_epoch estimation results
        
       

    def add_inference_args(self) -> argparse.Namespace:
        """
        TODO - add inference self.parser arguments
        """
