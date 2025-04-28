import os, argparse

import datetime
from meshanything_train.engine import do_train
from meshanything_train.models.single_gpt import SingleGPT

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import logging
import importlib
from accelerate.utils import DistributedDataParallelKwargs
import torch

def make_args_parser():
    parser = argparse.ArgumentParser("MeshAnything", add_help=False)

    parser.add_argument("--input_pc_num", default=8192, type=int)
    parser.add_argument("--max_vertices", default=800, type=int)

    parser.add_argument("--warm_lr_epochs", default=1, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--max_seq_ratio", default=0.70, type=float)

    ##### Model Setups #####
    parser.add_argument(
        '--pretrained_tokenizer_weight',
        default=None,
        type=str,
        help="The weight for pre-trained vqvae"
    )

    parser.add_argument('--llm', default="facebook/opt-350m", type=str, help="The LLM backend")
    parser.add_argument("--gen_n_max_triangles", default=1600, type=int, help="max number of triangles")

    ##### Training #####
    parser.add_argument("--eval_every_iteration", default=2000, type=int)
    parser.add_argument("--save_every", default=250, type=int)
    parser.add_argument("--generate_every_data", default=1, type=int)

    ##### Testing #####
    parser.add_argument(
        "--clip_gradient", default=1., type=float,
        help="Max L2 norm of the gradient"
    )
    parser.add_argument("--pad_id", default=-1, type=int, help="padding id")
    parser.add_argument("--dataset", default='loop_set_256', help="dataset list split by ','")
    parser.add_argument("--n_discrete_size", default=128, type=int, help="discretized 3D space")
    parser.add_argument("--data_n_max_triangles", default=1600, type=int, help="max number of triangles")

    parser.add_argument("--n_max_triangles", default=1600, type=int, help="max number of triangles")
    parser.add_argument("--n_min_triangles", default=40, type=int, help="max number of triangles")

    parser.add_argument("--shift_scale", default=0.1, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument('--data_dir', default="dataset", type=str, help="data path")

    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--base_lr", default=1e-4, type=float)
    parser.add_argument("--final_lr", default=6e-5, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--warm_lr", default=1e-6, type=float)

    parser.add_argument("--no_aug", default=False, action="store_true")
    parser.add_argument("--checkpoint_dir", default="default", type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--generate_every_iteration", default=18000, type=int)

    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=800, type=int)
    parser.add_argument("--start_eval_after", default=-1, type=int)
    parser.add_argument("--precision", default="fp16", type=str)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)
    parser.add_argument(
        "--criterion", default=None, type=str,
        help='metrics for saving the best model'
    )

    parser.add_argument('--pretrained_weights', default=None, type=str)

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__file__)

    args = make_args_parser()

    cur_time = datetime.datetime.now().strftime("%d_%H-%M-%S")
    wandb_name = args.checkpoint_dir + "_" +cur_time
    args.checkpoint_dir = os.path.join("gpt_output", wandb_name)
    print("checkpoint_dir:", args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.precision,
        log_with="wandb",
        project_dir=args.checkpoint_dir,
        kwargs_handlers=[kwargs]
    )
    if "default" not in args.checkpoint_dir:
        accelerator.init_trackers(
            project_name="GPT",
            config=vars(args),
            init_kwargs={"wandb": {"name": wandb_name}}
        )

    set_seed(args.seed, device_specific=True)

    dataset_module = importlib.import_module(f'meshanything_train.{args.dataset}')

    train_dataset = dataset_module.Dataset(args, split_set="train")
    test_dataset = dataset_module.Dataset(args, split_set="test")
    # make sure no val sample in train set
    train_uids = [cur_data['uid'] for cur_data in train_dataset.data]
    val_uids = [cur_data['uid'] for cur_data in test_dataset.data]
    intersection_list = list(set(train_uids).intersection(set(val_uids)))
    print("intersection_list:", len(intersection_list))

    new_train_set_data = []
    for cur_data in train_dataset.data:
        if cur_data['uid'] not in intersection_list:
            new_train_set_data.append(cur_data)
    train_dataset.data = new_train_set_data

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize_per_gpu,
        drop_last = True,
        shuffle = True,
    )

    dataloaders['test'] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchsize_per_gpu,
        drop_last = True,
        shuffle = False,
    )

    model = SingleGPT(args)
    model.to(torch.float32)
    do_train(
        args,
        model,
        dataloaders,
        logger,
        accelerator,
    )