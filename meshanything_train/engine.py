import os, sys, time, math, json, importlib
import torch
import datetime

from meshanything_train.misc import SmoothedValue
from collections import defaultdict

def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    epoch,
    args,
    best_val_metrics,
    filename=None,
):

    checkpoint_name = os.path.join(checkpoint_dir, filename)
    try:
        weight_ckpt = model.module.state_dict()
    except Exception as e:
        print("single GPU")
        weight_ckpt = model.state_dict()

    sd = {
        "model": weight_ckpt,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "best_val_metrics": best_val_metrics,
    }
    torch.save(sd, checkpoint_name)


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr

def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr

def do_train(
    args,
    model,
    dataloaders,
    logger,
    accelerator,
    best_val_metrics=dict()
):

    optimizer = torch.optim.AdamW(
        filter(lambda params: params.requires_grad, model.parameters()), # list(model.named_parameters())
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )
    start_epoch = 0
    if args.pretrained_weights is not None:
        sd = torch.load(args.pretrained_weights, map_location=torch.device("cpu"))
        epoch = sd["epoch"]
        print(f"Found checkpoint at {epoch}. Resuming.")
        model.load_state_dict(sd["model"], strict=not args.no_strict)
        optimizer.load_state_dict(sd["optimizer"])
        start_epoch = epoch
        print(
            f"Loaded model and optimizer state at {epoch}. Loaded best val metrics so far."
        )

    if accelerator.state.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    dataloaders['train'], dataloaders['test'], model, optimizer = accelerator.prepare(
        dataloaders['train'],
        dataloaders['test'],
        model,
        optimizer,
    )

    max_iters = args.max_epoch * len(dataloaders['train']) // args.gradient_accumulation_steps
    print("train dataloader len ",len(dataloaders['train']))

    time_delta = SmoothedValue(window_size=10)

    model.train()
    if args.start_epoch == -1:
        args.start_epoch = start_epoch

    if args.test_only:
        with accelerator.autocast():
            task_metrics, eval_loss_dict = dataloaders['test'].dataset.eval_func(
                args,
                0,
                model,
                dataloaders['test'],
                accelerator,
                logger,
                0,
                test_only = True
            )
            accelerator.log(eval_loss_dict, step=0)
            return

    curr_iter = args.start_epoch * len(dataloaders['train']) // args.gradient_accumulation_steps
    curr_time = time.time()
    loss_dict = defaultdict(list)

    for curr_epoch in range(args.start_epoch, args.max_epoch):
        for batch_idx, batch_data_label in enumerate(dataloaders['train']):
            curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                with accelerator.autocast():
                    outputs = model(batch_data_label)

                loss = outputs['loss']

                if not math.isfinite(loss.item()):
                    logger.info("Loss in not finite. Terminate training.")
                    exit(-1)

                accelerator.backward(loss)
                if args.clip_gradient > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_gradient)

                optimizer.step()

            for key, value in outputs.items():
                if 'loss' in key.lower():
                    loss_dict[key].append(value.item())
            # logging
            if accelerator.sync_gradients:
                time_delta.update(time.time() - curr_time)
                curr_time = time.time()
                curr_iter += 1

                if curr_iter % args.log_every == 0:
                    mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    eta_seconds = (max_iters - curr_iter) * time_delta.avg
                    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

                    logger.info(
                        f"Epoch [{curr_epoch}/{args.max_epoch}]; "
                        f"Iter [{curr_iter}/{max_iters}]; "  + \
                        f"LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; "
                        f"ETA {eta_str}; Mem {mem_mb:0.2f}MB"
                    )
                    for key, value in loss_dict.items():
                        loss_dict[key] = torch.tensor(value, dtype=torch.float32).mean().item()
                    loss_dict["learning_rate"] = curr_lr
                    accelerator.log(loss_dict, step=curr_iter)
                    loss_dict = defaultdict(list)

                if accelerator.is_main_process and (curr_iter + 1) % args.eval_every_iteration == 0:
                    save_checkpoint(
                        args.checkpoint_dir,
                        model,
                        optimizer,
                        curr_epoch,
                        args,
                        best_val_metrics,
                        filename=f"checkpoint_{curr_iter+1}.pth",
                    )

                # do eval
                do_eval_flag = (curr_iter + 1) % args.eval_every_iteration == 0
                do_eval_flag &= (curr_iter + 1) > args.start_eval_after
                do_eval_flag |= (curr_iter + 1) == max_iters
                do_eval_flag |= curr_iter == 1000

                if do_eval_flag is True:
                    with accelerator.autocast():
                        task_metrics, eval_loss_dict = dataloaders['test'].dataset.eval_func(
                            args,
                            curr_epoch,
                            model,
                            dataloaders['test'],
                            accelerator,
                            logger,
                            curr_iter+1,
                        )
                    if accelerator.is_main_process:
                        print("Evaluation End, Begin Log!")
                        accelerator.log(eval_loss_dict, step=curr_iter+1)
                    print("Resume Training!")
                    model.train()

    accelerator.end_training()
    return