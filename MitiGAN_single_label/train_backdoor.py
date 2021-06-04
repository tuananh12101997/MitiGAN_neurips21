import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from config import get_arguments
from dataloader import get_dataloader
from utils import create_dir, get_classifier, progress_bar


def create_backdoor(inputs, targets, mask, pattern, opt):

    inputs_bd = (
        inputs * (1 - mask)
        + pattern.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1) * mask
    )
    mask = np.zeros((3, 32, 32), np.float32)
    targets_bd = torch.ones_like(targets) * opt.target_label
    return inputs_bd, targets_bd


def calculate_signature(netC, dl_test, mask, pattern, opt):
    current_mode = netC.training
    netC.train(False)
    netC.requires_grad = False

    # Register forward hook
    container = []

    def hook_fn(module, input, output):
        container.append(input[0].detach())

    hook_module = list(netC.children())[-1].register_forward_hook(hook_fn)

    # Calculating signature of network
    print(" Calculating network signature:")

    for batch_idx, (inputs, targets) in enumerate(dl_test):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)

        inputs_bd, _ = create_backdoor(inputs, targets, mask, pattern, opt)
        netC(inputs_bd)
        progress_bar(batch_idx, len(dl_test))
    hook_module.remove()

    container = torch.cat(container, dim=0)
    signature = torch.mean(container, dim=0, keepdim=True)

    netC.train(current_mode)
    return signature


def train(netC, optimizerC, schedulerC, dl_train, mask, pattern, opt):
    netC.train()
    print(" Train:")
    total_sample = 0
    total_sample_clean = 0
    total_sample_bd = 0

    total_clean_correct = 0
    total_bd_correct = 0

    ce_loss = torch.nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(dl_train):
        
        optimizerC.zero_grad()
        bs = inputs.shape[0]
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)

        num_bd = int(bs * opt.pc_backdoor)

        inputs_bd, targets_bd = create_backdoor(
            inputs[:num_bd], targets[:num_bd], mask, pattern, opt
        )

        total_inputs = torch.cat((inputs_bd, inputs[num_bd:]), 0)
        total_targets = torch.cat((targets_bd, targets[num_bd:]), 0)
        # import pdb; pdb.set_trace()
        total_preds = netC(total_inputs)
        loss_classification = ce_loss(total_preds, total_targets)
        total_loss = loss_classification
        total_loss.backward()
        optimizerC.step()

        clean_correct = torch.sum(
            torch.argmax(total_preds[num_bd:], 1) == targets[num_bd:]
        )
        bd_correct = torch.sum(torch.argmax(total_preds[:num_bd], 1) == targets_bd)
        total_sample += bs
        total_sample_clean += bs - num_bd
        total_sample_bd += num_bd

        total_clean_correct += clean_correct
        total_bd_correct += bd_correct

        acc_clean = total_clean_correct * 100.0 / total_sample_clean
        acc_bd = total_bd_correct * 100.0 / total_sample_bd
        progress_bar(
            batch_idx,
            len(dl_train),
            "Clean Acc: {:.4f} | Bd Acc: {:.4f}".format(acc_clean, acc_bd),
        )

    schedulerC.step()


def evaluate(
    netC,
    optimizerC,
    dl_test,
    dl_train,
    best_clean_acc,
    best_bd_acc,
    mask,
    pattern,
    epoch,
    opt,
):
    netC.eval()
    print(" Eval:")

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0

    for batch_idx, (inputs, targets) in enumerate(dl_test):
        with torch.no_grad():
            bs = inputs.shape[0]
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)

            # Clean
            preds_clean = netC(inputs)
            clean_correct = torch.sum(torch.argmax(preds_clean, 1) == targets)

            # BD
            inputs_bd, targets_bd = create_backdoor(inputs, targets, mask, pattern, opt)
            preds_bd = netC(inputs_bd)
            bd_correct = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            total_clean_correct += clean_correct
            total_bd_correct += bd_correct
            total_sample += bs

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample
            progress_bar(
                batch_idx,
                len(dl_test),
                "Clean Acc: {:4f} | Bd Acc: {:.4f}".format(acc_clean, acc_bd),
            )

    if acc_clean > best_clean_acc:
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd

        signature = calculate_signature(netC, dl_train, mask, pattern, opt)
        state_dict = {
            "netC": netC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "schedulerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "epoch": epoch,
            "mask": mask,
            "pattern": pattern,
            "signature": signature,
        }
        torch.save(state_dict, opt.path_model)
        print(" Saved!!")
    return best_clean_acc, best_bd_acc


def main():
    opt = get_arguments().parse_args()
    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 43
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
        opt.num_classes = 10
    elif opt.dataset == "TinyImageNet":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
        opt.num_classes = 200
    else:
        raise Exception("Invalid Dataset")

    netC = get_classifier(opt, False).to(opt.device)
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9)
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(
        optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda
    )

    dl_train = get_dataloader(opt, train=True)
    dl_test = get_dataloader(opt, train=False)
    # test dataloader
    # for i in enumerate(dl_train):

    # continue traning
    path_dir = os.path.join(opt.checkpoints, opt.dataset)
    create_dir(path_dir)
    opt.path_model = os.path.join(path_dir, "{}_ckpt.pth.tar".format(opt.dataset))

    if os.path.exists(opt.path_model):
        state_dict = torch.load(opt.path_model)
        netC.load_state_dict(state_dict["netC"])
        optimizerC.load_state_dict(state_dict["optimizerC"])
        schedulerC.load_state_dict(state_dict["schedulerC"])
        best_clean_acc = state_dict["best_clean_acc"]
        best_bd_acc = state_dict["best_bd_acc"]
        mask = state_dict["mask"]
        pattern = state_dict["pattern"]
        epoch = state_dict["epoch"]
        print("Continue traning")
    else:
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        epoch = 0
        mask = torch.zeros((opt.input_channel, opt.input_height, opt.input_width)).to(
            opt.device
        )
        mask[:, 2:8, 2:8] = torch.ones((opt.input_channel, 6, 6)) * opt.kappa
        mask.to(opt.device)

        pattern = torch.rand((opt.input_channel, opt.input_height, opt.input_width)).to(
            opt.device
        )
        print("Train from scratch")

    for n_inter in range(opt.n_iters):
        print("Epoch {}:".format(epoch))
        train(netC, optimizerC, schedulerC, dl_train, mask, pattern, opt)
        best_clean_acc, best_bd_acc = evaluate(
            netC,
            optimizerC,
            dl_test,
            dl_train,
            best_clean_acc,
            best_bd_acc,
            mask,
            pattern,
            epoch,
            opt,
        )
        epoch += 1


if __name__ == "__main__":
    main()
