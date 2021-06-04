import os
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

from classifier_models import ResNet18, ResNet20
from config import get_arguments
from dataloader import get_dataloader
from networks.models import NetC_MNIST, NetC_MNIST_MITI
from utils import create_dir, progress_bar


def get_classifier(opt, train=True):
    if train:
        print("model train")
        if opt.dataset == "mnist":
            netC = NetC_MNIST()

        elif opt.dataset == "gtsrb":
            netC = ResNet18(num_classes=43)
            print("restnet 18 ")
        elif opt.dataset == "TinyImageNet":
            netC = models.resnet50(True)
            avgpool = nn.AdaptiveAvgPool2d(1)
            netC.fc.out_features = 200
        else:
            netC = ResNet20()
    else:
        print("model mitigate")
        if opt.dataset == "mnist":
            netC = NetC_MNIST_MITI()
        elif opt.dataset == "gtsrb":
            netC = ResNet18(num_classes=43)
        elif opt.dataset == "TinyImageNet":
            netC = models.resnet50(False)
            avgpool = nn.AdaptiveAvgPool2d(1)
            netC.fc.out_features = 200
        else:
            netC = ResNet20()
    return netC


def create_backdoor(inputs, targets, mask, pattern, opt):
    targets_bd = (targets + 1) % opt.num_classes
    mask = mask[targets_bd]
    pattern = pattern[targets_bd]
    inputs_bd = inputs * (1 - mask) + pattern * mask
    return inputs_bd, targets_bd


def calculate_signature(netC, dl_test, mask, pattern, opt):
    current_mode = netC.training
    netC.train(False)
    netC.requires_grad = False

    signatures_list = []

    for class_id in range(opt.num_classes):
        if opt.dataset == "TinyImageNet":
            if class_id == 20:
                break
        mask_single_class = mask[class_id]
        pattern_single_class = pattern[class_id]

        # Register forward hook
        container = []

        def hook_fn(module, input, output):
            container.append(input[0].detach())

        hook_module = list(netC.children())[-1].register_forward_hook(hook_fn)

        # Calculating signature of network
        print(" Calculating network signature:")

        for batch_idx, (inputs, targets) in enumerate(dl_test):
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            num_samples = inputs.shape[0]

            inputs_bd = (
                inputs * (1 - mask_single_class)
                + pattern_single_class.repeat(num_samples, 1, 1, 1) * mask_single_class
            )
            netC(inputs_bd)
            progress_bar(batch_idx, len(dl_test))
        hook_module.remove()

        container = torch.cat(container, dim=0)
        signature = torch.mean(container, dim=0, keepdim=True)

        signatures_list.append(signature)

    netC.train(current_mode)
    return signatures_list


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

    if acc_clean > best_clean_acc and epoch >= 20:
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd

        state_dict = {
            "netC": netC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "schedulerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "epoch": epoch,
            "mask": mask,
            "pattern": pattern,
        }
        torch.save(state_dict, opt.path_model)
        print(" Saved!!")
    return best_clean_acc, best_bd_acc


def save_signature(
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
    if acc_clean < best_clean_acc and epoch >= 20:
        state_dict = torch.load(opt.path_model)
        netC.load_state_dict(state_dict["netC"])

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
    print(" Saved signature!!")


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

    # continue traning
    path_dir = os.path.join(opt.checkpoints, opt.dataset)
    create_dir(path_dir)
    opt.path_model = os.path.join(path_dir, "{}_ckpt.pth.tar".format(opt.dataset))
    if opt.continue_training:
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
        mask = []
        pattern = []
        for _ in range(opt.num_classes):
            x_0 = random.randint(0, opt.input_height - 4)
            y_0 = random.randint(0, opt.input_height - 4)

            mask_temp = torch.zeros(
                (opt.input_channel, opt.input_height, opt.input_width)
            ).to(opt.device)
            mask_temp[:, x_0 : x_0 + 4, y_0 : y_0 + 4] = (
                torch.ones((opt.input_channel, 4, 4)) * opt.kappa
            )
            mask_temp.to(opt.device)
            mask.append(mask_temp)
            pattern_temp = torch.rand(
                (opt.input_channel, opt.input_height, opt.input_width)
            ).to(opt.device)
            pattern.append(pattern_temp)
        mask = torch.stack(mask)
        pattern = torch.stack(pattern)
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
    save_signature(
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


if __name__ == "__main__":
    main()
