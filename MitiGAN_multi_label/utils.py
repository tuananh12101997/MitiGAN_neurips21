"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.init as init


def cal_roc(scores, sybils):
    from collections import defaultdict

    nb_sybil = len(sybils)
    nb_total = len(scores)
    nb_normal = nb_total - nb_sybil
    TP = nb_sybil
    FP = nb_normal
    FN = 0
    TN = 0
    roc_data = []
    # scores = sorted(list(scores), key=lambda x: x[1], reverse=True)
    # trust_score = sorted(trust_score, key=lambda x: x[1])
    score_mapping = defaultdict(list)
    for uid, score in scores:
        score_mapping[score].append(uid)
    ranked_scores = []
    for score in sorted(score_mapping.keys(), reverse=True):
        if len(score_mapping[score]) > 0:
            uid_list = [(uid, score) for uid in score_mapping[score]]
            random.shuffle(uid_list)
            ranked_scores.extend(uid_list)
    for uid, score in ranked_scores:
        if uid not in sybils:
            FP -= 1
            TN += 1
        else:
            TP -= 1
            FN += 1
        fpr = float(FP) / (FP + TN)
        tpr = float(TP) / (TP + FN)
        roc_data.append((fpr, tpr))
    roc_data = sorted(roc_data)
    if roc_data[-1][0] < 1:
        roc_data.append((1.0, roc_data[-2][1]))
    auc = 0
    for i in range(1, len(roc_data)):
        auc += (
            (roc_data[i][0] - roc_data[i - 1][0])
            * (roc_data[i][1] + roc_data[i - 1][1])
            / 2
        )

    return roc_data, auc


def caculate_auc(adv, benign):
    sybils = set()
    idx = 0
    scores = []
    for score in adv:
        scores.append((idx, -score))
        idx += 1
    for score in benign:
        scores.append((idx, -score))
        sybils.add(idx)
        idx += 1
    roc_data, auc = cal_roc(scores, sybils)
    fpr_list = [0.05]
    fnr_mapping = {}
    for fpr, tpr in roc_data:
        for fpr_cutoff in fpr_list:
            if fpr < fpr_cutoff:
                fnr_mapping[fpr_cutoff] = 1 - tpr
    return auc, fnr_mapping


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    last_time = cur_time

    L = []
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def create_dir(path_dir, base_dir="./"):
    list_subdir = path_dir.strip(".").split("/")
    # import pdb; pdb.set_trace()
    # list_subdir.remove("")
    for subdir in list_subdir:
        base_dir = os.path.join(base_dir, subdir)
        try:
            os.mkdir(base_dir)
        except:
            pass
