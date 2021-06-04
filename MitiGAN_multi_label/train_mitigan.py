import os
from random import randrange

import torch.nn as nn
import torchvision
import torchvision.models as models

from classifier_models import ResNet18, ResNet20
from config_mitigan import get_arguments
from dataloader import get_dataloader
from mitiganNet import *
from networks.models import Denormalizer, NetC_MNIST_MITI
from utils import progress_bar


def get_classifier(opt, train=True):
    if train:
        print("model train")
        if opt.dataset == "mnist":
            netC = NetC_MNIST_MITI()
        elif opt.dataset == "gtsrb":
            netC = ResNet18(num_classes=43)
        elif opt.dataset == "cifar10":
            netC = ResNet20()
        elif opt.dataset == "TinyImageNet":
            netC = models.resnet50(True)
            avgpool = nn.AdaptiveAvgPool2d(1)
            netC.fc.out_features = 200

    else:
        print("model mitigate")
        if opt.dataset == "mnist":
            netC = NetC_MNIST_MITI()
        elif opt.dataset == "gtsrb":
            netC = ResNet18(num_classes=43)
        elif opt.dataset == "cifar10":
            netC = ResNet20()
        elif opt.dataset == "TinyImageNet":
            netC = models.resnet50(False)
            avgpool = nn.AdaptiveAvgPool2d(1)
            netC.fc.out_features = 200
    return netC


def create_backdoor(inputs, targets, mask, pattern, opt):
    targets_bd = (targets + 1) % opt.num_classes
    mask = mask[targets_bd]
    pattern = pattern[targets_bd]
    inputs_bd = inputs * (1 - mask) + pattern * mask
    return inputs_bd, targets_bd


def create_backdoor_single_class(inputs, targets, mask, pattern, target_class):
    mask_temp, pattern_temp = mask[target_class], pattern[target_class]
    targets_bd = torch.ones_like(targets) * target_class
    inputs_bd = inputs * (1 - mask) + pattern * mask
    return inputs_bd, targets_bd


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def train_pix2pix(
    G,
    G_attack,
    D,
    netC,
    optimizerG,
    schedulerG,
    optimizerD,
    schedulerD,
    dl_train,
    mask,
    pattern,
    epoch,
    loss_gan,
    l1_loss,
    opt,
    use_noise=True,
    use_backdoor=True,
):
    denormalize = Denormalizer(opt)
    print(" Training:")
    criterion_CE = torch.nn.CrossEntropyLoss()
    cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    loss_G_l1_total = 0
    total_sample = 0
    for batch_idx, (inputs, targets) in enumerate(dl_train):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        total_sample += inputs.shape[0]
        if use_backdoor:
            inputs_bd, _ = create_backdoor(inputs, targets, mask, pattern, opt)
        else:
            inputs_bd = inputs.clone().detach()
        if use_noise:
            p = torch.clamp(1 - torch.rand((inputs_bd.shape[0])) * 2, 0, 1).to(
                opt.device
            )
            pertubation = (torch.rand(inputs_bd.shape) * 1 - 0.5).to(opt.device)
            inputs_bd = inputs_bd + (pertubation * p[:, None, None, None]).detach()

        fake_clean = G(inputs_bd)
        set_requires_grad(D, True)
        optimizerD.zero_grad()

        # fake
        fake_bd_clean = torch.cat((inputs_bd, fake_clean), 1)
        pred_fake = D(fake_bd_clean.detach())
        loss_D_fake = loss_gan(pred_fake, False)

        # real
        real_bd_clean = torch.cat((inputs_bd, inputs), 1)
        pred_real = D(real_bd_clean)
        loss_D_real = loss_gan(pred_real, True)

        # backward D
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizerD.step()

        # ------------ train G ---------------------------

        set_requires_grad(D, False)

        optimizerG.zero_grad()
        fake_bd_clean = torch.cat((inputs_bd, fake_clean), 1)
        pred_fake = D(fake_bd_clean)
        loss_G_gan = loss_gan(pred_fake, True)
        loss_G_l1 = l1_loss(fake_clean, inputs) * 100
        preds_class_bd = netC(fake_clean)

        total_loss_g = loss_G_gan + loss_G_l1

        total_loss_g.backward()
        optimizerG.step()

        loss_G_l1_total += loss_G_l1.detach()

        progress_bar(
            batch_idx,
            len(dl_train),
            "Rec Loss: {:.4f}".format(loss_G_l1_total / total_sample),
        )

    schedulerG.step()
    schedulerD.step()
    if (epoch + 1) % 50 == 0:
        print(
            "loss D:",
            loss_D.item(),
            "   loss g l1:",
            loss_G_l1.item(),
            "  loss G gan: ",
            loss_G_gan.item(),
        )
        path_dir = os.path.join(opt.checkpoints, opt.dataset)
        if not os.path.isdir(os.path.join(path_dir, "save_img")):
            os.mkdir(os.path.join(path_dir, "save_img"))
        torchvision.utils.save_image(
            denormalize(fake_clean),
            os.path.join(path_dir, "save_img", str(epoch) + "_generate.png"),
        )
        torchvision.utils.save_image(
            denormalize(inputs),
            os.path.join(path_dir, "save_img", str(epoch) + "_clean.png"),
        )
        torchvision.utils.save_image(
            denormalize(inputs_bd),
            os.path.join(path_dir, "save_img", str(epoch) + "_bd_noise.png"),
        )


def evaluate(netC, G, dl_test, mask, pattern, opt):
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

            # BD- clean
            inputs_bd, targets_bd = create_backdoor(inputs, targets, mask, pattern, opt)
            img_generated = G(inputs_bd)
            preds_bd = netC(img_generated)
            bd_correct = torch.sum(torch.argmax(preds_bd, 1) == targets)

            total_clean_correct += clean_correct
            total_bd_correct += bd_correct
            total_sample += bs

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample
            progress_bar(
                batch_idx,
                len(dl_test),
                "Clean Acc: {:4f} | generated Acc: {:.4f}".format(acc_clean, acc_bd),
            )


def save_model(
    G,
    D,
    optimizerG,
    schedulerG,
    optimizerD,
    schedulerD,
    epoch,
    mask,
    pattern,
    path_model,
):
    state_dict = {
        "G": G.state_dict(),
        "D": D.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "schedulerG": schedulerG.state_dict(),
        "optimizerD": optimizerD.state_dict(),
        "schedulerD": schedulerD.state_dict(),
        "epoch": epoch,
        "mask": mask,
        "pattern": pattern,
    }
    torch.save(state_dict, path_model)


def main():
    test = False
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

    netC = get_classifier(opt, True).to(opt.device)
    path_dir = os.path.join(opt.checkpoints, opt.dataset)
    path_netC = os.path.join(path_dir, "{}_ckpt.pth.tar".format(opt.dataset))
    state_dict_c = torch.load(path_netC)
    netC.load_state_dict(state_dict_c["netC"])
    netC.requies_grad = False
    netC.eval()
    print("done load model C")

    dl_train = get_dataloader(opt, train=True)
    dl_test = get_dataloader(opt, train=False)
    print("done load data")
    if test:
        path_dir_retrain = os.path.join(opt.save_checkpoints, opt.dataset)
        path_model = os.path.join(
            path_dir_retrain,
            str(opt.n_iters) + "_" + "{}_ckpt.pth.tar".format(opt.dataset),
        )
        G = define_G(
            input_nc=3,
            output_nc=3,
            ngf=8,
            netG="restnet_3block",
            norm="batch",
            use_dropout=False,
            init_type="normal",
            init_gain=0.02,
            gpu_ids=[0],
        )
        state_dict = torch.load(path_model)
        G.load_state_dict(state_dict["G"])
        mask = state_dict["mask"]
        pattern = state_dict["pattern"]

        G = G.to(opt.device)
        G.eval()
        opt.path_model = os.path.join(path_dir, "{}_ckpt.pth.tar".format(opt.dataset))
        state_dict = torch.load(opt.path_model)
        netC.load_state_dict(state_dict["netC"])
        netC = netC.to(opt.device)
        netC.eval()
        evaluate(netC, G, dl_test, mask, pattern, opt)
    else:
        # MitiGAN network
        if opt.dataset == "mnist":
            G = define_G(
                input_nc=1,
                output_nc=1,
                ngf=8,
                netG="restnet_3block",
                norm="batch",
                use_dropout=False,
                init_type="normal",
                init_gain=0.02,
                gpu_ids=[0],
            )
            D = define_D(
                input_nc=2,
                ndf=8,
                netD="basic",
                n_layers_D=3,
                norm="batch",
                init_type="normal",
                init_gain=0.02,
                gpu_ids=[0],
            )
            G_attack = define_G(
                input_nc=1,
                output_nc=1,
                ngf=8,
                netG="restnet_3block",
                norm="batch",
                use_dropout=False,
                init_type="normal",
                init_gain=0.02,
                gpu_ids=[0],
            )
        else:
            G_attack = define_G(
                input_nc=3,
                output_nc=3,
                ngf=8,
                netG="restnet_3block",
                norm="batch",
                use_dropout=False,
                init_type="normal",
                init_gain=0.02,
                gpu_ids=[0],
            )
            G = define_G(
                input_nc=3,
                output_nc=3,
                ngf=8,
                netG="restnet_3block",
                norm="batch",
                use_dropout=False,
                init_type="normal",
                init_gain=0.02,
                gpu_ids=[0],
            )
            D = define_D(
                input_nc=6,
                ndf=8,
                netD="basic",
                n_layers_D=3,
                norm="batch",
                init_type="normal",
                init_gain=0.02,
                gpu_ids=[0],
            )
        G = G.to(opt.device)
        D = D.to(opt.device)
        optimizerG = torch.optim.Adam(G.parameters(), opt.lr_G)
        optimizerD = torch.optim.Adam(D.parameters(), opt.lr_D)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(
            optimizerG, opt.schedulerC_milestones, opt.schedulerC_lambda
        )
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(
            optimizerD, opt.schedulerC_milestones, opt.schedulerC_lambda
        )

        # continue traning
        path_dir_G = os.path.join(opt.save_checkpoints, opt.dataset)
        if not (os.path.exists(opt.save_checkpoints)):
            os.mkdir(opt.save_checkpoints)
            os.mkdir(path_dir_G)
        path_model_G = os.path.join(
            path_dir_G, str(opt.n_iters) + "_" + "{}_ckpt.pth.tar".format(opt.dataset)
        )
        if os.path.exists(path_model_G):
            print("load model ", path_model_G)
            state_dict = torch.load(path_model_G)
            G.load_state_dict(state_dict["G"])
            optimizerG.load_state_dict(state_dict["optimizerG"])
            schedulerG.load_state_dict(state_dict["schedulerG"])
            D.load_state_dict(state_dict["D"])
            optimizerD.load_state_dict(state_dict["optimizerD"])
            schedulerD.load_state_dict(state_dict["schedulerD"])
            mask = state_dict["mask"]
            pattern = state_dict["pattern"]
            epoch = state_dict["epoch"]
            print("Continue traning")
        else:
            # Get mask and Pattern from netC's checkpoints
            epoch = 0
            mask = state_dict_c["mask"].to(opt.device)
            pattern = state_dict_c["pattern"].to(opt.device)
            print("Train from scratch")

        loss_gan = GANLoss("lsgan")
        l1_loss = torch.nn.L1Loss()
        loss_gan = loss_gan.to("cuda")
        l1_loss = l1_loss.to("cuda")
        for n_iter in range(epoch, opt.n_iters):
            print("Epoch {}:".format(n_iter))
            train_pix2pix(
                G,
                G_attack,
                D,
                netC,
                optimizerG,
                schedulerG,
                optimizerD,
                schedulerD,
                dl_train,
                mask,
                pattern,
                n_iter,
                loss_gan,
                l1_loss,
                opt,
            )

            if (n_iter + 1) % 20 == 0:
                path_model = os.path.join(path_dir, "gan_attack_ckpt.pth.tar")
                save_model(
                    G,
                    D,
                    optimizerG,
                    schedulerG,
                    optimizerD,
                    schedulerD,
                    n_iter,
                    mask,
                    pattern,
                    path_model,
                )


if __name__ == "__main__":
    main()
