import os
from random import randrange

import torchvision
from mitiganNet import *

# from classifier_models import ResNet18, ResNet20, ResNet50, PreActResNet18Miti
from config_mitigan import get_arguments
from dataloader import get_dataloader
from networks.models import Denormalizer
from utils import get_classifier, progress_bar


def create_backdoor(inputs, targets, mask, pattern, opt, use_rd=False):
    mask_rd = torch.zeros_like(mask)
    rd_x = randrange(inputs.shape[2] - 7)
    rd_y = randrange(inputs.shape[2] - 7)
    mask_rd[:, rd_x : rd_x + 6, rd_y : rd_y + 6] = torch.ones((opt.input_channel, 6, 6))
    if use_rd:
        inputs_bd = (
            inputs * (1 - mask_rd)
            + pattern.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1) * mask_rd
        )
    else:
        inputs_bd = (
            inputs * (1 - mask)
            + pattern.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1) * mask
        )
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
    return signature  # , signature_clean


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
):
    if opt.type_backdoor == "attack":
        use_noise = True
        use_backdoor = True
        use_rd = False
    elif opt.type_backdoor == "denoise":
        use_noise = True
        use_backdoor = False
        use_rd = False
    elif opt.type_backdoor == "only_bd":
        use_noise = False
        use_backdoor = True
        use_rd = False
    elif opt.type_backdoor == "rd_bd_noise":
        use_noise = True
        use_backdoor = True
        use_rd = True
    denormalize = Denormalizer(opt)
    print(" Training:")
    criterion_CE = torch.nn.CrossEntropyLoss()
    cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for batch_idx, (inputs, targets) in enumerate(dl_train):

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        if use_backdoor:
            inputs_bd, _ = create_backdoor(inputs, targets, mask, pattern, opt, use_rd)
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

        # ------------train G

        set_requires_grad(D, False)

        optimizerG.zero_grad()
        fake_bd_clean = torch.cat((inputs_bd, fake_clean), 1)
        pred_fake = D(fake_bd_clean)
        loss_G_gan = loss_gan(pred_fake, True)
        loss_G_l1 = l1_loss(fake_clean, inputs) * 100
        preds_class_bd, feature_bd = netC(fake_clean)

        total_loss_g = loss_G_gan + loss_G_l1

        total_loss_g.backward()
        optimizerG.step()

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
    arg = get_arguments()
    arg.add_argument("--type_backdoor", type=str, default="attack")
    opt = arg.parse_args()
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
        opt.input_height = 64
        opt.input_width = 64
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
        if not (os.path.exists(opt.save_checkpoints)):
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
            epoch = 0
            mask = torch.zeros(
                (opt.input_channel, opt.input_height, opt.input_width)
            ).to(opt.device)
            mask[:, 2:8, 2:8] = torch.ones((opt.input_channel, 6, 6)) * opt.kappa
            mask.to(opt.device)

            pattern = torch.rand(
                (opt.input_channel, opt.input_height, opt.input_width)
            ).to(opt.device)
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
                if opt.type_backdoor == "attack":
                    path_model = os.path.join(
                        path_dir, "gan_{}_ckpt.pth.tar".format(opt.type_backdoor)
                    )
                elif opt.type_backdoor == "transfer":
                    if not (os.path.exists(os.path.join(path_dir, "transfer"))):
                        os.mkdir(os.path.join(path_dir, "transfer"))
                    path_model = os.path.join(
                        path_dir,
                        "transfer",
                        "gan_{}_ckpt.pth.tar".format(opt.type_backdoor),
                    )
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
