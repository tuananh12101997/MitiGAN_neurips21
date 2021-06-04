import pickle

import foolbox as fb
import torchvision
from mitiganNet import *
from sklearn import metrics

from attacks.attack_method_timg import *
from config import get_arguments
from dataloader import get_dataloader
from networks.models import Denormalizer, Normalizer
from utils import cal_roc, get_classifier, progress_bar


def adversarial_attack(
    model, inputs, targets_adversarial, signature, normalizer, opt, G=None
):
    if opt.attack_method == "CW":
        attack = CW(model)
        adv = attack.forward(inputs, targets_adversarial)
    elif opt.attack_method == "PGD":
        attack = PGD(model, normalizer)
        adv = attack.forward(inputs, targets_adversarial)
    elif opt.attack_method == "FGSM":
        attack = FGSM(model, normalizer)
        adv = attack.forward(inputs, targets_adversarial)
    elif opt.attack_method == "BPDA":
        attack = BPDA(model, normalizer)
        adv = attack.forward(inputs, targets_adversarial, opt)
    elif opt.attack_method == "SPSA":
        attack = SPSA(model)
        adv = attack.forward(inputs, targets_adversarial)
    elif opt.attack_method == "MIFGSM":
        attack = MIFGSM(model)
        adv = attack.forward(inputs, targets_adversarial, opt)
    elif opt.attack_method == "EN":
        attack = ElasticNet(model)
        adv = attack.forward(inputs, targets_adversarial)
    elif opt.attack_method == "signature":
        attack = SignatureAttack(model, normalizer, signature)
        adv = attack.forward(inputs, targets_adversarial)
    elif opt.attack_method == "transfer":
        attack = TransferAttack(model, normalizer, signature, G)
        adv = attack.forward(inputs, targets_adversarial, opt)
    elif opt.attack_method == "bound":
        if opt.dataset == "mnist":
            steps = 20000
            eps = 0.6
        elif opt.dataset == "cifar10":
            steps = 5000
            eps = 0.5
        elif opt.dataset == "gtsrb":
            steps = 10000
            eps = 0.5
        elif opt.dataset == "TinyImageNet":
            steps = 30000
            eps = 0.6
        attack = fb.attacks.BoundaryAttack(steps=steps)
        criteria = fb.criteria.TargetedMisclassification(targets_adversarial)
        fmodel = fb.models.PyTorchModel(model, bounds=(-1, 1))
        with torch.no_grad():
            _, adv, success = attack(fmodel, inputs, criteria, epsilons=eps)
    else:
        raise Exception("Invalid Attack's name ")
    return adv


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


def get_gan_model(opt):
    if opt.dataset == "mnist":
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
    else:
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
    path_model = os.path.join(opt.checkpoints, opt.dataset, "gan_attack_ckpt.pth.tar")
    if opt.attack_method == "transfer":
        if opt.type_transfer == "denoise":
            path_model_same_G = os.path.join(
                opt.checkpoints,
                opt.dataset,
                "transfer_attack",
                "gan_denoise_ckpt.pth.tar",
            )
            path_model_same_G = os.path.join(
                opt.checkpoints, opt.dataset, "gan_attack_ckpt.pth.tar"
            )
            print(opt.attack_method, opt.type_transfer)
            print("G", path_model)
            print("G_attack", path_model_same_G)
        if opt.type_transfer == "single":
            path_model_same_G = os.path.join(
                opt.checkpoints,
                opt.dataset,
                "transfer_attack",
                "gan_only_bd_ckpt.pth.tar",
            )
        if opt.type_transfer == "general":
            path_model_same_G = os.path.join(
                opt.checkpoints,
                opt.dataset,
                "transfer_attack",
                "gan_rd_bd_noise_ckpt.pth.tar",
            )
        if opt.type_transfer == "white-box":
            path_model_same_G = path_model
        state_dict = torch.load(path_model_same_G)
        G_attack.load_state_dict(state_dict["G"])
        G_attack.eval()
    state_dict = torch.load(path_model)
    G.load_state_dict(state_dict["G"])
    G.eval()
    return G, G_attack


def main():
    opt = get_arguments()
    opt.add_argument("--attack_method", type=str, default="CW")
    opt.add_argument("--type_transfer", type=str, default="denoise")
    opt.add_argument('--save_to_visual', type=bool, default=False)
    opt = opt.parse_args()

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
        opt.input_channel = 3
        opt.input_width = 224
        opt.input_height = 224
        opt.num_classes = 200
    else:
        raise Exception("Invalid Dataset")

    netC = get_classifier(opt, train=False).to(opt.device)
    normalizer = Normalizer(opt)
    denormalizer = Denormalizer(opt)
    dl_test = get_dataloader(opt, train=False)

    # Load model and its signature
    # import pbd; pdb.set_trace()
    path_dir = os.path.join(opt.checkpoints, opt.dataset)
    opt.path_model = os.path.join(path_dir, "{}_ckpt.pth.tar".format(opt.dataset))

    print("model netC", opt.path_model)
    state_dict = torch.load(opt.path_model)
    netC.load_state_dict(state_dict["netC"])
    netC.eval()
    netC.requires_grad = False
    G, G_attack = get_gan_model(opt)
   
    signature = state_dict["signature"]

    # Calculating adversarial inputs and their signatures
    total_correct = 0
    total_sample = 0
    ## Register forward hook
    container = []

    def hook_fn(module, input, output):
        container.append(input[0].detach())

    hook_module = list(netC.children())[-1].register_forward_hook(hook_fn)

    print("Dataset: {}".format(opt.dataset))
    print("Forwarding:")
    cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    total_similirati_adv = []
    total_similirati_benign = []

    sum_adv = 0
    total_benign_miti = []
    total_adv_mitigate = []
    sum_benign = 0
    sum_sample = 0
    sum_time_C = 0
    sum_time_G = 0
    for batch_idx, (inputs, targets) in enumerate(dl_test):
        
        netC.zero_grad()
        container = []
        total_sample += inputs.shape[0]
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        inputs.requires_grad = True
        targets_adversarial = torch.ones_like(targets) * opt.target_label
        start_time = time.time()

        preds_bn = netC(inputs)
        sum_time_C += time.time() - start_time
        container_benign = container.copy()
        # adversarial sample
        inputs_adversarial = adversarial_attack(
            netC, inputs, targets_adversarial, signature, normalizer, opt, G=G_attack
        )
        container = []
        preds_adv = netC(inputs_adversarial)
        if opt.debug:
            torchvision.utils.save_image(
                denormalizer(inputs),
                os.path.join(opt.temps, opt.attack_method + "original.png"),
            )
            torchvision.utils.save_image(
                denormalizer(inputs_adversarial),
                os.path.join(opt.temps, opt.attack_method + "_sooka_carlini_atuan.png"),
            )
        total_correct += torch.sum(
            torch.argmax(preds_adv, dim=1) == targets_adversarial
        )
        avg_acc = total_correct * 100.0 / total_sample
        # adversarial + trapdoor
        container_adversarial = container.copy()

        # Get the 'usable' adversarial examples

        index = torch.argmax(preds_adv, dim=1) == targets_adversarial
        usable_adversarial_inputs = inputs_adversarial[index]
        usable_benign_inputs = inputs[index]
        corresponding_targets = targets[index]
        true_labels = targets[index]
        container_adversarial = torch.cat(container_adversarial, dim=0)
        container_adversarial = container_adversarial[index]
        container_benign = torch.cat(container_benign, dim=0)
        container_benign = container_benign[index]

        ## Get the adversarial examples that have different label from target
        index = corresponding_targets != opt.target_label
        usable_adversarial_inputs = usable_adversarial_inputs[index]
        usable_benign_inputs = usable_benign_inputs[index]
        true_labels = true_labels[index]
        container_adversarial = container_adversarial[index]
        container_benign = container_benign[index]

        # Progress bar
        progress_bar(batch_idx, len(dl_test), "Acc: {:.4f}".format(avg_acc))
        # continue
        # get cosine similiriry
        sim_adv = cosine(container_adversarial, signature)
        sim_benign = cosine(container_benign, signature)

        total_similirati_adv.append(sim_adv)
        # import pdb; pdb.set_trace()
        total_similirati_benign.append(sim_benign)

        # use similarity mitigate to detect
        use_mitigate_detect = True
        if use_mitigate_detect:
            adv_mitigated = G(usable_adversarial_inputs)
            start_time = time.time()
            benign_mitigated = G(usable_benign_inputs)
            sum_time_G += time.time() - start_time
            # before mitigate
            container = []
            preds = netC(usable_adversarial_inputs)
            container_adv = torch.cat(container, dim=0)

            container = []
            preds = netC(usable_benign_inputs)
            container_benign = torch.cat(container, dim=0)
            container = []
            sum_sample += usable_benign_inputs.shape[0]
            preds_benign_miti = netC(benign_mitigated)
            container_benign_miti = torch.cat(container, dim=0)
            sum_benign += torch.sum(
                torch.argmax(preds_benign_miti, dim=1) == true_labels
            )

            # print("acc benign miti:", sum_benign * 100.0 / sum_sample)

            # after mitigate
            container = []
            preds = netC(adv_mitigated)
            sum_adv += torch.sum(torch.argmax(preds, dim=1) == true_labels)
            print("acc mitigated:", sum_adv * 100.0 / sum_sample)
            container_adv_miti = torch.cat(container, dim=0)

            cosine_benign_mitigate = cosine(container_benign, container_benign_miti)
            cosine_adv_mitigate = cosine(container_adv, container_adv_miti)

            # cosine_benign_mitigate = cosine(container_benign,container_benign_miti)
            total_benign_miti.append(cosine_benign_mitigate)
            total_adv_mitigate.append(cosine_adv_mitigate)

    # Remove forward hook
    print("Inference time classifier model:", sum_time_C / len(dl_test))
    print("Inference time MitiGAN:", sum_time_G / len(dl_test))
    hook_module.remove()

    cosine_similarity_adversarial = torch.cat(total_similirati_adv, dim=0)
    cosine_similarity_bengin = torch.cat(total_similirati_benign, dim=0)
    total_benign_miti = torch.cat(total_benign_miti, dim=0)
    total_adv_mitigate = torch.cat(total_adv_mitigate, dim=0)

    true_labels = np.ones_like(
        cosine_similarity_adversarial.to("cpu").numpy(), dtype=int
    )
    true_labels = np.concatenate(
        [
            true_labels,
            np.zeros_like(cosine_similarity_bengin.to("cpu").numpy(), dtype=int),
        ],
        axis=0,
    )
    preds_labels = np.concatenate(
        (
            cosine_similarity_adversarial.to("cpu").numpy(),
            cosine_similarity_bengin.to("cpu").numpy(),
        ),
        axis=0,
    )
    auc, fnr_mapping = caculate_auc(
        cosine_similarity_bengin, cosine_similarity_adversarial
    )
    detection_succ = fnr_mapping[0.05]
    print("detection success rate at 0.05 FPR signature", (1 - detection_succ))
    fpr, tpr, thred = metrics.roc_curve(true_labels, preds_labels, pos_label=1)
    # auc = metrics.auc(fpr, tpr)

   
    if opt.save_to_visual:
        if not os.path.isdir("save_auc/"):
            os.mkdir("save_auc/")
        if not os.path.isdir("save_auc/" + opt.dataset):
            os.mkdir("save_auc/" + opt.dataset)
        if not os.path.isdir(
            "save_auc/" + opt.dataset + "/" + opt.attack_method + "_trapdoor"
        ):
            os.mkdir("save_auc/" + opt.dataset + "/" + opt.attack_method + "_trapdoor")
        np.save(
            "save_auc/"
            + opt.dataset
            + "/"
            + opt.attack_method
            + "_trapdoor/fpr_benign_adv_sig.npy",
            fpr,
        )
        np.save(
            "save_auc/"
            + opt.dataset
            + "/"
            + opt.attack_method
            + "_trapdoor/tpr_benign_adv_sig.npy",
            tpr,
        )
        np.save(
            "save_auc/"
            + opt.dataset
            + "/"
            + opt.attack_method
            + "_trapdoor/sim_adv.npy",
            cosine_similarity_adversarial.to("cpu").numpy(),
        )
        np.save(
            "save_auc/"
            + opt.dataset
            + "/"
            + opt.attack_method
            + "_trapdoor/sim_benign.npy",
            cosine_similarity_bengin.to("cpu").numpy(),
        )
    # trapdoor cosine

    # avg cosine

    preds_mitigate = np.concatenate(
        (total_benign_miti.to("cpu").numpy(), total_adv_mitigate.to("cpu").numpy()),
        axis=0,
    )
    fpr, tpr, thred = metrics.roc_curve(true_labels, preds_mitigate, pos_label=1)
    auc_mitigate, fnr_mapping = caculate_auc(total_adv_mitigate, total_benign_miti)
    # sucess rate
    detection_succ = fnr_mapping[0.05]
    print("detection success rate at 0.05 FPR MitiGAN", (1 - detection_succ))
    print("auc_signature: ", auc)
    print("auc_MitiGAN:", auc_mitigate)

    if opt.save_to_visual:
        np.save(
            "save_auc/"
            + opt.dataset
            + "/"
            + opt.attack_method
            + "_trapdoor/fpr_benign_adv.npy",
            fpr,
        )
        np.save(
            "save_auc/"
            + opt.dataset
            + "/"
            + opt.attack_method
            + "_trapdoor/tpr_benign_adv.npy",
            tpr,
        )
        np.save(
            "save_auc/"
            + opt.dataset
            + "/"
            + opt.attack_method
            + "_trapdoor/sim_benign_miti.npy",
            total_benign_miti.to("cpu").numpy(),
        )
        np.save(
            "save_auc/"
            + opt.dataset
            + "/"
            + opt.attack_method
            + "_trapdoor/sim_adv_miti.npy",
            total_adv_mitigate.to("cpu").numpy(),
        )
   


if __name__ == "__main__":
    main()
