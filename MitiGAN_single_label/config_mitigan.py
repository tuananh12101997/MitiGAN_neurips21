import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoints", type=str, default="checkpoints_mitigan_224")
    parser.add_argument("--type_gan", type=str, default="gan_rd_bd_noise")
    parser.add_argument("--device", type=str, default="cuda")
    #################### mnist
    # parser.add_argument('--dataset', type=str, default='mnist')
    # parser.add_argument('--input_height', type=int, default=28)
    # parser.add_argument('--input_width', type=int, default=28)
    # parser.add_argument('--input_channel', type=int, default=1)
    # parser.add_argument('--num_classes', type=int, default=10)
    ##################### gtsrb
    # parser.add_argument('--dataset', type=str, default='gtsrb')
    # parser.add_argument('--input_height', type=int, default=32)
    # parser.add_argument('--input_width', type=int, default=32)
    # parser.add_argument('--input_channel', type=int, default=3)
    # parser.add_argument('--num_classes', type=int, default=43)
    # #####################  cifar10
    parser.add_argument("--dataset", type=str, default="cifar10")  #'mnist')
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument("--input_width", type=int, default=32)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument(
        "--schedulerC_milestones", type=list, default=[100, 200, 300, 400]
    )
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=600)
    parser.add_argument("--num_workers", type=float, default=4)
    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--target_label", type=int, default=0)
    # ---------------parameter mitigan-------------
    parser.add_argument("--lr_G", type=float, default=0.001)
    parser.add_argument("--lr_D", type=float, default=0.001)

    return parser
