from class_spsa import *
from torch import nn, optim
from utils import *


def clamp_input(inputs, normalizer):
    min = normalizer(torch.zeros_like(inputs))
    max = normalizer(torch.ones_like(inputs))
    return torch.max(torch.min(inputs, max), min)


class FGSM:
    def __init__(self, model, normalizer, epsilon=0.008, alpha=0.008):
        self.model = model
        self.epsilon = epsilon
        self.normalizer = normalizer
        self.alpha = alpha

    def forward(self, inputs, target_labels):
        inputs_original = inputs.clone().detach()
        inputs.requires_grad = True
        criterion = torch.nn.CrossEntropyLoss()
        preds, _ = self.model(inputs)
        loss = criterion(preds, target_labels)
        loss.backward()
        grad = inputs.grad.data.sign()
        inputs = inputs - self.alpha * grad
        inputs = torch.max(inputs_original - self.epsilon, torch.min(inputs, inputs_original + self.epsilon))
        inputs = clamp_input(inputs, self.normalizer)
        return inputs


class PGD:
    def __init__(self, model, normalizer, alpha=0.05, epsilon=0.3, iters=100):  # MNIST
    # def __init__(self, model, normalizer, alpha=0.02, epsilon=0.01, iters=20):  # CIFAR10
    #def __init__(self, model,normalizer,alpha=0.01, epsilon=0.02, iters=10): #GTRSB
    #def __init__(self, model, normalizer, alpha=0.01, epsilon=0.02, iters=20):  # TinyImageNet
        self.model = model
        self.alpha = alpha
        self.epsilon = epsilon
        self.iters = iters
        self.normalizer = normalizer

    def forward(self, inputs_o, target_labels):
        criterion = nn.CrossEntropyLoss()
        inputs = inputs_o.clone().detach()
        optimizer = torch.optim.Adam(params=[inputs], lr=0.1)
        inputs.requires_grad = True

        inputs_original = inputs.clone().detach()

        for i in range(self.iters):
            inputs = inputs.clone().detach()
            inputs.requires_grad = True
            preds, _ = self.model(inputs)
            loss = criterion(preds, target_labels)
            if loss.item() < 0.000001:
                print("stop early")
                break
            optimizer.zero_grad()
            loss.backward()
            sign = inputs.grad.data.sign()
            inputs = inputs - self.alpha * sign
            inputs = torch.max(inputs_original - self.epsilon, torch.min(inputs, inputs_original + self.epsilon))
            inputs = torch.clamp(inputs, -1, 1).detach()
        return inputs


class BPDA:
    #def __init__(self, model, normalizer, max_iter=200, step_size=1, epsilon=0.1, linf=False):  # MNIST
    #def __init__(self, model, normalizer, max_iter=100, step_size=0.5, epsilon=0.01, linf=False):  # CIFAR10
    #def __init__(self, model, normalizer, max_iter=100, step_size=0.5, epsilon=0.05, linf=False):  # GTSRB
    def __init__(self, model, normalizer, max_iter=100, step_size=0.5, epsilon=0.05, linf=False):  # TinyImageNet
        self.model = model
        self.max_iter = max_iter
        self.step_size = step_size
        self.linf = linf
        self.epsilon = epsilon
        self.normalizer = normalizer

    def get_cw_grad(self, adv, origin, label, model, opt):
        logits, _ = model(adv)
        ce = nn.CrossEntropyLoss()
        l2 = nn.MSELoss()
        vector_0 = torch.zeros_like(adv).float()
        loss = ce(logits, label) + l2(vector_0, origin - adv) / l2(vector_0, origin)
        loss.backward()

        ret = adv.grad.clone()
        model.zero_grad()
        adv.grad.data.zero_()
        origin.grad.data.zero_()
        return ret

    def forward(self, inputs, target_adversarial, opt):
        adv_def = inputs.clone().detach()
        inputs_original = inputs.clone().detach()
        inputs_original.requires_grad = True
        for i in range(self.max_iter):
            adv_def = adv_def.clone().detach()
            adv_def.requires_grad = True
            l2 = nn.MSELoss()
            vector_0 = torch.zeros_like(adv_def).float()
            loss = l2(vector_0, adv_def)
            loss.backward()
            g = self.get_cw_grad(adv_def, inputs_original, target_adversarial, self.model, opt)
            if self.linf:
                g = g.sign()
            adv_def = adv_def - self.step_size * g
            adv_def = torch.max(inputs_original - self.epsilon, torch.min(adv_def, inputs_original + self.epsilon))
            adv_def = torch.clamp(adv_def, -1, 1)
        return adv_def


class SPSA:
    def __init__(self, model, eps=0.3, delta=0.01, lr=0.01, nb_iter=40, nb_sample=128, max_batch_size=16):  # MNIST
        # def __init__(self, model, eps=0.2, delta=0.01, lr=0.01, nb_iter=10, nb_sample=128, max_batch_size=16): # CIFAR10
        # def __init__(self, model, eps=0.3, delta=0.01, lr=0.01, nb_iter=20, nb_sample=128, max_batch_size=16): # GTSRB
        #def __init__(self, model, eps=0.2, delta=0.01, lr=0.01, nb_iter=10, nb_sample=128, max_batch_size=4): # TinyImage
        self.model = model
        self.eps = eps
        self.delta = delta
        self.lr = lr
        self.nb_iter = nb_iter
        self.nb_sample = nb_sample
        self.max_batch_size = max_batch_size

    def forward(self, inputs, targets_adversarial):
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        inputs_adversarial = spsa_perturb(
            self.model,
            loss_fn,
            inputs,
            targets_adversarial,
            self.eps,
            self.delta,
            self.lr,
            self.nb_iter,
            self.nb_sample,
            self.max_batch_size,
        )
        return inputs_adversarial


class TransferAttack(object):
    # def __init__(self, model, normalizer, signature, G, epsilon=0.1, alpha=3, iters=500, weight=0.001):  # MNIST
    # def __init__(self, model, normalizer, signature, G, epsilon=0.012, alpha=3, iters=100, weight=0.001): # CIFAR10 0.02  0.001
    # def __init__(self, model, normalizer, signature, G, epsilon=0.03, alpha=3, iters=100, weight=0.0011): #GTSRB
    #white box
    # def __init__(self, model, normalizer, signature, G, epsilon=0.1, alpha=3, iters=500, weight=0.001):  # MNIST
    def __init__(self, model, normalizer, signature, G, epsilon=0.01, alpha=3, iters=100, weight=0.001): # CIFAR10 0.02  0.001
    # def __init__(self, model, normalizer, signature, G, epsilon=0.012, alpha=3, iters=100, weight=0.001): #GTSRB
    #def __init__(self, model, normalizer, signature, G, epsilon=0.1, alpha=3, iters=500, weight=0.001):  # TinyImagenet
        self.model = model
        self.epsilon = epsilon
        self.normalizer = normalizer
        self.alpha = alpha
        self.iters = iters
        self.weight = weight
        self.G = G
        self.signature = signature

    def forward(self, inputs, targets_adversarial, opt):

        inputs_original = inputs.clone().detach()
        criterion = nn.CrossEntropyLoss()
        inputs_adversarial = inputs.clone().detach()
        inputs_adversarial.requires_grad = True

        optimize = optim.Adam([inputs_adversarial], lr=0.01)
        cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

        for k in range(self.iters):
            inputs_adversarial = inputs_adversarial.clone().detach()
            inputs_adversarial.requires_grad = True
            preds, last_feature = self.model(inputs_adversarial)

            adversarial_mitigated = self.G(inputs_adversarial)

            preds_miti, last_feature_miti = self.model(adversarial_mitigated)

            loss1 = criterion(preds, targets_adversarial)
            loss2 = torch.mean(cosine(last_feature, last_feature_miti))
            total_loss = loss1 - self.alpha * loss2
            optimize.zero_grad()
            self.model.zero_grad()
            total_loss.backward()
            grad = inputs_adversarial.grad.data
            grad = torch.sign(grad)
            inputs_adversarial = inputs_adversarial - self.weight * grad
            inputs_adversarial = clamp_input(inputs_adversarial, self.normalizer)
            inputs_adversarial = torch.max(
                inputs_original - self.epsilon, torch.min(inputs_adversarial, inputs_original + self.epsilon)
            )
            if torch.sum(torch.argmax(preds, dim=1) == targets_adversarial) == preds.shape[0]:
                break
        return inputs_adversarial


class SignatureAttack(object):
    #def __init__(self, model, normalizer, signature, epsilon=0.5, alpha=3, iters=400, weight=0.1): # MNIST
    # def __init__(self, model, normalizer, signature, epsilon=0.05, alpha=3, iters=300, weight=0.01): # CIFAR10
    # def __init__(self, model, normalizer, signature, epsilon=0.03, alpha=3, iters=100, weight=0.005):  # GTSRB
    def __init__(self, model, normalizer, signature, epsilon=0.03, alpha=3, iters=100, weight=0.01):  # Tinyimagenet
        self.model = model
        self.epsilon = epsilon
        self.normalizer = normalizer
        self.signature = signature
        self.alpha = alpha
        self.iters = iters
        self.weight = weight

    def forward(self, inputs, targets_adversarial):

        inputs_original = inputs.clone().detach()
        criterion = nn.CrossEntropyLoss()
        inputs_adversarial = inputs.clone().detach()
        inputs_adversarial.requires_grad = True
        signature_batch = self.signature.repeat(inputs_adversarial.shape[0], 1)
        optimize = optim.Adam([inputs_adversarial], lr=0.01)
        cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
       
        for k in range(self.iters):
            inputs_adversarial = inputs_adversarial.clone().detach()
            inputs_adversarial.requires_grad = True
            preds, last_feature = self.model(inputs_adversarial)
            loss1 = criterion(preds, targets_adversarial)
            loss2 = torch.mean(cosine(last_feature, signature_batch))
            total_loss = loss1 + self.alpha * loss2
            optimize.zero_grad()
            self.model.zero_grad()
            total_loss.backward()
            grad = inputs_adversarial.grad.data
            grad = torch.sign(grad)
            inputs_adversarial = inputs_adversarial - self.weight * grad
            inputs_adversarial = clamp_input(inputs_adversarial, self.normalizer)
            inputs_adversarial = torch.max(
                inputs_original - self.epsilon, torch.min(inputs_adversarial, inputs_original + self.epsilon)
            )
            if torch.sum(torch.argmax(preds, dim=1) == targets_adversarial) == preds.shape[0]:
                break
        return inputs_adversarial


class MIFGSM:
    def __init__(self, model, eps=0.1, steps=8, decay=1.0, alpha=0.02):  # MNIST
    # def __init__(self, model, eps=0.01, steps=8, decay=1.0, alpha=0.01): # CIFAR10
    # def __init__(self, model, eps=0.015, steps=8, decay=1.0, alpha=0.005): # GTSRB
    # def __init__(self, model, eps=0.005, steps=8, decay=1.0, alpha=0.005): # TinyImageNet
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.model = model

    def forward(self, images, labels, opt):
        images = images.clone().detach().to(opt.device)
        labels = labels.clone().detach().to(opt.device)
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(opt.device)
        adv_images = images.clone().detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs, _ = self.model(adv_images)

            cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1] + [1] * (len(grad.shape) - 1))
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

        return adv_images


class CW:
    def __init__(self, model, c=1e-3, kappa=0, steps=300, lr=0.01):    # MNIST
    # def __init__(self, model, c=1e-3, kappa=0, steps=100, lr=0.05):    # TinyImage
    # def __init__(self, model, c=1e-3, kappa=0, steps=300, lr=0.05):    # CIFAR10
    # def __init__(self, model, c=1e-1, kappa=0, steps=500, lr=0.01):  # GTSRB

        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.model = model

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        binary_step = 9
        batch_size = images.shape[0]
        array_c = torch.ones(batch_size, device="cuda", dtype=torch.float32) * self.c
        upper_bound = torch.ones(batch_size, device="cuda", dtype=torch.float32) * 1e10
        lower_bound = torch.ones(batch_size, device="cuda", dtype=torch.float32)

        for step_binary in range(binary_step):

            images = images.clone().detach().to("cuda")
            labels = labels.clone().detach().to("cuda")

            w = self.inverse_tanh_space(images * 0.99999).detach().float().cuda()  # 0.99999
            w_pert = torch.zeros_like(w).cuda().float()
            w_pert.requires_grad = True

            best_adv_images = images.clone().detach()

            best_L2 = 1e10 * torch.ones((len(images))).to("cuda")

            dim = len(images.shape)

            MSELoss = nn.MSELoss(reduction="none")
            Flatten = nn.Flatten()

            optimizer = optim.Adam([w_pert], lr=self.lr)
            mask_save = torch.zeros(batch_size, device="cuda", dtype=torch.float32)
            for step in range(self.steps):
                # Get Adversarial Images
                adv_images = self.tanh_space(w + w_pert)
                current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
                L2_loss = current_L2.mean()
                self.model.eval()
                outputs, _ = self.model(adv_images)
                f_Loss = self.f(outputs, labels)
                cost = L2_loss + (array_c * f_Loss).mean()
                optimizer.zero_grad()
                cost.backward(retain_graph=True)
                optimizer.step()

                # Update Adversarial Images
                _, pre = torch.max(outputs.detach(), 1)
                correct = (pre == labels).float()

                mask = correct * (best_L2 > current_L2.detach())
                best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
                mask_save = torch.logical_or(mask_save, mask)

                mask = mask.view([-1] + [1] * (dim - 1))
                best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images
            index_success = mask_save.int()

            # Case 1: attack succesfully
            upper_bound_temp = torch.min(upper_bound, array_c)
            temp_mask = (upper_bound_temp < 1e9).int()
            array_c_case1 = (lower_bound + upper_bound_temp) / 2
            array_c_case1 = temp_mask * array_c_case1 + (1 - temp_mask) * array_c

            # Case 2: attack failed
            lower_bound_temp = torch.max(lower_bound, array_c)
            array_c_case2_1 = (lower_bound_temp + upper_bound) / 2
            array_c_case2_2 = array_c * 10
            temp_mask = (upper_bound < 1e9).int()
            array_c_case2 = temp_mask * array_c_case2_1 + (1 - temp_mask) * array_c_case2_2

            # Update global array_c, upper_bound, lower_bound
            array_c = index_success * array_c_case1 + (1 - index_success) * array_c_case2
            upper_bound = index_success * upper_bound_temp + (1 - index_success) * upper_bound
            lower_bound = (1 - index_success) * lower_bound_temp + index_success * lower_bound
        return best_adv_images

    def tanh_space(self, x):
        return torch.tanh(x)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to("cuda")
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp((i - j), min=-self.kappa)


class ElasticNet:
    def __init__(self, model, c=1e-1, kappa=0, steps=100, lr=0.05, alpha=1e-3):  # MNIST
    # def __init__(self, model, c=1e-1, kappa=0, steps=100, lr=0.05, alpha=1e-3):    # CIFAR10
    # def __init__(self, model, c=1e-1, kappa=0, steps=100, lr=0.05, alpha=1e-3):    # GTSRB
    # def __init__(self, model, c=1e-1, kappa=0, steps=100, lr=0.05, alpha=1e-3):  # TinyImageNet
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.model = model
        self.alpha = alpha

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        binary_step = 9
        batch_size = images.shape[0]
        array_c = torch.ones(batch_size, device="cuda", dtype=torch.float32) * self.c
        upper_bound = torch.ones(batch_size, device="cuda", dtype=torch.float32) * 1e10
        lower_bound = torch.ones(batch_size, device="cuda", dtype=torch.float32)

        for step_binary in range(binary_step):

            images = images.clone().detach().to("cuda")
            labels = labels.clone().detach().to("cuda")
            w = self.inverse_tanh_space(images * 0.99999).detach().float().cuda()
            w_pert = torch.zeros_like(w).cuda().float()
            w_pert.requires_grad = True

            best_adv_images = images.clone().detach()

            best_L2 = 1e10 * torch.ones((len(images))).to("cuda")
            dim = len(images.shape)

            MSELoss = nn.MSELoss(reduction="none")
            L1_loss = nn.L1Loss(reduction="none")
            Flatten = nn.Flatten()

            optimizer = optim.Adam([w_pert], lr=self.lr)
            mask_save = torch.zeros(batch_size, device="cuda", dtype=torch.float32)
            for step in range(self.steps):
                # Get Adversarial Images
                adv_images = self.tanh_space(w + w_pert)

                current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
                current_L1 = L1_loss(Flatten(adv_images), Flatten(images)).sum(dim=1)
                mean_l1_loss = current_L1.mean()
                L2_loss = current_L2.mean()
                self.model.eval()
                outputs, _ = self.model(adv_images)
                f_Loss = self.f(outputs, labels)
                cost = L2_loss + (array_c * f_Loss).mean() + self.alpha * mean_l1_loss
                optimizer.zero_grad()
                cost.backward(retain_graph=True)
                optimizer.step()
                # Update Adversarial Images
                _, pre = torch.max(outputs.detach(), 1)
                correct = (pre == labels).float()

                mask = correct * (best_L2 > current_L2.detach())
                best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
                mask_save = torch.logical_or(mask_save, mask)

                mask = mask.view([-1] + [1] * (dim - 1))
                best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            index_success = mask_save.int()

            # Case 1: attack succesfully
            upper_bound_temp = torch.min(upper_bound, array_c)
            temp_mask = (upper_bound_temp < 1e9).int()
            array_c_case1 = (lower_bound + upper_bound_temp) / 2
            array_c_case1 = temp_mask * array_c_case1 + (1 - temp_mask) * array_c

            # Case 2: attack failed
            lower_bound_temp = torch.max(lower_bound, array_c)
            array_c_case2_1 = (lower_bound_temp + upper_bound) / 2
            array_c_case2_2 = array_c * 10
            temp_mask = (upper_bound < 1e9).int()
            array_c_case2 = temp_mask * array_c_case2_1 + (1 - temp_mask) * array_c_case2_2

            # Update global array_c, upper_bound, lower_bound
            array_c = index_success * array_c_case1 + (1 - index_success) * array_c_case2
            upper_bound = index_success * upper_bound_temp + (1 - index_success) * upper_bound
            lower_bound = (1 - index_success) * lower_bound_temp + index_success * lower_bound
        return best_adv_images

    def tanh_space(self, x):
        return torch.tanh(x)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to("cuda")
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp((i - j), min=-self.kappa)
