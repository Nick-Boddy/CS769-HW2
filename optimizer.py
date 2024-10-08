from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                betas = group["betas"]
                epsilon = group["eps"]
                weight_decay = group["weight_decay"]
                beta1 = betas[0]
                beta2 = betas[1]

                # initialize for t = 0
                if "t" not in state:
                    state["t"] = 0                        # time step
                    state["m"] = torch.zeros_like(p.data) # 1st moment
                    state["v"] = torch.zeros_like(p.data) # 2nd moment

                state["t"] += 1
                t = state["t"]
                m_t = state["m"]
                v_t = state["v"]

                # Update first and second moments of the gradients
                m_t.mul_(beta1).add_(grad.mul(1 - beta1))
                v_t.mul_(beta2).add_(grad.mul(grad).mul(1 - beta2))

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if group["correct_bias"]:
                    alpha_t = alpha * (1 - beta2**t)**0.5 / (1 - beta1**t)
                else:
                    alpha_t = alpha

                # Update parameters
                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if weight_decay != 0:
                    p.data.add_(p.data.mul(-alpha * weight_decay))
                p.data.addcdiv_(m_t, v_t.sqrt().add_(epsilon), value=-alpha_t)
        return loss
