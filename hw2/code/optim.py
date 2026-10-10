import torch



class SGD:
    def __init__(self, model, name):
        self.weight_decay = 0.0005 if name == "lenet" else 0.0001
        self.params = []
        for key, param in model.named_parameters():
            conv_bias = key.endswith("bias") and "conv" in key
            doubled = key.endswith("bias") if name == "lenet" else conv_bias or key == "fc.bias"
            decay = 0 if name != "lenet" and conv_bias else 1
            self.params.append((param, 2 if doubled else 1, decay))
        self.history = [torch.zeros_like(p) for p, _, _ in self.params]

    @torch.no_grad()
    def step(self, lr, momentum):
        for (param, mult, decay), history in zip(self.params, self.history):
            gradient = param.grad.add(param, alpha=self.weight_decay * decay)
            history.mul_(momentum).add_(gradient, alpha=lr * mult)
            param.sub_(history)

    def state_dict(self):
        return [history.cpu() for history in self.history]

    def load_state_dict(self, state):
        for history, saved in zip(self.history, state):
            history.copy_(saved)


def schedule(mode, step):
    if mode == "pc":
        return 0.35 * 0.1 ** ((step >= 50000) + (step >= 70000)), 0.9
    if mode == "clr":
        return 0.1 + 2.9 * (1 - abs(step / 10000 - 1)), 0.9
    if mode == "inv":
        return 0.01 * (1 + 0.0001 * step) ** -0.75, 0.9
    if step <= 586:
        fraction = step / 586
        return 0.01 + 0.09 * fraction, 0.95 - 0.15 * fraction
    if step <= 1172:
        fraction = (step - 586) / 586
        return 0.1 - 0.09 * fraction, 0.8 + 0.15 * fraction
    fraction = (step - 1172) / 233
    return 0.01 + (0.00001 - 0.01) * fraction, 0.95
