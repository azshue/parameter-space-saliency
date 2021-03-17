import torch


def sort_grads(model, method):
    grads = []
    # print(model.named_parameters)
    for param in model.named_parameters():
        name = param[0]
        content = param[1]
        grad = content.grad.detach()
        # print(name)
        if method == 'column-wise':
            if "attention" in name:
                if "Norm" not in name:
                    # print(name)
                    if len(grad.size()) == 1:
                        grads.append(grad.view(-1).cpu())
                    elif len(grad.size()) == 2:
                        grads.append(grad.mean(-1).view(-1).cpu())
        elif method == 'naive':
            grads.append(grad.view(-1).cpu())
    grads = torch.abs(torch.cat(grads))
    return grads
