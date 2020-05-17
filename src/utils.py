import torch.nn

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def log_parameter_metrics(name, model):
    for param in model.parameters():
        print(name, param.min(), param.max(), param.mean())
