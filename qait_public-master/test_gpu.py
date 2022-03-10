import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
t = torch.tensor(3)
t = t.to(device)
print(t * t)
