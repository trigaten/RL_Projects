import torch
x = torch.FloatTensor([[3, 4], [5, 6], [7, 8]])

y = [0, 1, 0]

z = x[range(len(x)), y]

print(z)