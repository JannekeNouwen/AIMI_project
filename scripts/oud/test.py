print("TEST")
import torch
import random
print(random.randint(0, 1000))

x= torch.Tensor([1,2,3]).to("cuda")
print(x.shape)
