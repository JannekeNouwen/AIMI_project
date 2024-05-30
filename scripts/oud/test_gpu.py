import time
import torch


x = torch.Tensor([1, 2, 3]).to("cuda")
print(x.shape)
x.to("cpu")

with open("/home/ljulius/log/weird_tensor.txt", "w") as file:
	file.write(str(x))

print("helloooooo")

for i in range(30):
	time.sleep(5)
	print(i)
