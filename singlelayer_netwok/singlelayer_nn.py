import torch

def sigmoid(x):
	return 1/(1 + torch.exp(-x))

# Setting the seed for random numbers generation
torch.manual_seed(15)

# Creating the tensor of random feature tensor of size 1x6 
features = torch.randn(1,6)

# Creating the tensor of random weight tensor of size 1x6 
weights = torch.randn_like(features)

bias = torch.rand(1)

prediction = sigmoid(torch.mm(features, weights.view(6,-1)) + bias)

print(prediction)