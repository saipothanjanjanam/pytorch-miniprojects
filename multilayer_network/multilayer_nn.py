import torch


def sigmoid(x):
	return 1/(1 + torch.exp(-x))

# Setting the seed for the generation of random numbers
torch.manual_seed(30)

# 3 layers - units in input = 4, units in hidden = 2 and units in output = 1
input_layer = 4
hidden_layer = 2
output_layer = 1

W1 = torch.randn(input_layer, hidden_layer) #Size = 4x2
W2 = torch.randn(hidden_layer, output_layer) #Size = 2x1

b1 = torch.randn(hidden_layer,1) #Size = 2x1
b2 = torch.randn(output_layer,1) #Size = 1x1

input_features = torch.randn(input_layer,1) #Size = 4x1

h =  sigmoid(torch.mm(input_features.view(-1,4), W1) + b1.view(-1,2)) #Size = 1x2
prediction = sigmoid(torch.mm(h,W2) + b2) #Size = 1x1

print(prediction)