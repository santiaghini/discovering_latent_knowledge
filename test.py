import torch

# define the list of tensors
tensor_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([0, 8, 9]), torch.tensor([7, 0, 5])]

# stack the tensors into a new tensor
stacked_tensor = torch.stack(tensor_list)

# use the argmax function to find the index of the tensor with the largest value at each position
max_indices = stacked_tensor.argmax(dim=0)

# print the resulting tensor
print(max_indices)