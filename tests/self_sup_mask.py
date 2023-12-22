import torch

def divide_matrix(matrix):
    # Convert the matrix to a PyTorch tensor

    random_matrix = torch.randint(2, size=matrix.shape, device=matrix.device)
    # Flatten the tensor to work with 1D tensors
    mask1 = matrix * random_matrix
    mask2 = matrix * (1 - random_matrix)

    return mask1, mask2

# Example usage
# boolean_matrix = torch.tensor([[1, 0, 1],
#                                [0, 1, 0],
#                                [1, 1, 0]])
#
# mask1, mask2 = divide_matrix(boolean_matrix)
#
# print("Original Matrix:")
# print(boolean_matrix)
#
# print("\nMask 1:")
# print(mask1)
#
# print("\nMask 2:")
# print(mask2)
