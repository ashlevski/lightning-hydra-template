import torch
def divide_matrix(matrix):
    # Convert the matrix to a PyTorch tensor

    random_matrix = torch.randint(2, size=matrix.shape, device=matrix.device)


    # Flatten the tensor to work with 1D tensors
    mask1 = matrix * random_matrix
    mask2 = matrix * (1 - random_matrix)

    return mask1, mask2


def divide_matrix_ratio(matrix, threshold=0.5):
    # Convert the matrix to a PyTorch tensor

    random_matrix = torch.rand(matrix.shape)
    random_matrix = torch.where(random_matrix < threshold, torch.tensor(0), torch.tensor(1)).to(matrix.device)
    # Flatten the tensor to work with 1D tensors
    mask1 = matrix * random_matrix # ratio set on this mask
    mask2 = matrix * (1 - random_matrix)

    return mask1, mask2
