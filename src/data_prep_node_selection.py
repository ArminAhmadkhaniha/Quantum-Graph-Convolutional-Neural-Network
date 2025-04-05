import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA


def apply_pca(features, n_components=256):
    pca = PCA(n_components=n_components)
    return torch.tensor(pca.fit_transform(features.numpy()), dtype=torch.float)




def cora_data_node_selection(batch_size=32, num_nodes=1024):
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]


    num_nodes = 1024
    selected_nodes = torch.randperm(data.x.shape[0])[:num_nodes]
    x = apply_pca(data.x[selected_nodes])
    y = data.y[selected_nodes]


    class_0, class_1 = 2, 4
    mask = (y == class_0) | (y == class_1)
    x = x[mask]
    y_binary = (y[mask] == class_1).long()

    train_ratio = 0.8
    train_size = int(train_ratio * x.shape[0])
    test_size = x.shape[0] - train_size
    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y_binary[:train_size], y_binary[train_size:]

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    A = to_dense_adj(data.edge_index).squeeze(0)
    A = A[selected_nodes][:, selected_nodes]
    A = A[mask][:, mask]
    A = A + torch.eye(A.shape[0])
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(A.sum(dim=1)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return train_loader, test_loader, A_norm, len(test_dataset)