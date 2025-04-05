import torch



def test(model, test_loader, A_norm, len_test_dataset):
    device = torch.device("cpu")
    model.eval()
    correct = 0
    test_accuracies = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = model(batch_x, A_norm)
            pred = out.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
    accuracy = correct / len_test_dataset
    test_accuracies.append(accuracy)
    return test_accuracies