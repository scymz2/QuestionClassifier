# reference: https://www.cs.toronto.edu/~lczhang/360/lec/w03/nn.html
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


def compute_acc(model, data_loader):
    correct = 0
    total = 0
    for features, labels in data_loader:
        outputs = model(features)
        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += features.shape[0]
        return correct / total


def train(model, train_data, val_data, batch_size=64, num_epoch=100, mode='train'):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    epoch = 0
    num_iters, losses, train_accs, val_accs = [], [], [], []
    for epoch in range(num_epoch):
        for train_features, train_labels in iter(train_loader):
            # train model
            outputs = model(train_features)                         # forward pass
            loss = criterion(outputs, train_labels)                 # compute loss
            loss.backward()                                         # backward pass
            optimizer.step()                                        # update weights
            optimizer.zero_grad()                                   # clean gradients

            # record training information
            num_iters.append(epoch)                                 # counts of iterations
            losses.append(float(loss/batch_size))                   # average loss of the batch
            train_acc = compute_acc(model, train_loader)
            train_accs.append(train_acc)                            # training accuracy
            val_acc = compute_acc(model, val_loader)
            val_accs.append(val_acc)                                # validation accuracy

            epoch += 1

            # print information
            if mode == 'develop':
                print('Epoch: ' + str(epoch) +
                      ' Train Set Accuracy: ' + str(train_acc) +
                      ' Validation Set Accuracy: ' + str(val_acc)
                      )


def test():
    pass


