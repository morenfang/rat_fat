from process_data import *
import torch
import torch.nn.functional as F
import numpy as np
import composition_stats as coms

train_data, labels = get_data('./data/fat_data.xlsx')
train_data = torch.from_numpy(train_data)
labels = torch.from_numpy(labels)

# log10 to train_data
train_data = log_plus_1(train_data)
# CLR to train_data
# train_data = coms.clr(train_data)
print(train_data)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(174, 512)
        self.layer2 = torch.nn.Linear(512, 256)
        self.layer2_1 = torch.nn.Linear(256, 128)
        self.layer3 = torch.nn.Linear(128, 32)
        self.layer4 = torch.nn.Linear(32, 2)

    def forward(self, din):
        dout = F.tanh(self.layer1(din))
        dout = F.tanh(self.layer2(dout))
        dout = F.tanh(self.layer2_1(dout))
        dout = F.tanh(self.layer3(dout))
        dout = F.softmax(self.layer4(dout), dim=0)
        return dout


def train():
    eval_id = 0
    pos = 0

    # 10% validation
    val = [3, 7, 15, 19, 33, 39]
    test_data = []
    for i in val:
        test_data.append(train_data[i])
    model = MLP()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    for epoch in range(500):  # epoch per validation
        train_loss = 0.0
        for i in range(train_data.shape[0]):
            if i not in val:
                optimizer.zero_grad()
                out = model(train_data[i])
                loss = loss_func(out, labels[i])
                loss.backward()
                optimizer.step()
                train_loss += loss
        if epoch % 1 == 0:
            print("Epoch: {} \tTraining Loss: {:.10f}".format(epoch, train_loss))
    # eval
    for i in val:
        out = torch.argmax(model(train_data[i]))
        if torch.argmax(labels[i]) == out:
            pos = pos + 1
        print("data {} evaluate: {} vs. {}".format(i, out, torch.argmax(labels[i])))
    print("Eval Acc = {:.4f}".format(pos / len(val)))

    # N-1 cross validation
    '''
    for i in range(len(train_data)):
        model = MLP()
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
        for epoch in range(1):  # 500 epoch per validation
            train_loss = 0.0
            for j in range(train_data.shape[0]):
                optimizer.zero_grad()
                if i == j:
                    eval_id = i
                else:
                    out = model(train_data[j])
                    loss = loss_func(out, labels[i])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss
            if epoch % 1 == 0:
                print("Epoch: {} \tTraining Loss: {:.10f}".format(epoch, train_loss))
        # eval by data i
        out = torch.argmax(model(train_data[i]))
        if torch.argmax(labels[i]) == out:
            pos = pos + 1
        print("data {} evaluate: {} vs. {}".format(i, out, torch.argmax(labels[i])))
    print("Eval Acc = {:.4f}".format(pos / len(train_data)))
    '''
    '''
    for epoch in range(2000):
        train_loss = 0.0
        for i in range(train_data.shape[0]):
            optimizer.zero_grad()
            out = model(train_data[i])
            loss = loss_func(out, labels[i])
            loss.backward()
            optimizer.step()
            train_loss += loss
            # print("training loss = ", loss)
        if epoch % 10 == 0:
            print("Epoch: {} \tTraining Loss: {:.10f}".format(epoch, train_loss))
    '''


if __name__ == '__main__':
    train()
