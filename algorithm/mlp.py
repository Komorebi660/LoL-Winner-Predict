import random
import numpy as np
import os
import argparse
from tqdm import trange
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
import pandas as pd
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)                           # python random seed
    np.random.seed(seed)                        # numpy random seed
    os.environ['PYTHONHASHSEED'] = str(seed)    # python hash seed
    torch.manual_seed(seed)                     # pytorch random seed
    torch.cuda.manual_seed(seed)                # cuda random seed


class DenseBlock(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dims*3+1, input_dims*2),
            nn.Linear(input_dims*2, input_dims),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )
        self.norm = nn.BatchNorm1d(input_dims) 
    def forward(self, inputs, dense):
        x = self.norm(self.mlp(inputs))
        out = inputs + torch.cat([x, dense], dim=1)
        return out


class DenseModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()

        #self.norm = nn.BatchNorm1d(input_dims)  

        self.input_up_proj = nn.Sequential(
            nn.Linear(input_dims, int(input_dims*1.5)),
        )

        self.dense_layers = nn.ModuleList([DenseBlock(input_dims//2) for _ in range(6)])

        self.output_down_proj = nn.Sequential(
            nn.Linear(int(input_dims*1.5), input_dims),
            nn.Linear(input_dims, 1),
            #nn.Tanh(),
        )
        
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, labels=None):
        #inputs = self.norm(inputs)                      # [batch_size, 39]
        x = self.input_up_proj(inputs)                  # [batch_size, 58]
        for dense in self.dense_layers:
            x = dense(x, inputs)                        # [batch_size, 58]
        prediction_scores = self.output_down_proj(x)    # [batch_size, 1]
        logits = self.sigmoid(prediction_scores)        # [batch_size, 1]

        if labels is None:
            return logits
        else:
            loss = self.BCEloss(prediction_scores, labels.unsqueeze(1).float())
            return logits, loss



class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.features = X
        self.labels = Y
        print(self.features.shape)
        print(self.labels.shape)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        with torch.no_grad():
            data_item = {"inputs": self.features[index], "label": self.labels[index]}
        return data_item


def create_data_loader(X_train, Y_train, X_test, Y_test, batch_size):
    train_dataset = MyDataset(X_train, Y_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = MyDataset(X_test, Y_test)
    test_data_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

    return train_data_loader, test_data_loader



def train(args, device, X_train, Y_train, X_test, Y_test):
    train_dataloader, dev_dataloader = create_data_loader(X_train, Y_train, X_test, Y_test, args.bsz)
    # get full dev dataset
    dev_dataset = []
    for dev_data in dev_dataloader:
        dev_inputs = dev_data['inputs']
        dev_labels = dev_data['label']
        dev_dataset.append((dev_inputs, dev_labels))
    input_dims = dev_dataset[0][0].shape[1]
    print(f"example data: {dev_dataset[0][0]}")
    print(f"input_dims: {input_dims}")

    # get model
    model = DenseModel(input_dims).to(device)
    model.train()
    # get optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # learning rate coefficient for each step
    def lr_lambda(step):
        fraction = step / (args.epochs * len(train_dataloader))
        return 1 - fraction
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # train & validate on-the-fly
    training_losses = []
    training_accuracy = []
    dev_losses = []
    dev_accuracy = []
    for epoch in trange(args.epochs):
        #print(f"Epoch {epoch+1} ...")
        # training
        all_loss = []
        all_acc = []
        for data in train_dataloader:
            inputs = data['inputs'].to(device)
            labels = data['label'].to(device)
            optimizer.zero_grad()
            prediction_scores, loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            all_loss.append(loss.detach().cpu().numpy())
            prediction = (prediction_scores > 0.5).int().cpu().numpy()
            all_acc.append(accuracy_score(labels.cpu().numpy(), prediction))
            scheduler.step()
        _loss = sum(all_loss)/len(all_loss)
        training_losses.append(_loss)
        _acc = sum(all_acc)/len(all_acc)
        training_accuracy.append(_acc)
        #print(f"training Loss: {_loss}")
        #print(f"learning rate: {optimizer.param_groups[0]['lr']}")
        #print(f"training accuracy: {_acc}")
        
        # evaluation
        model.eval()
        with torch.no_grad():
            all_dec_acc = []
            all_dev_loss = []
            for dev_inputs, dev_labels in dev_dataset:
                dev_inputs = dev_inputs.to(device)
                dev_labels = dev_labels.to(device)
                dev_prediction_scores, dev_loss = model(dev_inputs, dev_labels)
                dev_prediction = (dev_prediction_scores > 0.5).int().cpu().numpy()
                all_dec_acc.append(accuracy_score(dev_labels.cpu().numpy(), dev_prediction))
                all_dev_loss.append(dev_loss.cpu().numpy())
            dev_loss = sum(all_dev_loss)/len(all_dev_loss)
            dev_losses.append(dev_loss)
            dev_acc = sum(all_dec_acc)/len(all_dec_acc)
            dev_accuracy.append(dev_acc)
            #print(f"dev Loss: {dev_loss}")
            #print(f"dev accuracy: {dev_acc}")
        model.train()
    
    return training_losses, dev_losses, training_accuracy, dev_accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=8192, help="batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="current process rank")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")    
    args = parser.parse_args()

    set_seed(2023)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # load data
    with open('../data/train_processed.csv', 'r') as f:
        dataset = pd.read_csv(f)
        
    features = dataset.drop(['win'],axis=1).values.astype(np.float32)
    labels = dataset['win'].values.astype(np.float32)
    print(features.shape)
    print(labels.shape)

    # cross validation
    scores = []
    t_loss = np.zeros(args.epochs)
    d_loss = np.zeros(args.epochs)
    t_acc = np.zeros(args.epochs)
    d_acc = np.zeros(args.epochs)
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    for train_index, test_index in ss.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]
        training_losses, dev_losses, training_accuracy, dev_accuracy = train(args, device, X_train, Y_train, X_test, Y_test)
        scores.append(dev_accuracy[-1])
        t_loss += np.array(training_losses, dtype=np.float32)
        d_loss += np.array(dev_losses, dtype=np.float32)
        t_acc += np.array(training_accuracy, dtype=np.float32)
        d_acc += np.array(dev_accuracy, dtype=np.float32)

    print(scores, np.mean(np.array(scores, dtype=np.float32)))
    
    # plot loss curve
    plt.ylim(0.30, 0.35)
    plt.plot(t_loss/5, label='training loss', linestyle='--')
    plt.plot(d_loss/5, label='dev loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('../figs/loss.png')
    plt.close()
    # plot accuracy curve
    plt.ylim(0.84, 0.87)
    plt.plot(t_acc/5, label='training accuracy', linestyle='--')
    plt.plot(d_acc/5, label='dev accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('../figs/accuracy.png')
    plt.close()
