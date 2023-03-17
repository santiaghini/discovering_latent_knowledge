import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

############# CCS #############
class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class CCS(object):
    def __init__(self, x0, x1, x2, x3, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        
        # TODO: play with lr and weight_decay

        # data
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.x2 = self.normalize(x2)
        self.x3 = self.normalize(x3)
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
        self.probe = self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

        
    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)    


    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)

        return normalized_x

        
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        x2 = torch.tensor(self.x2, dtype=torch.float, requires_grad=False, device=self.device)
        x3 = torch.tensor(self.x3, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1, x2, x3
    

    def get_loss(self, p0, p1, p2, p3):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        # TODO: verify loss functions:
        # the lower the more confident
        # TODO OH: Entropy of outputs, minimize entropy
        # informative_loss = ((1 - torch.max(p0, torch.max(p1, torch.max(p2, p3))))**2).mean(0)
        # informative_loss = ((1 - torch.max(p0, torch.max(p1, torch.max(p2, p3))) + torch.min(p0, torch.min(p1, torch.min(p2, p3))))**2).mean(0)
        # informative_loss = ((1 - torch.max(p0, torch.max(p1, torch.max(p2, p3))))**2).mean(0)
        # informative_loss = ((1 - torch.max(p0, torch.max(p1, torch.max(p2, p3))))**2).mean(0)
        informative_loss = ((-(p0*torch.log(p0) + p1*torch.log(p1) + p2*torch.log(p2) + p3*torch.log(p3)))**2).mean(0)
        print("informative_loss: {informative_loss}")
        consistent_loss = (((p0 + p1 + p2 + p3) - 1)**2).mean(0)
        # TODO: play with weighting if it doesnt work. Try a grid
        # downweighting consistency loss, not too much, or upweighting
        return informative_loss + consistent_loss


    def get_acc(self, x0_test, x1_test, x2_test, x3_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        x2 = torch.tensor(self.normalize(x2_test), dtype=torch.float, requires_grad=False, device=self.device)
        x3 = torch.tensor(self.normalize(x3_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1, p2, p3 = self.best_probe(x0), self.best_probe(x1), self.best_probe(x2), self.best_probe(x3)
        # TODO: check what confidence we want here
        # avg_confidence = 0.5*(p0 + (1-p1)) # original
        # avg_confidence = 0.25*(p0 + p1 + (1 - p2 - p3))
        # avg_confidence = 0.5*(p0 + (1 - p1 - p2 - p3))
        # predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        # p0c = 0.25*(p0 + (1 - p0 - p2 - p3) + (1 - p0 - p1 - p3) + (1 - p0 - p1 - p2))
        # p1c = 0.25*(p1 + (1 - p1 - p2 - p3) + (1 - p0 - p1 - p3) + (1 - p0 - p1 - p2))
        # p2c = 0.25*(p2 + (1 - p1 - p2 - p3) + (1 - p0 - p2 - p3) + (1 - p0 - p1 - p2))
        # p3c = 0.25*(p3 + (1 - p1 - p2 - p3) + (1 - p0 - p2 - p3) + (1 - p0 - p1 - p3))
        # p0f, p1f, p2f, p3f = [pi.detach().cpu().numpy()[:, 0] for pi in [p0c, p1c, p2c, p3c]]

        p0f, p1f, p2f, p3f = [pi.detach().cpu().numpy()[:, 0] for pi in [p0, p1, p2, p3]]

        predictions = np.zeros(p0.shape[0], dtype=int)
        predictions[np.logical_and.reduce((p0f > p1f, p0f > p2f, p0f > p3f))] = 0
        predictions[np.logical_and.reduce((p1f > p0f, p1f > p2f, p1f > p3f))] = 1
        predictions[np.logical_and.reduce((p2f > p0f, p2f > p1f, p2f > p3f))] = 2
        predictions[np.logical_and.reduce((p3f > p0f, p3f > p1f, p3f > p2f))] = 3

        acc = (predictions == y_test).mean()
        # acc = max(acc, 1 - acc)

        return acc
    
        
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1, x2, x3 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1, x2, x3 = x0[permutation], x1[permutation], x2[permutation], x3[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
                x2_batch = x2[j*batch_size:(j+1)*batch_size]
                x3_batch = x3[j*batch_size:(j+1)*batch_size]
            
                # probe
                p0, p1, p2, p3 = self.probe(x0_batch), self.probe(x1_batch), self.probe(x2_batch), self.probe(x3_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1, p2, p3)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()
    
    def repeated_train(self):
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss
