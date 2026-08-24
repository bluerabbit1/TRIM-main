"""
Trainer for Node Classification (Temporal-Motif) with Trim
============================================================
"""

import time
import torch
import numpy as np
from tqdm import tqdm
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class NC_Motif_Trainer(object):
    def __init__(self, args, model, data, writer=None, **kwargs):
        self.args = args
        self.data = data
        self.model = model
        self.writer = writer
        self.len = len(data['edge_index_list'])
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        self.node_mask = [torch.arange(data['x_list'][i].shape[0]) for i in range(self.len)]
        x = data['x_list'][-1].float().to(args.device)
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x
        setup_seed(args.seed)
        print('total length: {}, test length: {}'.format(
            self.len, args.testlength))

    def run(self):
        args = self.args
        max_acc = 0
        max_test_acc = 0
        max_train_acc = 0
        min_epoch = args.min_epoch
        max_patience = args.patience

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        t_total0 = time.time()
        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                average_epoch_loss, average_train_acc, average_val_acc, average_test_acc, test_acc_list = \
                    self.train(epoch, self.data)

                if average_val_acc > max_acc:
                    max_acc = average_val_acc
                    max_test_acc = average_test_acc
                    max_train_acc = average_train_acc
                    best_test_acc_list = test_acc_list
                    patience = 0
                    best_epoch = epoch
                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break

                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print("Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(
                        epoch, average_epoch_loss, time.time() - t0))
                    print(f"Current: Epoch:{epoch}, Train ACC:{average_train_acc:.4f}, "
                          f"Val ACC: {average_val_acc:.4f}, Test ACC: {average_test_acc:.4f}")
                    print(f"Best: Epoch:{best_epoch}, Train ACC:{max_train_acc:.4f}, "
                          f"Val ACC: {max_acc:.4f}, Test ACC: {max_test_acc:.4f}")
                    print(f"Every Test: Test28:{best_test_acc_list[0]:.4f}, "
                          f"Test29: {best_test_acc_list[1]:.4f}, Test30: {best_test_acc_list[2]:.4f}")

        epoch_time = (time.time() - t_total0) / (epoch - 1)
        return (epoch, average_train_acc, average_val_acc, average_test_acc,
                best_epoch, max_train_acc, max_acc, max_test_acc,
                epoch_time, best_test_acc_list)

    def train(self, epoch, data):
        self.model.train()
        optimizer = self.optimizer

        embedding_list = self.model(
            data['edge_index_list'], self.x[:self.len_train], self.len_train)

        criterion = torch.nn.CrossEntropyLoss()
        per_step_loss = []
        for t in range(self.len_train):
            z = embedding_list[:, t, :].squeeze()
            pred = self.cal_pred(z, self.model.cs_decoder,
                                 self.node_mask[t].to(self.args.device))
            step_loss = criterion(
                pred,
                data['label_list'][t].squeeze().to(self.args.device))
            per_step_loss.append(step_loss)

        per_step_loss = torch.stack(per_step_loss)

        # === Trim: mean + λ * variance ===
        task_loss = per_step_loss.mean()
        tirm_penalty = per_step_loss.var()

        la = self.args.weight1
        loss = task_loss + la * tirm_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Eval
        self.model.eval()
        train_acc_list, val_acc_list, test_acc_list = [], [], []
        embedding_list = self.model(data['edge_index_list'], self.x, self.len)
        for t in range(self.len):
            z = embedding_list[:, t, :].squeeze()
            acc = self.predict(z, self.model.cs_decoder,
                               self.node_mask[t].to(self.args.device),
                               data['label_list'][t])
            if t < self.len_train:
                train_acc_list.append(acc)
            elif t < self.len_train + self.len_val:
                val_acc_list.append(acc)
            else:
                test_acc_list.append(acc)

        return (loss.item(), np.mean(train_acc_list),
                np.mean(val_acc_list), np.mean(test_acc_list), test_acc_list)

    def cal_pred(self, z, decoder, node_masks):
        pred = decoder(z)[node_masks]
        return pred

    def predict(self, z, decoder, node_mask, y):
        pred = decoder(z)[node_mask]
        pred = pred.argmax(dim=-1).squeeze()
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        acc = (pred == y).sum().item() / y.shape[0]
        return float(acc)