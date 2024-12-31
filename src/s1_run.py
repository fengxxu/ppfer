import torch
import torch.nn as nn
from train_params import batch_size, learning_rate, num_epochs, num_workers, gpu_ids, log_path, tensorboard_log_path, output_path_time
from models import get_unet, get_controller, get_r3d, get_resnet50
from loader import Cremad
from torch.utils.tensorboard import SummaryWriter
from pytorch_wavelets import DWTForward, DWTInverse
from sklearn.metrics import accuracy_score 

import logging


class Sovler:

    def __init__(self) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.lr = learning_rate
        self.gpu_ids = gpu_ids
        self.dwt = DWTForward(J=1, wave='db1')
        self.idwt = DWTInverse(wave='db1')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # models
        self.f_hpr = get_unet()
        self.f_lpr = get_unet()
        self.c_hpr = get_controller()
        self.c_lpr = get_controller()
        self.f_hpr.to(self.device)
        self.f_lpr.to(self.device)
        self.c_hpr.to(self.device)
        self.c_lpr.to(self.device)
        self.f_hpr = nn.DataParallel(self.f_hpr, device_ids=self.gpu_ids)
        self.f_lpr = nn.DataParallel(self.f_lpr, device_ids=self.gpu_ids)
        self.c_hpr = nn.DataParallel(self.c_hpr, device_ids=self.gpu_ids)
        self.c_lpr = nn.DataParallel(self.c_lpr, device_ids=self.gpu_ids)

        # optimizers
        self.optimizer_hpr = torch.optim.Adam(self.f_hpr.parameters(), lr=self.lr)
        self.optimizer_lpr = torch.optim.Adam(self.f_lpr.parameters(), lr=self.lr)

        # loss, the loss is after c_hpr and c_lpr
        self.hpr_loss = nn.CrossEntropyLoss()
        self.lpr_loss = nn.CrossEntropyLoss()

        # dataloader
        self.train_ds = Cremad('train',1)
        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_ds = Cremad('validation',1)
        self.val_loader = torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # tensorboard writer
        self.log_path = log_path
        self.tensorboard_log_path = tensorboard_log_path
        self.output_path_time = output_path_time

        
    def train(self, epoch: int, writer: SummaryWriter):
        losses_h=[]
        losses_l=[]
        acc_h=[]
        acc_l=[]
        # f_hpr, f_lpr in train, c_hpr, c_lpr in eval, and freezing during training
        self.f_hpr.train()
        self.f_lpr.train()
        self.c_hpr.eval()
        self.c_lpr.eval()
        for i, (imgs, labels) in enumerate(self.train_loader):
            # imgs: (batch_size, 16, 3, 112, 112)
            # labels: (batch_size, (identity, emotion))
            labels = labels.to(self.device)
            # imgs to 16, batch_size, 3, 112, 112
            imgs = imgs.view(-1, 3, 112, 112)
            # wavelet transform imgs
            low, high = self.dwt(imgs)
            low_reshape = low.view(self.batch_size, 16, 3, *low.shape[2:])
            high_reshape = high[0].view(self.batch_size, 16, 3, *high[0].shape[2:])

            low_reshape = low_reshape.to(self.device)
            high_reshape = high_reshape.to(self.device)

            # forward pass
            high_x = self.f_hpr(high_reshape)
            high_x = self.c_hpr(high_x)
            low_x = self.f_lpr(low_reshape)
            low_x = self.c_lpr(low_x)

            # accuracy
            _, high_preds = torch.max(high_x, 1)
            _, low_preds = torch.max(low_x, 1)
            high_acc = accuracy_score(labels[:, 0], high_preds)
            low_acc = accuracy_score(labels[:, 1], low_preds)
            acc_h.append(high_acc)
            acc_l.append(low_acc)

            # calculate loss
            high_loss = -self.hpr_loss(high_x, labels)
            low_loss = -self.lpr_loss(low_x, labels)
            losses_h.append(high_loss.item())
            losses_l.append(low_loss.item())

            # backpropagation
            self.optimizer_hpr.zero_grad()
            self.optimizer_lpr.zero_grad()
            high_loss.backward()
            low_loss.backward()

            self.optimizer_hpr.step()
            self.optimizer_lpr.step()
            if i % 10 == 0:
                logging.info(f'S1 Training Epoch: {epoch}, Iteration: {i}, High Loss: {high_loss.item()}, Low Loss: {low_loss.item()}')
                writer.add_scalar('S1 Training Loss/High', high_loss.item(), epoch*self.batch_size+i)
                writer.add_scalar('S1 Training Loss/Low', low_loss.item(), epoch*self.batch_size+i)
                writer.add_scalar('S1 Training Accuracy/High', high_acc, epoch*self.batch_size+i)
                writer.add_scalar('S1 Training Accuracy/Low', low_acc, epoch*self.batch_size+i)
            
        return sum(losses_h)/len(losses_h), sum(losses_l)/len(losses_l)

    def eval(self, epoch: int, writer: SummaryWriter):
        acc_h,acc_l,losses_h,losses_l =[]
        self.f_hpr.eval()
        self.f_lpr.eval()
        self.c_hpr.eval()
        self.c_lpr.eval()

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(self.val_loader):
                labels = labels.to(self.device)
                imgs = imgs.view(-1, 3, 112, 112)
                low, high = self.dwt(imgs)
                low_reshape = low.view(self.batch_size, 16, 3, *low.shape[2:])
                high_reshape = high[0].view(self.batch_size, 16, 3, *high[0].shape[2:])

                low_reshape = low_reshape.to(self.device)
                high_reshape = high_reshape.to(self.device)

                high_x = self.f_hpr(high_reshape)
                high_x = self.c_hpr(high_x)
                low_x = self.f_lpr(low_reshape)
                low_x = self.c_lpr(low_x)

                _, high_preds = torch.max(high_x, 1)
                _, low_preds = torch.max(low_x, 1)
                high_acc = accuracy_score(labels[:, 0], high_preds)
                low_acc = accuracy_score(labels[:, 1], low_preds)
                acc_h.append(high_acc)
                acc_l.append(low_acc)

                high_loss = -self.hpr_loss(high_x, labels)
                low_loss = -self.lpr_loss(low_x, labels)
                losses_h.append(high_loss.item())
                losses_l.append(low_loss.item())

                if i % 10 == 0:
                    logging.info(f'S1 Vali: Epoch: {epoch}, Iteration: {i}, High Loss: {high_loss.item()}, Low Loss: {low_loss.item()}')
                    writer.add_scalar('S1 Vali Loss/High', high_loss.item(), epoch*self.batch_size+i)
                    writer.add_scalar('S1 Vali Loss/Low', low_loss.item(), epoch*self.batch_size+i)
                    writer.add_scalar('S1 Vali Accuracy/High', high_acc, epoch*self.batch_size+i)
                    writer.add_scalar('S1 Vali Accuracy/Low', low_acc, epoch*self.batch_size+i)
        return sum(losses_h)/len(losses_h), sum(losses_l)/len(losses_l)


    def run(self):
        writer = SummaryWriter()
        # initialize as the smallest negative value
        best_f_hpr_train_loss = float('-inf')
        best_f_lpr_train_loss = float('-inf')
        best_f_hpr_loss = float('-inf')
        best_f_lpr_loss = float('-inf')
        for epoch in range(self.num_epochs):
            train_loss_h, train_loss_l = self.train(epoch, writer)
            val_loss_h, val_loss_l = self.eval(epoch, writer)

            # save the best model
            if train_loss_h > best_f_hpr_train_loss:
                best_f_hpr_train_loss = train_loss_h
                torch.save(self.f_hpr.state_dict(), self.output_path_time + 'f_hpr_train_best.pth')
            if train_loss_l > best_f_lpr_train_loss:
                best_f_lpr_train_loss = train_loss_l
                torch.save(self.f_lpr.state_dict(), self.output_path_time + 'f_lpr_train_best.pth')
            if val_loss_h > best_f_hpr_loss:
                best_f_hpr_loss = val_loss_h
                torch.save(self.f_hpr.state_dict(), self.output_path_time + 'f_hpr_best.pth')
            if val_loss_l > best_f_lpr_loss:
                best_f_lpr_loss = val_loss_l
                torch.save(self.f_lpr.state_dict(), self.output_path_time + 'f_lpr_best.pth')
    
        writer.close()



if __name__ == "__main__":
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(asctime)s: %(message)s')
    solver = Sovler()
    solver.run()

