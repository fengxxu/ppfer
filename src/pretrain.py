
import torch
from models import UNet
from torch.utils.data import DataLoader
from baseline1.pretrain_loader import DFEW_pretrain
import torch.nn as nn
from torch.nn import L1Loss
from pretrain_params import gpu_ids, batch_size,learning_rate, num_epochs, tensorboard_log_path, output_path_time, log_path
import logging
from torch.utils.tensorboard import SummaryWriter
import os



class UnetPretrain:
    def __init__(self, mode):
        self.mode = mode
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = learning_rate


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(n_channels=3, n_classes=3).to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        self.model.cuda()

        self.criterion = L1Loss()
        self.criterion.cuda()
        self.train_dataset = DFEW_pretrain('train')
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_dataset = DFEW_pretrain("validation")
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.tensorboard_log_path = tensorboard_log_path
        self.logs_output_path_time = output_path_time


    def train_epoch(self, epoch, writer:SummaryWriter):
        self.model.train()
        all_loss = 0
        for i, (inputs, _ ) in enumerate(self.train_loader):
            inputs  = inputs.view(-1, 3, 224, 224)
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, inputs)
            loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                logging.info(f'Epoch: {epoch}/{self.num_epochs}, step{i}/{len(self.train_loader)}, iter {i}, training loss: {loss.item()}')
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(self.train_loader) + i)
        return all_loss / len(self.train_loader)
    
    def validate(self, epoch, writer: SummaryWriter):
        self.model.eval()
        with torch.no_grad():
            self.model.eval()
            all_loss = 0
            for i, (inputs, _) in enumerate(self.test_loader):
                inputs  = inputs.view(-1, 3, 224, 224)
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                all_loss += loss.item()
            logging.info(f'Epoch: {epoch}/{self.num_epochs}, validation loss: {all_loss / len(self.test_loader)}')
            writer.add_scalar('Test/Loss', all_loss / len(self.test_loader), epoch)
        return all_loss / len(self.test_loader)
        
    def run(self):
        writer = SummaryWriter(log_dir=self.tensorboard_log_path)
        best_train_loss = float('inf')
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch, writer)
            val_loss = self.validate(epoch, writer)
            # save the best model
            flag = False
            if train_loss < best_train_loss or val_loss < best_val_loss:
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    flag = True
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    flag = True
            if flag:
                best_model = os.path.join(self.logs_output_path_time, 'f_a_init_weight.pth')
                torch.save(self.model.state_dict(), best_model)
        writer.close()

if __name__ == '__main__':
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(asctime)s: %(message)s')
    unet_pretrain = UnetPretrain('train')
    unet_pretrain.run()
