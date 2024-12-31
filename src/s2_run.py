import torch
import torch.nn as nn
from models import get_uvit, get_feature_controller
from train_params import gpu_ids, image_size, s2_learning_rate, s2_batch_size, num_frames, num_epochs, log_path, tensorboard_log_path, output_path_time
from loader import Cremad
from torch.utils.tensorboard import SummaryWriter
import logging




class S2Solver:
    def __init__(self):
        
        # models
        self.model = get_uvit()
        self.controller = get_feature_controller()
        self.model.to('cuda')
        self.controller.to('cuda')
        self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
        self.controller = nn.DataParallel(self.controller, device_ids=gpu_ids)
        self.controller.eval()
        
        # cross entropy loss
        self.loss = nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # learning rate 
        self.lr = s2_learning_rate

        self.num_frames = num_frames
        self.num_epochs = num_epochs

        # read the infered frames load from intermediate results from stage 1 
        self.train_ds = Cremad('train',2)
        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=s2_batch_size, shuffle=True, num_workers=8)
        self.val_ds = Cremad('validation',2)
        self.val_loader = torch.utils.data.DataLoader(self.val_ds, batch_size=s2_batch_size, shuffle=False, num_workers=8)

        # tensorboard writer
        self.log_path = log_path
        self.tensorboard_log_path = tensorboard_log_path
        self.output_path_time = output_path_time

    def train(self, epoch: int, writer: SummaryWriter):
        losses=[]
        acc=[]
        self.model.train()
        self.controller.eval()
        for i, (images, labels) in enumerate(self.train_loader):
            # reshape the video frames to 4D tensor: (s2_batch_size*num_frames, 3, 224, 224)
            images = images.view(-1, 3, 224, 224)
            images = images.to('cuda')
            labels = labels[1].to('cuda')
            logits = self.model(images)
            logits = self.controller(logits)
            loss = self.loss(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            acc.append((logits.argmax(1) == labels).float().mean().item())
            if i % 10 == 0:
                print(f"Epoch {epoch}, step {i}, loss: {loss.item()}")
                writer.add_scalar('S2 Loss/train', sum(losses)/len(losses), epoch)
                writer.add_scalar('S2 Accuracy/train', sum(acc)/len(acc), epoch)
            
        return sum(losses)/len(losses), sum(acc)/len(acc)

    def evaluation(self, epoch: int, writer: SummaryWriter):
        losses=[]
        acc=[]
        self.model.eval()
        self.controller.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                images = images.view(-1, 3, 224, 224)
                images = images.to('cuda')
                labels = labels.to('cuda')
                logits = self.model(images)
                logits = self.controller(logits)
                loss = self.loss(logits, labels)
                losses.append(loss.item())
                acc.append((logits.argmax(1) == labels).float().mean().item())
                if i % 10 == 0:
                    print(f"Epoch {epoch}, step {i}, loss: {loss.item()}")
                    writer.add_scalar('S2 Loss/validation', sum(losses)/len(losses), epoch)
                    writer.add_scalar('S2 Accuracy/validation', sum(acc)/len(acc), epoch)
        return sum(losses)/len(losses), sum(acc)/len(acc)


    

    def run(self):
        writer = SummaryWriter(log_dir=self.tensorboard_log_path)
        best_loss = 100
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train(epoch, writer)
            val_loss, val_acc = self.evaluation(epoch, writer)
            logging.info(f'S2: Epoch: {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Val Loss: {val_loss}, Val Accuracy: {val_acc}')
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(self.model.state_dict(), self.output_path_time+'/s2_best_model.pth')
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.output_path_time+'/s2_best_model.pth')
        writer.close()


    def __call__(self, x):
        return self.run()


if __name__ == '__main__':
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(asctime)s: %(message)s')
    s2 = S2Solver()
    s2()



