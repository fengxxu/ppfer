import torch
import torch.nn as nn
from models import get_r3d
from train_params import log_path, tensorboard_log_path, output_path_time, num_epochs,learning_rate, gpu_ids



class S3Solver:
    def __init__(self):

        self.num_epochs = num_epochs
        self.lr = learning_rate


        # models
        self.model = get_r3d()
        self.model.to('cuda')
        self.model = nn.DataParallel(self.model, device_ids=gpu_ids)

        # cross entropy loss
        self.loss = nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # read the infered frames load from intermediate results from stage 1 
        self.train_ds = Cremad('train',3)
        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=s2_batch_size, shuffle=True, num_workers=8)
        self.val_ds = Cremad('validation',3)
        self.val_loader = torch.utils.data.DataLoader(self.val_ds, batch_size=s2_batch_size, shuffle=False, num_workers=8)


    def train(self, epoch: int, writer: SummaryWriter):
        losses=[]
        acc=[]
        self.model.train()
        for i, (images, labels) in enumerate(self.train_loader):
            images = images.view(-1, 3, 224, 224)
            images = images.to('cuda')
            labels = labels[1].to('cuda')
            logits = self.model(images)
            loss = self.loss(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if i % 10 == 0:
                print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')
                print(f'Epoch: {epoch}, Iteration: {i}, Accuracy: {(logits.argmax(1) == labels).float().mean().item()}')
            writer.add_scalar('train_loss', sum(losses)/len(losses), epoch)
            writer.add_scalar('train_accuracy', sum(acc)/len(acc), epoch)
        return sum(losses)/len(losses)

    def validate(self, epoch: int, writer: SummaryWriter):
        losses=[]
        acc=[]
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                images = images.view(-1, 3, 224, 224)
                images = images.to('cuda')
                labels = labels[1].to('cuda')
                logits = self.model(images)
                loss = self.loss(logits, labels)
                losses.append(loss.item())
                acc.append((logits.argmax(1) == labels).float().mean().item())
        writer.add_scalar('val_loss', sum(losses)/len(losses), epoch)
        writer.add_scalar('val_accuracy', sum(acc)/len(acc), epoch)
        return sum(losses)/len(losses), sum(acc)/len(acc)

    def run(self):
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        writer = SummaryWriter(tensorboard_log_path)
        for epoch in range(self.num_epochs):
            train_loss = self.train(epoch, writer)
            val_loss, val_acc = self.validate(epoch, writer)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'{output_path_time}/best_model.pth')
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(self.model.state_dict(), f'{output_path_time}/best_train_model.pth')

        