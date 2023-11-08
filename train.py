from leavesdata import LeavesData
from torch.utils.data import DataLoader
import torch
import torchvision.models as models
from torch import nn
from defs import classes_num
from tqdm import tqdm
import torch.nn.functional as F

class ModelTrain:
    def __init__(self):
        train_name = 'train.csv'
        file_path = ''
        self.train_dataset = LeavesData(train_name, file_path, mode='train')
        self.valid_dataset = LeavesData(train_name, file_path, mode='valid')
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=32, 
            shuffle=False,
            pin_memory=True,
            num_workers=5
        )
        self.valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=32, 
            shuffle=False,
            pin_memory=True,
            num_workers=5
        )
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, classes_num)
        
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.0001, weight_decay=0.00001)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode="min", factor=0.5, patience=3, min_lr = 0.000001)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0, last_epoch=-1)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)


        self.criterion = nn.CrossEntropyLoss()

        self.epoch_num = 20
    def train(self):
        best_acc = 0

        for epoch in range(self.epoch_num):
            self.model.train()

            train_loss_list = []
            train_acc_list = []

            for batch in tqdm(self.train_loader):
                imgs, labels = batch
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(imgs)

                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                acc = (logits.argmax(dim=-1) == labels).float().mean()

                train_acc_list.append(acc)
                train_loss_list.append(loss.item())
            self.scheduler.step()
            train_loss = sum(train_loss_list) / len(train_loss_list)
            train_acc = sum(train_acc_list) / len(train_acc_list)

            print(f"[ Train | {epoch + 1:03d}/{self.epoch_num:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

            self.model.eval()

            valid_loss_list = []
            valid_acc_list = []

            for batch in tqdm(self.valid_loader):
                imgs, labels = batch
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    logits = self.model(imgs)
                
                loss = self.criterion(logits, labels)

                acc = (logits.argmax(dim=-1) == labels).float().mean()

                valid_acc_list.append(acc)
                valid_loss_list.append(loss)

            valid_loss = sum(valid_loss_list) / len(valid_loss_list)
            valid_acc = sum(valid_acc_list) / len(valid_acc_list)

            print(f"[ Valid | {epoch + 1:03d}/{self.epoch_num:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(self.model.state_dict(), 'pre_res_model.ckpt')






        

        