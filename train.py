from leavesdata import LeavesData
from torch.utils.data import DataLoader
import torch
import torchvision.models as models
from torch import nn
from defs import classes_num, num_to_class
from tqdm import tqdm
import pandas as pd
import random

class ModelTrain:
    def __init__(self):
        self.train_name = 'train.csv'
        self.test_name = 'test.csv'
        self.file_path = ''
        self.model_path = 'model.ckpt'
        self.savefile_path = 'submission.csv'
        self.train_dataset = LeavesData(self.train_name, self.file_path, mode='train')
        self.valid_dataset = LeavesData(self.train_name, self.file_path, mode='valid')
        self.test_dataset = LeavesData(self.test_name, self.file_path, mode='test')
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
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=32, 
            shuffle=False,
            num_workers=5
        )
        if torch.cuda.is_available():
            self.device = 'cuda:0'
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

        self.model_num = 2
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
                torch.save(self.model.state_dict(), self.model_path)
    def test(self):
        model_path = ['resnet50_20.ckpt',
                      'resnet50_30.ckpt',
                      ]
        predictions_list = []
        for i in range(self.model_num):
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(self.model.fc.in_features, classes_num)
            
            model = self.model.to(self.device)
            model.load_state_dict(torch.load(model_path[i], map_location=torch.device('cuda:0')))

            model.eval()

            predictions = []
            for batch in tqdm(self.test_loader):
                imgs = batch

                imgs = imgs.to(self.device)


                with torch.no_grad():
                    logits = model(imgs)
                logits = torch.argmax(logits,dim=1).reshape(-1)
                predictions.extend(logits.cpu().numpy().tolist())
            predictions_list.append(predictions)
        
        for i in range(len(predictions_list[0])):
            possible_list = []
            for j in range(self.model_num):
                possible_list.append(predictions_list[j][i])
            predictions.append(random.sample(possible_list, 1)[0])

        preds = []
        for i in predictions:
            preds.append(num_to_class[i])
        test_data = pd.read_csv(self.test_name)
        test_data['label'] = pd.Series(preds)
        submission = pd.concat([test_data['image'], test_data['label']], axis=1)
        submission.to_csv(self.savefile_path, index=False)
                    






        

        