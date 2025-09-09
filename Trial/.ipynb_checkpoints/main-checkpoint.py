# Configuration variables
BATCH_SIZE = 128
MODEL_TYPE = 'ConTL'
DATA_PATH = r'D:\SEED-IV\eeg_feature_smooth'
LEARNING_RATE = 0.0002
W_MODE = 'w'
NUM_EPOCHS = 500
B1 = 0.5
B2 = 0.999
NUM_TRIALS = 1
LABEL_TYPE = 'valence_labels'
LSTM_HIDDEN_SIZE = 8
SUBJECT = '15'
SAVE_FILE_NAME = 'seed4_result.csv'
N_UNITS = 106
N_CLASSES = 4

import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.tools import getPathList, getSEED4Data
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.optim as optim
import models

class Solver(object):
    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

    def build(self, cuda=True):
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        if cuda:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = getattr(models, MODEL_TYPE)().to(self.device)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LEARNING_RATE,
            betas=(B1, B2)
        )

    def train(self):
        saveroot = f'checkpoints/{MODEL_TYPE}'
        saved_model_name = f'sub{SUBJECT}_model.ckpt'
        save_optim_name = f'sub{SUBJECT}_optim_best.std'
        save_path = os.path.join(saveroot, '4')
        os.makedirs(save_path, exist_ok=True)

        curr_patience = patience = 7
        best_valid_loss = float('inf')
        num_trials = NUM_TRIALS

        for epoch in range(NUM_EPOCHS):
            self.model.train()
            self.train_help(epoch, self.train_data_loader)
            valid_loss, valid_acc, valid_f1_score = self.evaluate(self.val_data_loader)

            if valid_loss < best_valid_loss:
                curr_patience = patience
                best_valid_loss = valid_loss
                print(f'========Epoch: {epoch}========')
                print('valid_loss: {:.5f} | valid_acc: {:.3f} | valid_f1_score: {:.3f}'.format(valid_loss, valid_acc, valid_f1_score))
                print('current patience: {}'.format(curr_patience))
                torch.save(self.model.state_dict(), os.path.join(saveroot, '4', saved_model_name))
                torch.save(self.optimizer.state_dict(), os.path.join(saveroot, '4', save_optim_name))
            else:
                curr_patience -= 1
                if curr_patience <= 0:
                    self.model.load_state_dict(torch.load(os.path.join(saveroot, '4', saved_model_name)))
                    self.optimizer.load_state_dict(torch.load(os.path.join(saveroot, '4', save_optim_name)))
                    curr_patience = patience
                    num_trials -= 1
                    print('trials left: ', num_trials)
            if num_trials <= 0:
                print('Running out of patience, training stops!')
                return ''

    def train_help(self, epoch, data_loader):
        left_epochs = 4
        for features, labels in data_loader:
            features = features.to(self.device)
            if left_epochs == 4:
                self.optimizer.zero_grad()
            left_epochs -= 1
            labels = labels.to(self.device)
            x, outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            if not left_epochs:
                self.optimizer.step()
                left_epochs = 4

    def evaluate(self, data_loader, is_load=False):
        if is_load:
            self.model.load_state_dict(torch.load(os.path.join(f'checkpoints/{MODEL_TYPE}', '4', f'sub{SUBJECT}_model.ckpt')))
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            epoch_loss = 0.0
            epoch_f1_score = 0.0
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                epoch_loss, correct, total, epoch_f1_score = self.eval_help(features, labels, epoch_loss, epoch_f1_score, total, correct)
            epoch_loss /= len(data_loader)
            epoch_f1_score /= len(data_loader)
        return epoch_loss, (100 * (correct / total)), 100 * epoch_f1_score

    def eval_help(self, features, labels, epoch_loss, epoch_f1_score, total, correct):
        x, outputs = self.model(features)
        loss = self.criterion(outputs, labels)
        epoch_loss += loss
        predicted = torch.argmax(outputs.data, 1)
        epoch_f1_score += f1_score(list(labels.detach().cpu().numpy()), list(predicted.detach().cpu().numpy()), average='weighted')
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        return epoch_loss, correct, total, epoch_f1_score

class CusEEGDataset(torch.utils.data.Dataset):
    def __init__(self, feature_list, label_list):
        self.feature_list = feature_list
        self.label_list = label_list

    def __getitem__(self, index):
        feature = np.reshape(self.feature_list[index], (self.feature_list[index].shape[0] * self.feature_list[index].shape[1]))
        feature = np.expand_dims(feature, axis=0)
        feature = torch.from_numpy(feature).float()
        label = torch.from_numpy(np.asarray(self.label_list[index])).long()
        return feature, label

    def __len__(self):
        return len(self.label_list)

def train(X_train, X_val, X_test, y_train, y_val, y_test):
    train_data = CusEEGDataset(X_train, y_train)
    val_data = CusEEGDataset(X_val, y_val)
    test_data = CusEEGDataset(X_test, y_test)

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    solver = Solver(train_data_loader, val_data_loader, test_data_loader)
    solver.build()
    solver.train()

    test_loss, test_acc, test_f1_score = solver.evaluate(test_data_loader, is_load=True)
    print('test_loss: {:.3f} | test_acc: {:.3f} | test_f1_score: {:.3f}'.format(test_loss, test_acc, test_f1_score))
    return test_acc, test_f1_score

def printResults(acc_avg, acc_std, f1_avg, f1_std):
    print(f'=======SEED-IV | {MODEL_TYPE}_result=======')
    print('subject_test_acc_mean:', acc_avg)
    print('subject_test_acc_std:', acc_std)
    print('subject_test_f1_avg:', f1_avg)
    print('subject_test_f1_std:', f1_std)

def saveResults(acc_avg, acc_std, f1_avg, f1_std):
    with open(f'./result/{SAVE_FILE_NAME}', W_MODE) as f:
        if W_MODE == 'w':
            f.write('LSTM,')
            f.write('label_type,')
            f.write('lstm_hidden_units,')
            f.write('model,')
            f.write('classification accuracy,')
            f.write('acc std,')
            f.write('F1,')
            f.write('F1 std,')
            f.write('data,')
            f.write('n_ranges,')
            f.write('\n')
        f.write('True,')
        f.write(LABEL_TYPE + ',')
        f.write(str(LSTM_HIDDEN_SIZE) + ',')
        f.write(MODEL_TYPE + ',')
        f.write(str(round(acc_avg, 2)) + ',')
        f.write(str(round(acc_std, 2)) + ',')
        f.write(str(round(f1_avg, 2)) + ',')
        f.write(str(round(f1_std, 2)) + ',')
        f.write('4,')
        f.write(str(N_CLASSES) + '\n')

def calcAccMeanStd(acc_list, f1_list):
    acc_avg = np.mean(acc_list)
    acc_std = np.std(acc_list)
    f1_avg = np.mean(f1_list)
    f1_std = np.std(f1_list)
    return acc_avg, acc_std, f1_avg, f1_std

def get_subject_file_map(filelist):
    file_map = {}
    for path in filelist:
        splits = path.split(os.path.sep)
        fn = splits[-1]
        current_subject = fn.split('_')[0]
        if current_subject not in file_map:
            file_map[current_subject] = []
        file_map[current_subject].append(path)
    return file_map

def write_cls_accs_for_total_subjects(acc_list, savepath):
    with open(savepath, W_MODE) as f:
        if W_MODE == 'w':
            f.write('Subjects,')
            for i in range(1, len(acc_list) + 1):
                f.write(str(i))
                if not i == len(acc_list):
                    f.write(',')
            f.write('\n')
        f.write(MODEL_TYPE + ',')
        for i in range(len(acc_list)):
            f.write(str(round(acc_list[i], 2)))
            if i != (len(acc_list) - 1):
                f.write(',')
        f.write('\n')

def run():
    if not os.path.exists('result'):
        os.makedirs('result', exist_ok=True)

    test_acc_list = []
    test_f1_list = []

    n_total_subjects = 15
    all_files = getPathList(DATA_PATH)
    file_map = get_subject_file_map(all_files)

    for su in range(1, n_total_subjects + 1):
        print(f'================Subject{su}===================')
        global SUBJECT
        SUBJECT = str(su)

        train_x, test_x, train_y, test_y = getSEED4Data(file_map[SUBJECT])
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=7)

        test_acc, test_f1_score = train(train_x, val_x, test_x, train_y, val_y, test_y)
        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1_score)

    write_cls_accs_for_total_subjects(test_acc_list, f'./result/subject_wise_accuracy_for_seed4_data.csv')
    subject_test_acc_avg, subject_test_acc_std, subject_test_f1_avg, subject_test_f1_std = calcAccMeanStd(test_acc_list, test_f1_list)
    printResults(subject_test_acc_avg, subject_test_acc_std, subject_test_f1_avg, subject_test_f1_std)
    saveResults(subject_test_acc_avg, subject_test_acc_std, subject_test_f1_avg, subject_test_f1_std)

if __name__ == '__main__':
    SEED = 336
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    run()