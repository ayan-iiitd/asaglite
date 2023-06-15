## Define requirements

from nltk.tokenize import sent_tokenize as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import re
import configparser
import math
import os
from sklearn.metrics import mean_squared_error


## Read Configuration

config = configparser.ConfigParser()
config.read('config.ini')


BASE_LR = float(config['DEFAULT']['base_learning_rate'])
BATCH_SIZE = int(config['DEFAULT']['batch_size'])
FINAL_LR = float(config['DEFAULT']['final_learning_rate'])
LOG_PATH = str(config['DEFAULT']['log_path'])
GPU_ID = int(config['DEFAULT']['gpu_id'])
MODEL_SAVE_PATH = str(config['DEFAULT']['model_save_path'])
NO_EPOCHS = int(config['DEFAULT']['number_of_epochs'])
NO_OF_SENTENCES = int(config['DEFAULT']['number_of_sentences'])
OPTIM = str(config['DEFAULT']['optimizer'])
RETRAIN = bool(int(config['DEFAULT']['restart_training']))
SEED = int(config['DEFAULT']['seed'])
TEST_DATA = str(config['DEFAULT']['test_data'])
TRAIN_DATA = str(config['DEFAULT']['train_data'])


if OPTIM == 'SGD':
    OPTIM_MOMENTUM = float(config['DEFAULT']['optimizer_momentum'])
    OPTIM_WEIGHT_DECAY = float(config['DEFAULT']['optimizer_weight_decay'])

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
torch.manual_seed(SEED)


## Get data and pre-trained model

train_data = pd.read_csv(TRAIN_DATA)
test_data = pd.read_csv(TEST_DATA)
sbert_model = SentenceTransformer('all-mpnet-base-v2')



## Merge or Pad sentence embeddings

def mod_ans_emb(model_ans, st_ans):

    model_anss, st_anss = [], []
    global NO_OF_SENTENCES

    for index in range(len(model_ans)):

        m_a = sbert_model.encode(st(model_ans[index]))
        s_a = sbert_model.encode(st(st_ans[index]))

        if len(m_a) < NO_OF_SENTENCES:
            mask_size = NO_OF_SENTENCES - len(m_a)
            m_a = torch.tensor(m_a)
            m_a = torch.nn.functional.pad(m_a, (0, 0, 0, mask_size))
        
        elif len(m_a) > NO_OF_SENTENCES:
            x = torch.tensor(m_a)
            x[NO_OF_SENTENCES - 1] = x[NO_OF_SENTENCES - 1:].sum(dim = 0)
            x = x[:NO_OF_SENTENCES]
            m_a = x.clone()
        
        else:
            m_a = torch.tensor(m_a)
        
        model_anss.append(m_a)

        if len(s_a) < NO_OF_SENTENCES:
            mask_size = NO_OF_SENTENCES - len(s_a)
            s_a = torch.tensor(s_a)
            s_a = torch.nn.functional.pad(s_a, (0, 0, 0, mask_size))
        
        elif len(s_a) > NO_OF_SENTENCES:
            x = torch.tensor(s_a)
            x[NO_OF_SENTENCES - 1] = x[NO_OF_SENTENCES - 1:].sum(dim = 0)
            x = x[:NO_OF_SENTENCES]
            s_a = x.clone()
        
        else:
            s_a = torch.tensor(s_a)
        
        st_anss.append(s_a)

    model_anss = torch.stack((model_anss)).float().cuda()
    st_anss = torch.stack((st_anss)).float().cuda()

    return model_anss, st_anss



## Define Custom SCheduler

class CosineScheduler:
    
    def __init__(self,
                  max_update,
                  base_lr = 0.01,
                  final_lr = 0,
                  warmup_steps = 0,
                  warmup_begin_lr = 0):
        
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        
        if epoch <= self.max_update:
        
            self.base_lr = self.final_lr + (
        
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        
        return self.base_lr



## Define custom PyTorch dataset class

class marksDataset(torch.utils.data.Dataset):
  
  def __init__(self, df_data):

    self.model_ans = df_data['model_ans']
    self.st_ans = df_data['student_ans']
    self.marks = df_data['marks']
    self.remove_tag = re.compile(r'<[^>]+>')
 
  def __len__(self):
    return len(self.model_ans)
   
  def __getitem__(self, idx):
    
    ref_ans = self.remove_tag.sub('', self.model_ans.iloc[idx])
    actual_ans = self.remove_tag.sub('', self.st_ans.iloc[idx])
    grade = self.marks.iloc[idx]

    return ref_ans, actual_ans, grade



## Load data

marks_data_train = marksDataset(train_data)
train_loader = torch.utils.data.DataLoader(marks_data_train, batch_size = BATCH_SIZE, shuffle = False)

marks_data_test = marksDataset(test_data)
test_loader = torch.utils.data.DataLoader(marks_data_test, batch_size = BATCH_SIZE, shuffle = False)


## Define model

class both_cnn(torch.nn.Module):
    
    def __init__(self):

        super(both_cnn, self).__init__()
        self.l1 = torch.nn.Conv1d(768*4, 768, 1, bias = True)
        self.l12 = torch.nn.Conv1d(768*4, 768, 1, bias = True)
        self.linears = torch.nn.Sequential(
            torch.nn.Linear(768*2, 384),
            torch.nn.PReLU(),
            torch.nn.Linear(384, 192),
            torch.nn.PReLU(),
            torch.nn.Linear(192, 96),
            torch.nn.PReLU(),
            torch.nn.Linear(96, 192),
            torch.nn.PReLU(),
            torch.nn.Linear(192, 384),
            torch.nn.PReLU(),
            torch.nn.Linear(384, 1))

  
    def forward(self, model, student):
        
        model = model.flatten(start_dim = 1)
        student = student.flatten(start_dim = 1)

        model = torch.transpose(model, 1, 0)
        student = torch.transpose(student, 1, 0)
        
        merged_model = self.l1(model)
        merged_model = merged_model.T.float()
        
        merged_student = self.l12(student)
        merged_student = merged_student.T.float()

        merged = torch.cat((merged_model, merged_student), dim = 1).float()
        y_pred_continuous = self.linears(merged).float()

        y_pred = torch.tensor(())

        for i in y_pred_continuous.squeeze():

            y = torch.zeros(1)

            if i < 0.2:
                y[0] = 0.0
            elif i < 0.75:
                y[0] = 0.5
            elif i < 1.2:
                y[0] = 1.0
            elif i < 1.65:
                y[0] = 1.5
            elif i < 1.8:
                y[0] = 1.75
            elif i < 2.2:
                y[0] = 2.0
            elif i < 2.4:
                y[0] = 2.25
            elif i < 2.8:
                y[0] = 2.5
            elif i < 3.1:
                y[0] = 3.0
            elif i < 3.35:
                y[0] = 3.25
            elif i < 3.575:
                y[0] = 3.5
            elif i < 3.7:
                y[0] = 3.625
            elif i < 3.85:
                y[0] = 3.75
            elif i < 4.1:
                y[0] = 4.0
            elif i < 4.25:
                y[0] = 4.125
            elif i < 4.65:
                y[0] = 4.5
            elif i < 4.8:
                y[0] = 4.75
            else:
                y[0] = 5.0
            
            y_pred = torch.cat((y_pred, y), 0)

        return y_pred, y_pred_continuous.squeeze()



## Configure model and training parameters

model = both_cnn()
model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),\
                            lr = 0.01,\
                            momentum = 0.9,\
                            weight_decay = 5e-4)
scheduler = CosineScheduler(max_update = NO_EPOCHS * len(train_loader),\
                            warmup_steps = NO_EPOCHS * len(train_loader)/10,\
                            base_lr = 0.01,\
                            final_lr = 0.0001)



## Load saved model

if RETRAIN:
    saved_model = torch.load(str(config['DEFAULT']['model_to_retrain_path']))
    no_steps = saved_model['steps']
    model.load_state_dict(saved_model['model_state_dict'])
else:
    no_steps = 0



## Train model

min_tr_loss, min_tst_loss = 99999, 99999


for epoch in range(NO_EPOCHS):

    batch_rmse = []
    
    for batch_index, (model_ans, st_ans, marks) in enumerate(train_loader):
        
        model_anss, st_anss = mod_ans_emb(model_ans, st_ans)
        
        optimizer.zero_grad()
        
        y_pred, y_pred_continuous = model(model_anss, st_anss)

        marks = marks.cuda()
        loss = criterion(y_pred_continuous.float(), marks.float())

        y = marks.tolist()
        y_cap = y_pred.tolist()

        rmse = math.sqrt(mean_squared_error(y, y_cap))
        batch_rmse.append(rmse)

        for param_group in optimizer.param_groups:
          param_group['lr'] = scheduler(no_steps)
        
        no_steps += 1
        
        with open(LOG_PATH + "/train.log", "a") as log_file:
            log_file.write(f'Train Loss:\tEpoch:\t{epoch}\tBatch\t{batch_index}\tLoss:\t{loss.item()}\tMin Train RMSE:\t{min_tr_loss}\n')
        
        print(f'Train Loss:\tEpoch:\t{epoch}\tBatch\t{batch_index}\tLoss:\t{loss.item()}\tMin Train RMSE:\t{min_tr_loss}')

        loss.backward()
        optimizer.step()


    print()
    batch_rmse = sum(batch_rmse)/len(batch_rmse)
    
    if batch_rmse < min_tr_loss:
        min_tr_loss = batch_rmse
        path = os.path.join(MODEL_SAVE_PATH, 'model_rmse_' + str(batch_rmse) + '_epoch_' + str(epoch) + '_checkpoints.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'steps': no_steps,
            'learning_rate': param_group['lr']
            }, path)

    
    with torch.no_grad():

        batch_rmse = []
        
        for batch_index, (model_ans, st_ans, marks) in enumerate(test_loader):

            model_anss, st_anss = mod_ans_emb(model_ans, st_ans)
            y_pred, y_pred_continuous = model(model_anss, st_anss)

            marks = marks.cuda()
            loss = criterion(y_pred_continuous.float(), marks.float())

            y = marks.tolist()
            y_cap = y_pred.tolist()

            rmse = math.sqrt(mean_squared_error(y, y_cap))
            batch_rmse.append(rmse)

            with open(LOG_PATH + "/test.log", "a") as log_file:
                log_file.write(f'Test Loss:\tEpoch:\t{epoch}\tBatch\t{batch_index}\tLoss:\t{loss.item()}\tMin Test RMSE:\t{min_tst_loss}\n')
            
            print(f'Test Loss:\tEpoch:\t{epoch}\tBatch\t{batch_index}\tLoss:\t{loss.item()}\tMin Test RMSE:\t{min_tst_loss}')
        
        print()
        batch_rmse = sum(batch_rmse)/len(batch_rmse)
        
        if batch_rmse < min_tst_loss:
            
            min_tst_loss = batch_rmse
            path = os.path.join(MODEL_SAVE_PATH, 'test_model_rmse_' + str(batch_rmse) + '_epoch_' + str(epoch) + '_checkpoints.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'steps': no_steps,
                'learning_rate': param_group['lr']
                }, path)
