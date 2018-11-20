
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
#from torchvision import transforms
#from torchvision.utils import make_grid


# In[4]:


df_train = pd.read_csv('data/train.csv')
df_train.head()


# In[5]:


df_test = pd.read_csv('data/test.csv')
df_test.head()


# In[6]:


from sklearn.preprocessing import StandardScaler
y = np.array(df_train['label'])
X = np.array(df_train.iloc[:,1:])
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
X = X / 255
print(X.shape, y.shape)
print(type(y), type(X))
#print(X[:5])
#print(y[:5])


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long

X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
y_val = torch.from_numpy(y_val).type(torch.LongTensor)


# In[8]:


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
#
#
        # no entity name for each layer
        # conv-batchnorm-relu * 4, maxpool * 2
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
          
        # dropout-fc-batchnorm-relu *2 + dropout-fc
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 10),
        )
          
        # weight init(conv, batchnorm, fc)
        for m in self.features.children():
            # conv layer weight initialization
            # n 为该层参数数量
            # torch.Tensor.normal_(mean=0, std=1, *, generator=None)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            # batchnorm weight init
            # torch.Tensor.fill_(value)
            # Fills self tensor with the specified value.
            elif isinstance(m, nn.BatchNorm2d):
#                 print(m.weight.data.shape,m.bias.data.shape)
#                 y=γ(x-μ)/σ+β，weight=γ，bias=β
#                 torch.Size([32]) torch.Size([32])
#                 torch.Size([32]) torch.Size([32])
#                 torch.Size([64]) torch.Size([64])
#                 torch.Size([64]) torch.Size([64])
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        for m in self.classifier.children():
            # fc weight init
            # weight Xaview initialization
            # W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
#         self.cnn1 = nn.Conv2d(1, 16, 5)
#         self.relu1=nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
#         self.cnn2=nn.Conv2d(16, 32, 5)
#         self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)
#         self.conv2_drop = nn.Dropout2d()
        
#         self.dropout = nn.Dropout()

#         self.fc1 = nn.Linear(32*4*4, 10)
        
#     def forward(self, x):
        
#         out = self.cnn1(x)
#         out = self.relu1(out)
#         out = self.maxpool1(out)
        
#         out =self.cnn2(out)
#         out = self.relu2(out)
#         out = self.maxpool2(out)
#         out = self.conv2_drop(out)
        
#         out = out.view(out.size(0), -1)
#         out = self.dropout(out)
#         out = self.fc1(out)
#         """
#         torch.Size([200, 1, 28, 28])
#         torch.Size([200, 16, 24, 24])
#         torch.Size([200, 16, 24, 24])
#         torch.Size([200, 16, 12, 12])
#         torch.Size([200, 32, 8, 8])
#         torch.Size([200, 32, 8, 8])
#         torch.Size([200, 32, 4, 4])
#         torch.Size([200, 32, 4, 4])
#         torch.Size([200, 512])
#         torch.Size([200, 512])
#         torch.Size([200, 10])
#         """
#         return out
    
batch_size = 300
n_iters = 4000
num_epochs = n_iters * batch_size / len(y_train) 
num_epochs = int(num_epochs)
print('Number of epochs is {}.'.format(num_epochs))
# TensorDataset wrapping data and target tensors.
# Each sample will be retrieved by indexing 
# both tensors along the first dimension.
train = TensorDataset(X_train, y_train)
val = TensorDataset(X_val, y_val)
#print(type(train))

# Data loader. Combines a dataset and a sampler, 
# and provides single- or multi-process iterators over the dataset.
train_loader = DataLoader(train, batch_size = batch_size)
val_loader = DataLoader(val, batch_size = batch_size)
#print(type(train_loader))

model = CNNModel()
# The input is expected to contain scores for each class, 
# a 2D Tensor of size (minibatch, C).
# A class index (0 to C-1) as the target for each value 
# of a 1D tensor of size minibatch
# The loss can be described as:
# loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j])))
#               = -x[class] + log(\sum_j exp(x[j]))
criterion  = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
learning_rate = 0.001
momentum = 0
optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
count_minibatches = 0
loss_list = []
iteration_list = []
accuracy_list = []


# In[ ]:



for epoch in range(num_epochs):
    #print('epoch: {}'.format(epoch))
    for i, (X_train_, y_train_) in enumerate(train_loader):
        X_train_ = Variable(X_train_.view([batch_size, 1, 28, 28]))
        y_train_ = Variable(y_train_)
        if torch.cuda.is_available():
            X_train_ = X_train_.cuda()
            y_train_ = y_train_.cuda()
        
        optimizer.zero_grad()
        model.train()
        scores = model(X_train_)
        # print(type(scores))
        # <class 'torch.Tensor'>
        loss = criterion (scores, y_train_)
        # print(type(loss))
        # <class 'torch.Tensor'>
        loss.backward()
        
        optimizer.step()
        del X_train_, y_train_
        count_minibatches += 1
        if count_minibatches % 25 == 0:
            correct = 0
            total =0
            for (X_val_, y_val_) in val_loader:
                X_val_ = Variable(X_val_.view([batch_size,1,28,28]))
                y_val_ = Variable(y_val_)
                if torch.cuda.is_available():
                    X_val_ = X_val_.cuda()
                    y_val_ = y_val_.cuda()
                model.eval()
                scores = model(X_val_)
                pred_y = torch.max(scores.data, 1)[1]
                total += len(y_val_)
                correct += (pred_y == y_val_).sum()
                del X_val_, y_val_
            accuracy = 100 * correct / float(total)
            
            loss_list.append(loss.data)
            iteration_list.append(count_minibatches)
            accuracy_list.append(accuracy)
            if count_minibatches % 200 == 0:
                print('Iteration: {}   Loss: {:.4f}   Accuracy:{} %'                       .format(count_minibatches, loss, accuracy))


# In[28]:


# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()


# In[29]:


X_test = df_test.values
X_test = (torch.from_numpy(X_test)                 .type(torch.FloatTensor)                 / 255)                 .view([-1,1,28,28])
print(type(X_test), X_test.shape)

model.eval()
test_pred = torch.LongTensor()
if torch.cuda.is_available():
    test_pred = test_pred.cuda()
test_loader = DataLoader(X_test, batch_size=batch_size)
for i, batch in enumerate(test_loader):
    batch = Variable(batch)
    if torch.cuda.is_available():
        batch = batch.cuda()
    scores_test = model(batch)
    del batch
    pred_y_test = torch.max(scores_test.data, 1)[1]
    test_pred = torch.cat((test_pred, pred_y_test), dim=0)

print(type(test_pred), test_pred.shape)


# In[30]:


result = pd.Series(test_pred, name='Label')
result.index += 1
result.to_csv('result_cnn.csv', index_label='ImageId', header=True)

