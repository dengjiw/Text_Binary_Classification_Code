import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import scipy.io as io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc,roc_curve


def getData():
    data = io.loadmat('/root/jw_workspace/code_repository/Text_Binary_Classification/datasets/Yelp Review Sentiment Dataset/data/train_100000.mat')
    X = data['X']
    y = data['label'].squeeze()
    vocab_size = data['vocab_size'].item()
    print(X.shape,y.shape)

    #将数据按照0.8 0.1 0.1 分为训练集 验证集 测试集
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20,random_state = 404)
    X_val,X_test,y_val,y_test = train_test_split(X_val,y_val,test_size=0.50,random_state = 404)
    _,X_test,_,y_test = train_test_split(X_test,y_test,test_size=0.50,random_state = 404)
    _,X_test,_,y_test = train_test_split(X_test,y_test,test_size=0.50,random_state = 404)
    _,X_test,_,y_test = train_test_split(X_test,y_test,test_size=0.50,random_state = 404)
    return X_train,X_val,X_test,y_train,y_val,y_test,vocab_size


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]

def load_data(train_dataset,val_dataset,batch_size):
    """下载数据集，然后将其加载到内存中"""
    return (DataLoader(train_dataset, batch_size, shuffle=True,
                            num_workers=0),
            DataLoader(val_dataset, batch_size, shuffle=False,
                            num_workers=0)
    )


#创建一个模型
class Attention(nn.Module):
    def __init__(self,input_size, vocab_size,num_class):
        super(Attention, self).__init__()

        self.embed = nn.Embedding(vocab_size, 128, padding_idx=1)
        # 多层堆叠的自注意力层
        self.attention = nn.MultiheadAttention(128,1,dropout=0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size*128, num_class)

    def forward(self, x):
        x = self.embed(x)
        x,_ = self.attention(x,x,x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x 

# 看一下是在cpu还是GPU上
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义训练过程
class Trainer():
    def train(net,train_iter,val_iter,learning_rate,num_epochs,weight_decay,device):
        #初始化参数
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)  
        net.apply(init_weights)
        # 将模型移到相应device上
        net = net.to(device)
        net.device = device
        #交叉熵损失
        loss = nn.CrossEntropyLoss()

        optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay)

        best_loss = 100.0
        for epoch in range(num_epochs):
            #模型训练模式
            net.train()
        
            #训练损失和训练精度
            train_loss = []
            for batch in tqdm(train_iter):
                X, y = batch
                
                #将X和labels移到相应的device上
                X = X.to(device)
                y = y.to(device)
                
                y_hat = net(X)
                
                l = loss(y_hat, y)
                #更新梯度为0
                optimizer.zero_grad()
                #计算参数的梯度
                l.backward()
                #根据梯度更新参数
                optimizer.step()

                # 保存损失和精确度
                train_loss.append(l.item())
            #计算整体的训练损失和精确度
            train_loss = sum(train_loss) / len(train_loss)

            #打印结果
            print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}")


            #模型检测模式
            net.eval()
            
            valid_loss = []
            for batch in tqdm(val_iter):
                X, y = batch

                X = X.to(device)
                y = y.to(device)
                
                #不计算梯度
                with torch.no_grad():
                    y_hat = net(X)
                
                l = loss(y_hat, y)

                valid_loss.append(l.item())
            
            valid_loss = sum(valid_loss) / len(valid_loss)

            print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}")

            if valid_loss < best_loss :
                best_loss = valid_loss
                torch.save(net.state_dict(), 'Attention/Attention_state_dict_y.pth')
                print('saving model with loss {:.3f}'.format(best_loss))
    def test(X_test,y_test,vocab_size,device):
        model = Attention(X_train.shape[1],vocab_size,2)

        model = model.to(device)
        model.load_state_dict(torch.load('Attention/Attention_state_dict_y.pth'))

        model.eval()
        
        X_test_tensor = torch.from_numpy(X_test)
        y_test_tensor = torch.from_numpy(y_test)

        X = X_test_tensor.to(device)
        y = y_test_tensor.to(device)
                    
        #不计算梯度
        with torch.no_grad():
            y_hat = model(X)
                
        # 计算指标
        pred = y_hat.argmax(dim=-1).to(y.dtype)
        y = y.cpu().numpy()
        pred = pred.cpu().numpy()

        acc = accuracy_score(y,pred)

        prec = precision_score(y,pred)

        rec = recall_score(y,pred,zero_division=1)

        f1 = f1_score(y,pred,zero_division=1)

        fpr, tpr, _ = roc_curve(y, pred, pos_label=1)
        auc_ = auc(fpr, tpr)


        print(f"Precison = {prec:.5f},Recall = {rec:.5f},ACC = {acc:.5f},F1 = {f1:.5f},AUC = {auc_:.5f}")



if __name__ == "__main__":
    lr, num_epochs, batch_size, weight_decay= 3e-4, 50, 16,1e-3
    device = get_device()
    X_train,X_val,X_test,y_train,y_val,y_test,vocab_size = getData()
    # train_dataset = MyDataset(X_train, y_train)
    # val_dataset = MyDataset(X_val, y_val)
    # net = Attention(X_train.shape[1],vocab_size,2)
    # train_iter,val_iter = load_data(train_dataset,val_dataset,batch_size)
    # Trainer.train(net,train_iter,val_iter,lr,num_epochs,weight_decay,device)
    Trainer.test(X_test,y_test,vocab_size,device)