import torch
#from AlexNet_206 import *
from SoyDNGP import *
from dataproces import *
from data_loader  import *
from torch.utils.data import DataLoader
from draw_pic import draw_pic
import time
def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
def train_for_epoch(train_dataloader,updater,loss,net):
        loss_ = 0.0
        t1 = time.time()
        for num_data,(genomap,target_trait) in enumerate(train_dataloader):
            genomap,target_trait = genomap.to(device),target_trait.to(device)
            trait_het  = net(genomap)
            loss_for_batch = loss(trait_het,target_trait)
            loss_+= loss_for_batch
            if isinstance(updater,torch.optim.Optimizer):
                updater.zero_grad()
                loss_for_batch.backward()
                updater.step()
        t2 = time.time()
        print('time',t2-t1)
        return loss_ / num_data

def train(train_dataloader,test_dataloader,updater,loss,epoch,net,trait): 
    t1 = time.time()
    train_loss = []
    test_loss = []
    epoch_list = []
    net.apply(init_weights)
    num_epoch = epoch
    min_loss = 999.0
    while epoch:
        net.train()
        avg_train_MSE = train_for_epoch(train_dataloader,updater,loss,net) 
        net.eval()
        with torch.no_grad():
            loss_test = 0.0
            m = 0
            for index,(test_genomap,test_trait) in enumerate(test_dataloader):
                m += 1
                test_genomap,test_trait= test_genomap.to(device),test_trait.to(device)
                y_het = net(test_genomap)
                test_MSE = loss(y_het,test_trait)
                if m == 1:
                    het = y_het.to('cpu').detach().numpy()
                    acc = test_trait.to('cpu').numpy()
                else:
                    het = np.insert(het,-1,y_het.to('cpu').detach().numpy(),axis=1)
                    acc = np.insert(acc,-1,test_trait.to('cpu').numpy(),axis=1)
                loss_test += test_MSE
            r = np.corrcoef(het,acc)[0][1]
            avg_test_MSE = loss_test / index
        print(f'{trait} epoch {num_epoch - epoch + 1} |  train loss:{avg_train_MSE},test loss:{avg_test_MSE},r:{r}')
        
        train_loss.append(avg_train_MSE.to('cpu').detach().numpy())
        test_loss.append(avg_test_MSE.to('cpu').detach().numpy())
        epoch_list.append(num_epoch - epoch + 1)
        if test_loss[-1] <= min_loss :
            torch.save(net,os.path.join(weight_save_path,f'{trait}_best.pt'))
            min_loss = test_loss[-1]
            print(f'epoch {num_epoch - epoch + 1} : weight has update')
        epoch -= 1
    t2 = time.time()
    print(f"training has finished used time : {t2 - t1}")
    torch.save(net,os.path.join(weight_save_path,f'{trait}_last.pt'))
    return train_loss,test_loss,epoch_list
trait_list = ['Palmitic','Steartic','Oleic','Linoleic','Linolenic'][::1]
#构建数据集
weight_save_path = r'/Data5/pfGao/xtwang/data/deep_learning/2000_sample/206_worker/test'
vcf_path = r'/Data5/pfGao/xtwang/data/deep_learning/2000_sample/result.vcf'
trait_path = r'/Data5/pfGao/xtwang/data/deep_learning/2000_sample/206_worker/5000_trait.csv'
save_dir = r'/Data5/pfGao/xtwang/data/deep_learning/2000_sample/206_worker/test'
for trait in trait_list:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net=SoyDNGP()
    #net,name = a.modules()
    print(torch.cuda.is_available())
    net.to(device)
    updater = torch.optim.Adam(net.parameters())
    epoch = 200
    loss = torch.nn.MSELoss()
    # for trait_for_epoch in trait_list:
    # trait_ = trait_for_epoch
    # print(type(trait_for_epoch))
    #训练数据
    data_train = data_process(vcf_path,trait_path,save_dir)
    train_data,train_label,test_data,test_label = data_train.to_dataset([trait],if_n_trait= True)
    train_dataloader = DataLoader(data_loader(train_data,train_label),batch_size=16,shuffle=True,num_workers=1)

    # 测试数据
    test_dataloader = DataLoader(data_loader(test_data,test_label),batch_size=1,shuffle=True,num_workers=1)

    save_path = os.path.join(save_dir,f'{trait}_loss.png')
    train_loss,test_loss,epoch_list = train(train_dataloader,test_dataloader,updater,loss,epoch,net,trait)
    draw_pic(epoch_list,train_loss,test_loss,save_path)
    time.sleep(5*60)

