
import matplotlib.pyplot as plt
import torch
import numpy
def draw_pic(epoch_list,train_loss,test_loss,save_path):
    plt.plot(epoch_list, train_loss, marker='o', markersize=0.05, color='skyblue', linewidth=1.5, label='train loss')
    plt.plot(epoch_list, test_loss, marker='o', markersize=0.05, color='orange', linewidth=1.5, label='test loss')
    plt.legend(loc = 'best')
    plt.title('MSE of Net')
    plt.xticks([])
    plt.xlabel(f'Epoch = {len(epoch_list)}')
    plt.ylabel('Mean Square Error')
    plt.savefig(save_path,dpi = 400)
    print("result has saved!")
