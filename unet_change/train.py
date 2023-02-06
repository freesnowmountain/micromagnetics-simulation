from pix2Topix import pix2pixG_32
import argparse
from mydatasets import CreateDatasets
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import xlwt
import os
from PIL import Image
import matplotlib.pyplot as plt
import  img
import time

def plot_loss(loss_g):
    fig2=plt.subplot()
    plt.plot(loss_g)
    plt.show()

    try:
      import os
      os.makedirs('output')
    except FileExistsError:
      pass
    plt.savefig('output/loss.png')


def train_one_epoch(G, train_loader, optim_G, device,  epoch, loss_t1,epoch_number,output_path):
    #加载进度条
    pd = tqdm(train_loader)
    G.train()
    print('\n')
    print("epoch=",epoch)
    loss_sum=0
    total = 0

    train_out=[]
    train_label=[]
    for data in pd:
        
        input = data[0].to(device)#16*3*32*32
        label = data[1].to(device)#16*3*32*32
        #比较模型所需时间
        output=G(input)#(16,3,32,32)    
        out1=torch.zeros_like(output)

        #输出归一化
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'    
        norm = torch.sqrt( torch.einsum( 'ijkl,ijkl -> ikl', 
                                         output, output) )
        for l in range(3):
            out1[:,l,:,:] = output[:,l,:,:]/norm
        
        loss_1 = loss_t1(out1, label)
        loss_1.backward()
        optim_G.step()
        optim_G.zero_grad()
        loss_sum=loss_1.item()+loss_sum
        total += 1

        train_out.append(out1)
        train_label.append(label)
        
        # input = data[0].to(device)#16*3*32*32
        # label = data[1].to(device)#16*3*32*32
        # #比较模型所需时间
        # output=G(input)#(16,3,32,32)    
        # loss_1 = loss_t1(output, label)
        # loss_1.backward()
        # optim_G.step()
        # optim_G.zero_grad()
        # loss_sum=loss_1.item()+loss_sum
        # total += 1

        # train_out.append(output)
        # train_label.append(label)
    
    if epoch==epoch_number-1:
        #保存out和label
        output_array=torch.stack(train_out).permute(0,1,3,4,2).reshape(-1,32,32,3)
        label_array=torch.stack(train_label).permute(0,1,3,4,2).reshape(-1,32,32,3)

        output_array=output_array.detach().cpu().numpy() 
        label_array=label_array.detach().cpu().numpy() 

        #输出的数组路径
        output_npy_path = output_path+ 'output.npy'
        #打乱后的label的数组路径
        label_npy_path = output_path+ 'label.npy'

        np.save(output_npy_path,output_array)
        np.save(label_npy_path,label_array)
        loss_sum = loss_sum/total
        turn_img(output_array,label_array,output_path)
        return loss_sum


    loss_sum=loss_sum/total

    return loss_sum

#绘制彩色图像
def turn_img(output_array,label_array,output_path):
    
    #output的彩色图片路径
    output_color_path = output_path + 'output_color_img/'
    if not os.path.exists(output_color_path):
        os.makedirs(output_color_path)          
    img.npy2jpg(output_array,output_color_path) 

    #label的彩色图片路径
    label_color_path = output_path + 'label_color_img/'
    if not os.path.exists(label_color_path):
        os.makedirs(label_color_path)         
    img.npy2jpg(label_array,label_color_path) 
 
    

def turn_histogram(output_array,label_array,output_path):
    #output的直方图路径
    output_Histogram_path = output_path + 'output_histogram_img/'
    if not os.path.exists(output_Histogram_path):
        os.makedirs(output_Histogram_path)         
    img.plot_histogram(output_array,output_Histogram_path)
    #label的直方图路径
    label_Histogram_path = output_path + 'label_histogram_img/'
    if not os.path.exists(label_Histogram_path):
        os.makedirs(label_Histogram_path)        
    img.plot_histogram(label_array,label_Histogram_path)


def train(opt):

    #输入
    batch = opt.batch
    img_size = opt.imgsize
    epoch_number =opt.epoch_number
    spin_case =opt.spin_case

    #输入npy数据文件夹
    npy_path = "npy_data/size_"+str(img_size)+"/case_"+str(spin_case)
    train_label_path = npy_path + "/train_label.npy"
    train_input_path = npy_path + "/train_input.npy"
    test_label_path = npy_path + "/test_label.npy"
    test_input_path = npy_path + "/test_input.npy"

    #输出文件夹
    train_output_path = "output/size_"+str(img_size)+"/case_"+str(spin_case) + '/train/'
    test_output_path = "output/size_"+str(img_size)+"/case_"+str(spin_case) + '/test/'

    #excel保存loss的数据
    excel_path = 'unet_change/excel/'+'loss_size'+ str(img_size) + '_case' +str(spin_case) + 'no_norm'+'.xls'

    #模型保存路径
    model_path = "unet_change/weights/"
    train_model_path = model_path + "train_model_" + str(img_size) + "_case" +str(spin_case) +".pth"
    test_model_path = model_path + "test_model_" + str(img_size) + "_case" +str(spin_case) +".pth"

    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path)  
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)  
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #pdb.set_trace()
    # 加载数据集
    print("当前行为为：加载数据集,数据大小为",img_size,"*",img_size)

    train_datasets = CreateDatasets(train_input_path,train_label_path)
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch, shuffle=True, num_workers=opt.numworker,
                              drop_last=True)
    
    test_dataset = CreateDatasets(test_input_path,test_label_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False,
                              drop_last=True)
    
    # 实例化网络
    print("当前行为为：实例化网络")
    pix_G = pix2pixG_32().to(device)

    # 定义优化器和损失函数
    optim_G = optim.Adam(pix_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    #创建excel
    workbook=xlwt.Workbook("Epoch_loss_mse")
    sheet=workbook.add_sheet("loss+mse")
    col=["Epoch","loss","mse"]
    for i in range(len(col)):
        sheet.write(0,i,col[i])
    loss_g=[]
    loss_t1 = nn.MSELoss()

    #test参数
    best_loss = 1
    best_epoch = 0
    test_outnpy_path = test_output_path + 'output.npy'
    test_outlabel_path = test_output_path + 'label.npy'
    # 开始训练
    for epoch in range(epoch_number): 
        loss_sum = train_one_epoch(G=pix_G,  train_loader=train_loader,optim_G=optim_G, device=device, 
        epoch=epoch,loss_t1=loss_t1,epoch_number=epoch_number,output_path=train_output_path
        )

        print("epoch=%d loss=%.4f"%(epoch, loss_sum))
        loss_g.append(loss_sum)

        mse=0
        label=[epoch+1,loss_sum,mse]
        #将数据保存进excel
        for i in range(len(label)):
            sheet.write(epoch+1,i,label[i])

        # 保存模型
        torch.save({
            'G_model': pix_G.state_dict(),
            'epoch': epoch
        }, train_model_path)

        #进行预测
        test_loss,test_npy,test_label_arr = test(img_size,train_model_path,test_loader)
        print('test:',epoch,'finish')
        if test_loss < best_loss:
            best_loss = test_loss 
            best_epoch = epoch
            best_npy = test_npy.numpy()
            best_label = test_label_arr.numpy()
            torch.save({
                'G_model': pix_G.state_dict(),
                'epoch': epoch
            }, test_model_path)
            
            np.save(test_outnpy_path,best_npy)
            np.save(test_outlabel_path,best_label)

    workbook.save(excel_path)
    #plot_loss(loss_g)


    #预测的数据处理
    print("******************")       
    print("test result:")
    print("best epoch:{}".format(best_epoch))      
    print("Test loss is %.4f"%(best_loss))
    print("Saving the model output to %s"%test_outnpy_path) 
    #转化为图片
    test_output_array = np.load(test_outnpy_path)
    test_label_array = np.load(test_outlabel_path)
    turn_img(test_output_array,test_label_array,test_output_path)


    #绘制最后的直方图
    print('训练预测已完成，接下来为直方图绘制')
    train_arr = np.load(train_output_path + 'output.npy')
    train_label = np.load(train_output_path + 'label.npy')

    time_start = time.time()
    turn_histogram(train_arr,train_label,train_output_path)
    turn_histogram(test_output_array,test_label_array,test_output_path)
    time_finish = time.time()

    print(' 两个直方图绘制时间: {:f} s '.format(time_finish-time_start))

    print('程序执行完毕')
    


    



def test(img_size,model_path,test_loader):
    #要修改的地址和变量：输入和预测的文件地址，模型数据的保存地址
    # 实例化网络
    G = pix2pixG_32().to('cuda')
    G.eval()
    ckpt = torch.load(model_path)
    G.load_state_dict(ckpt['G_model'], strict=False)

    mse = torch.nn.MSELoss()

    test_hd=[]
    test_label=[]
    tot = 0
    loss_m = 0
    epoch = 0

    for dt in test_loader:
        spin = dt[0].to('cuda')
        tgt  = dt[1].to('cuda')

        out = G(spin)#(16,256,256,3)
        out2=torch.zeros_like(out)
        #输出归一化
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'    
        norm = torch.sqrt( torch.einsum( 'ijkl,ijkl -> ikl', 
                                         out, out) )
        for l in range(3):
            out2[:,l,:,:] = out[:,l,:,:]/norm

        loss = mse(out2, tgt)
        loss_m += loss.item() * spin.size(0)
        tot += spin.size(0)

        test_hd.append(out2.clone().detach().cpu())
        test_label.append(tgt.clone().detach().cpu())
        #print("Epoch %d: Loss %.4f"%(epoch, loss))


    #pdb.set_trace()

    test_array=torch.stack(test_hd).reshape(-1,3, img_size, img_size).permute(0, 2, 3, 1)
    test_label_arr=torch.stack(test_label).reshape(-1,3, img_size, img_size).permute(0, 2, 3, 1)
    return loss_m/tot,test_array,test_label_arr




def cfg():
    #所有要修改的数据在这里改
    parse = argparse.ArgumentParser()

    #输入：batch_size
    parse.add_argument('--batch', type=int, default=16)

    #输入：epoch轮次
    parse.add_argument('--epoch_number', type=int, default=100)

    #输入：图片大小
    parse.add_argument('--imgsize', type=int, default=32)

    #输入：多线程内核数
    parse.add_argument('--numworker', type=int, default=2)

    #输入：spin_case
    parse.add_argument('--spin_case', type=int, default=10)

    #输出：loss的数据保存地址
    # parse.add_argument('--excelPath', type=str, default='unet_change/excel/loss_mse_32_case2.xls')

    opt = parse.parse_args()
    return opt                              


if __name__ == '__main__':
    opt = cfg()
    print(opt)

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    torch.use_deterministic_algorithms(True)

    train(opt)