#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:38:01 2022

"""
import MAG2305_torch
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt
import time
import torch
import os
import xlwt
#################
# Prepare model #
#################

img_size=32#图片大小



# Iteration number
itern_num = 1000
#spin初始化情况
spin_case = 0

npy_path="npy_data/size_"+str(img_size)+"/case_"+str(spin_case)

if not os.path.exists(npy_path):
    os.makedirs(npy_path)

#input_path
input_path=npy_path + '/input.npy'
#label_path
label_path=npy_path + '/label.npy'


#spin为四维数组(img_size,img_size,1,3)，使用五维数组储存数据
arr_number=500
Spin_arr=np.zeros((arr_number,img_size,img_size,1,3))
Spin_label=np.zeros((arr_number,img_size,img_size,1,3))

# Test-model size

test_size = (img_size,img_size,1)
cell_num  = test_size[0] * test_size[1] * test_size[2]
# Test-cell size 
test_cell=(4,4,1)
# Saturation magnetization
Ms = 1000

# Create a test-model
time_start = time.time()
film0 = MAG2305_torch.mmModel(types='bulk', size=test_size, cell=test_cell, Ms=Ms, device='cuda')
time_finish = time.time()
print('Time cost: {:f} s for creating model \n'.format(time_finish-time_start))

# Initialize demag matrix
time_start = time.time()
film0.DemagInit()
time_finish = time.time()
print('Time cost: {:f} s for getting demag matrix of the whole model \n'.format(time_finish-time_start))
 
# Initialize spin state
spin = np.empty( tuple(film0.size) + (3,) )#初始输入spin （256，256，3）

if   spin_case == 0:
    for ijk in np.ndindex(tuple(film0.size)):
        spin[ijk] = [0.0, 1.0, 0.0]
        
elif spin_case == 1:
    for ijk in np.ndindex(tuple(film0.size)):
        spin[ijk] = [0.0, 1.0, 0.0] if ijk[1] < film0.size[1]//2 else [0.0, -1.0, 0.0]
        
elif spin_case == 2:
    for ijk in np.ndindex(tuple(film0.size)):
        spin[ijk]  = [0.0, 1.0, 0.0] if ijk[1] < film0.size[1]//2 else [0.0, -1.0, 0.0]
        spin[ijk] += [1.0, 0.0, 0.0] if ijk[0] > film0.size[0]//2 else [-1.0, 0.0, 0.0]

elif spin_case == 3:
    for ijk in np.ndindex(tuple(film0.size)):
        if ijk[0] + ijk[1] > (film0.size[0] + film0.size[1])/2:
            if ijk[0] - ijk[1] < (film0.size[0] - film0.size[1])/2:
                spin[ijk] = [0.0,-1.0, 0.0]
            else:
                spin[ijk] = [1.0, 0.0, 0.0]
        else:
            if ijk[0] - ijk[1] < (film0.size[0] - film0.size[1])/2:
                spin[ijk] = [-1.0, 0.0, 0.0]
            else:
                spin[ijk] = [0.0, 1.0, 0.0]

for ijk in np.ndindex(tuple(film0.size)):
    spin[ijk,0] += np.random.uniform(-0.01,0.01)
    spin[ijk,1] += np.random.uniform(-0.01,0.01)
    #spin[ijk,2] += np.random.uniform(-0.01,0.01)
    


time_start = time.time()
stat,spin_zero=film0.SpinInit(spin)

# Spin_train[0]=spin_zero.cpu().detach().numpy()#保存第一个spin

#保存第一个spin
Spin_arr[0]=spin_zero.cpu().detach().numpy()

time_finish = time.time()
print('Time cost: {:f} s for getting input spin \n'.format(time_finish-time_start)) 


#####################
# Update spin state #
#####################
print('\nBegin spin updating\n')

# Pseudo time step
dtime = 1.0e-5

# Convergence control
error_min = 1.0e-5


# External field
Hext_val = 0.0
Hext_vec = np.array([1.0, 0.0, 0.0])
Hext = Hext_val * Hext_vec


#Create a canvas
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal', adjustable='box')
# X = np.arange(film0.size[0]) * film0.cell[0]
# Y = np.arange(film0.size[1]) * film0.cell[1]

# #绘画hd
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
# ax1.set_aspect('equal', adjustable='box')
# M = np.arange(film0.size[0]) * film0.cell[0]
# N = np.arange(film0.size[1]) * film0.cell[1]

# Do iteration
spin_sum  = np.zeros(3)
error_all = np.zeros(tuple(film0.size))

error_img=[]

#用于下面数组下标计数
t=0

for itern in range(itern_num):
    
    time_start = time.time()
    Hd = film0.DemagField_FFT()#第一个输出
 
    time_finish = time.time()
    print('  Hd    time cost: {:f} s '.format(time_finish-time_start))
    spin, error = film0.SpinDescent(Hext=Hext, dtime=dtime)#循环进行第一次时，这里对应第二个spin
    
    error_img.append(error)
    
    
    #error曲线
    # workbook=xlwt.Workbook("error")
    # sheet=workbook.add_sheet("loss+mse")
    # col=["Epoch","loss","mse"]
    # for i in range(len(col)):
    #     sheet.write(0,i,col[i])
    # workbook.save(excel_path)

    #0到248,共249个数据，加上之前初始化，共250个
    if itern<249:
        Spin_arr[itern+1]=spin
    #从itern=252开始

    if itern>=249 and itern%3==1:
        Spin_arr[250+t]=spin
        t=t+1


    

    time_finish = time.time()
    print('  Other time cost: {:f} s '.format(time_finish-time_start))
    
    for l in range(3):
        spin_sum[l] = spin[...,l].reshape(-1).sum()
    spin_sum = spin_sum / cell_num


    if itern==itern_num-1:
        #绘制error的曲线
        # fig2=plt.subplot()
        # plt.xlim(xmin=0,xmax=500)
        # plt.plot(error_img)
        # plt.show()
        # try:
        #     import os
        #     os.makedirs('output')
        # except FileExistsError:
        #     pass
        # plt.savefig('output/case_{}error.png'.format(spin_case))        

        #匹配
        for i in range(arr_number):
            if i==arr_number-1:
                Spin_arr[i]=Spin_arr[0]
                Spin_label[i]=Spin_arr[1]
            else:
                Spin_label[i]=Spin_arr[i+1]
  
        #转变类型
        input_arr = Spin_arr.astype(np.float32)
        input_arr = np.squeeze(Spin_arr,3)
        label_arr = Spin_label.astype(np.float32)
        label_arr = np.squeeze(Spin_label,3)
        #打乱
        np.random.seed(0)
        arr = np.arange(arr_number) # 生成0到itern_num个数
        np.random.shuffle(arr) # 随机打乱arr数组
        Spin_arr = Spin_arr[arr] # 将input以arr索引重新组合
        Spin_label = Spin_label[arr] # 将label以arr索引重新组合
        #保存 
        np.save(input_path,input_arr)
        np.save(label_path,label_arr)



    if itern%10==0 or error<=error_min :
        print('  M_avg={},  error={:.8f}\n'.format(spin_sum, error))
        # ax.cla()
        # ax.quiver( X, Y, spin[:,:,0,0], spin[:,:,0,1], 
        #            spin[:,:,0,2], clim=[-0.5,0.5] )
        # ax.set_title('Iteration {:d}'.format(itern))
        # path="image_data/v_spin_32_case3/"+str(itern)+".png"
        # fig.savefig(path)
        # plt.pause(0.8)
        
        # #绘画hd
        # ax1.cla()
        # ax1.quiver( M, N, Hd[:,:,0,0], Hd[:,:,0,1], 
        #            Hd[:,:,0,2], clim=[-0.5,0.5] )
        # ax1.set_title('Iteration {:d}'.format(itern))
        # path="image_data/v_hd_32_case3/"+str(itern)+".png"
        # fig1.savefig(path)
        # plt.pause(0.8)

        if error<=error_min :
            break

        