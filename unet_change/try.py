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
import matplotlib
matplotlib.use("Agg")
#实验1：寻找一组input和output看得出明显差异有代表性的数据。
test_input=np.load(r'D:\Download\project\output\size_32\mixcase_new\test\input.npy')
test_label=np.load(r'D:\Download\project\output\size_32\mixcase_new\test\label.npy')
test_output=np.load(r'D:\Download\project\output\size_32\mixcase_new\test\output.npy')

train_input=np.load(r'D:\Download\project\output\size_32\mixcase_new\train\input.npy')
train_label=np.load(r'D:\Download\project\output\size_32\mixcase_new\train\label.npy')
train_output=np.load(r'D:\Download\project\output\size_32\mixcase_new\train\output.npy')
#print(test_output[20][18][16],test_label[20][18][16])#第一维<50，第二维<32,3wei<32,第四维三个数据打印出来
#print(train_output[20][18][16],train_label[20][18][16])

#print(test_output[20][18][16],test_label[20][18][16])
#print(train_output[20][18][16],train_label[20][18][16])




#实验2：将这个有代表性的数据的output和label的numpy数组的32*32*3数据打印出来截图。
# np.set_printoptions(threshold=np.inf)
# print('************************')
# print("print test label:")
# print(test_label)
# print('************************')
# print("print test_input:")
# print(test_input)
# print('************************')
# print("print test_output:")
# print(test_output)

# print('************************')
# print("print train label:")
# print(train_label)
# print('************************')
# print("print train_input:")
# print(train_input)
# print('************************')
# print("print train_output:")
# print(train_output)
# print('over')
# #输出归一化
# os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'    
# norm = torch.sqrt( torch.einsum( 'ijkl,ijkl -> ikl', 
#                                     out, out) )
# for l in range(3):
#     out2[:,l,:,:] = out[:,l,:,:]/norm
# print('************************')
# print(test_input)
# print('************************')
# print(test_output)







img_size=32
#输出文件夹
train_output_path = "output/size_"+str(img_size)+ "/mixcase_new" + '/train/'
test_output_path = "output/size_"+str(img_size)+ "/mixcase_new" + '/test/'
#输出最大最小
print(test_label.min(),test_label.max())
print(test_input.min(),test_input.max())
print(test_output.min(),test_output.max())

#彩色图
train_input_color=train_output_path+'input_color_img/'
train_output_color=train_output_path+'output_color_img/'
train_label_color=train_output_path+'label_color_img/'

test_input_color=test_output_path+'input_color_img/'
test_output_color=test_output_path+'output_color_img/'
test_label_color=test_output_path+'label_color_img/'

#直方图
train_input_zhifang=train_output_path+'input_zhifang_img/'
train_output_zhifang=train_output_path+'output_zhifang_img/'
train_label_zhifang=train_output_path+'label_zhifang_img/'

test_input_zhifang=test_output_path+'input_zhifang_img/'
test_output_zhifang=test_output_path+'output_zhifang_img/'
test_label_zhifang=test_output_path+'label_zhifang_img/'

#relitu
train_reli=train_output_path+'input_reli_img/'
test_reli=test_output_path+'input_reli_img/'



#shiliangtu 
train_input_vector=train_output_path+'input_vector_img/'
train_output_vector=train_output_path+'output_vector_img/'
train_label_vector=train_output_path+'label_vector_img/'

test_input_vector=test_output_path+'input_vector_img/'
test_output_vector=test_output_path+'output_vector_img/'
test_label_vector=test_output_path+'label_vector_img/'

#color

# if not os.path.exists(train_input_color):
#     os.makedirs(train_input_color)  
# if not os.path.exists(train_label_color):
#     os.makedirs(train_label_color)  
# if not os.path.exists(train_output_color):
#     os.makedirs(train_output_color) 

# if not os.path.exists(test_input_color):
#     os.makedirs(test_input_color)  
# if not os.path.exists(test_label_color):
#     os.makedirs(test_label_color)  
# if not os.path.exists(test_output_color):
#     os.makedirs(test_output_color) 

# img.npy2jpg(train_input,train_input_color)
# img.npy2jpg(train_label,train_label_color)
# img.npy2jpg(train_output,train_output_color)

# img.npy2jpg(test_input,test_input_color)
# img.npy2jpg(test_label,test_label_color)
# img.npy2jpg(test_output,test_output_color)

#zhifangtu
# if not os.path.exists(train_input_zhifang):
#     os.makedirs(train_input_zhifang)  
# if not os.path.exists(train_label_zhifang):
#     os.makedirs(train_label_zhifang)  
# if not os.path.exists(train_output_zhifang):
#     os.makedirs(train_output_zhifang) 

# if not os.path.exists(test_input_zhifang):
#     os.makedirs(test_input_zhifang)  
# if not os.path.exists(test_label_zhifang):
#     os.makedirs(test_label_zhifang)  
# if not os.path.exists(test_output_zhifang):
#     os.makedirs(test_output_zhifang) 

# img.plot_histogram(train_input,train_input_zhifang)
# img.plot_histogram(train_label,train_label_zhifang)
# img.plot_histogram(train_output,train_output_zhifang)

# img.plot_histogram(test_input,test_input_zhifang)
# img.plot_histogram(test_label,test_label_zhifang)
# img.plot_histogram(test_output,test_output_zhifang)

#矢量图
# if not os.path.exists(train_input_vector):
#     os.makedirs(train_input_vector)  
# if not os.path.exists(train_label_vector):
#     os.makedirs(train_label_vector)  
# if not os.path.exists(train_output_vector):
#     os.makedirs(train_output_vector) 

# if not os.path.exists(test_input_vector):
#     os.makedirs(test_input_vector)  
# if not os.path.exists(test_label_vector):
#     os.makedirs(test_label_vector)  
# if not os.path.exists(test_output_vector):
#     os.makedirs(test_output_vector) 

# img.vector(train_input,train_input_vector)
# img.vector(train_label,train_label_vector)
# img.vector(train_output,train_output_vector)

# img.vector(test_input,test_input_vector)
# img.vector(test_label,test_label_vector)
# img.vector(test_output,test_output_vector)

# print('over')



#热力图
if not os.path.exists(train_reli):
    os.makedirs(train_reli)  
if not os.path.exists(train_reli):
    os.makedirs(train_reli)  


# img.relitu(train_output,train_label,train_reli)

img.relitu(test_output,test_label,test_reli)
print('ok')


# te_in_p_pp=test_output_path+'output_exam_img/'
# img.plot_histogram(test_label,te_in_p_pp)


# test_label=np.load(r'D:\Download\project\output\size_32\mixcase_new\test\output.npy')
# tr_in_p_p=train_output_path+'input_histogram_img/'
# te_in_p_p=test_output_path+'input_histogram_img/'
# img.plot_histogram(test_input,te_in_p_p)
# img.plot_histogram(train_input,tr_in_p_p)

# tr_in_p=train_output_path+'input_color_img/'
# te_in_p=test_output_path+'input_color_img/'
# if not os.path.exists(tr_in_p):
#     os.makedirs(tr_in_p)  
# if not os.path.exists(te_in_p):
#     os.makedirs(te_in_p)  
    
# img.npy2jpg(train_input,tr_in_p)
# img.npy2jpg(test_input,te_in_p)