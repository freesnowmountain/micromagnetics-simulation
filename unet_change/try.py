# import numpy as np
# case0_input=np.load('npy_data\size_32\case_0\input_all.npy')
# case0_label=np.load('npy_data\size_32\case_0\label_all.npy')
# case1_input=np.load('npy_data\size_32\case_1\input_all.npy')
# case1_label=np.load('npy_data\size_32\case_1\label_all.npy')
# case2_input=np.load('npy_data\size_32\case_2\input_all.npy')
# case2_label=np.load('npy_data\size_32\case_2\label_all.npy')
# case3_input=np.load('npy_data\size_32\case_3\input_all.npy')
# case3_label=np.load('npy_data\size_32\case_3\label_all.npy')


# input_all= np.vstack((case0_input,case1_input,case2_input,case3_input))

# label_all= np.vstack((case0_label,case1_label,case2_label,case3_label))

# #打乱
# np.random.seed(0)
# arr = np.arange(2000) # 生成0到itern_num个数
# np.random.shuffle(arr) # 随机打乱arr数组
# input_all = input_all[arr] # 将input以arr索引重新组合
# label_all = label_all[arr] # 将label以arr索引重新组合

# a = np.arange(2000) # 生成0到itern_num个数
# np.random.shuffle(a) # 随机打乱arr数组
# input_all = input_all[a] # 将input以arr索引重新组合
# label_all = label_all[a] # 将label以arr索引重新组合

# train_in=input_all[:1600]
# train_la=label_all[:1600]
# te_in=input_all[1600:]
# te_la=label_all[1600:]




# np.save(r'D:\Download\project\npy_data\size_32\train_input.npy',train_in)
# np.save(r'D:\Download\project\npy_data\size_32\train_label.npy',train_la)
# np.save(r'D:\Download\project\npy_data\size_32\test_input.npy',te_in)
# np.save(r'D:\Download\project\npy_data\size_32\test_label.npy',te_la)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

#train的output
arr=np.load('')
head_path=''
def vector(arr,head_path):
    #传入四维数组，(number,img_size,img_size,3)
    for i in range(100):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        
        X = np.arange(32) 
        Y = np.arange(32) 
        ax.cla()
        ax.quiver( X, Y, arr[:,:,0,0], arr[:,:,0,1], 
                arr[:,:,0,2], clim=[-0.5,0.5] )
        ax.set_title('Vector graphics')
        path=head_path+str(i)+".png"
        fig.savefig(path)
        # plt.pause(0.8)