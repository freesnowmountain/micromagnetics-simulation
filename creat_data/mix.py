import numpy as np
case0_input=np.load('npy_data\size_32\case_0\input.npy')
case0_label=np.load('npy_data\size_32\case_0\label.npy')
case1_input=np.load('npy_data\size_32\case_1\input.npy')
case1_label=np.load('npy_data\size_32\case_1\label.npy')
case2_input=np.load('npy_data\size_32\case_2\input.npy')
case2_label=np.load('npy_data\size_32\case_2\label.npy')
case3_input=np.load('npy_data\size_32\case_3\input.npy')
case3_label=np.load('npy_data\size_32\case_3\label.npy')


input_all= np.vstack((case0_input,case1_input,case2_input,case3_input))

label_all= np.vstack((case0_label,case1_label,case2_label,case3_label))

#打乱
np.random.seed(1)
arr = np.arange(2000) # 生成0到itern_num个数
np.random.shuffle(arr) # 随机打乱arr数组
input_all = input_all[arr] # 将input以arr索引重新组合
label_all = label_all[arr] # 将label以arr索引重新组合

a = np.arange(2000) # 生成0到itern_num个数
np.random.shuffle(a) # 随机打乱arr数组
input_all = input_all[a] # 将input以arr索引重新组合
label_all = label_all[a] # 将label以arr索引重新组合
input_all = input_all.astype(np.float32)
label_all = label_all.astype(np.float32)

train_in=input_all[:1600]
train_la=label_all[:1600]
te_in=input_all[1600:]
te_la=label_all[1600:]




np.save(r'D:\Download\project\npy_data\size_32\mixcase\train_input.npy',train_in)
np.save(r'D:\Download\project\npy_data\size_32\mixcase\train_label.npy',train_la)
np.save(r'D:\Download\project\npy_data\size_32\mixcase\test_input.npy',te_in)
np.save(r'D:\Download\project\npy_data\size_32\mixcase\test_label.npy',te_la)