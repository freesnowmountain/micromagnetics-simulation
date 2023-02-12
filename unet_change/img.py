import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from tqdm import tqdm

def npy2jpg(npy,img_path):

    #输入为hd，case=1
    case=2
    if case==1:
        #当输入为hd时Hd归一化到RGB方法:(Hd + 12566) / 25132 *255
        #其中12566=4*PI*Ms，Ms，代码里默认值用的1000
        npy=(npy + 12566) / 25132 * 255
    #其它情况
    if case==2:
        npy=(npy-np.min(npy))/(np.max(npy)-np.min(npy))*255


    count=0
    for con in npy:
        arr=npy[count]
        arr=np.transpose(arr,(2,1,0))
        r=Image.fromarray(arr[0]).convert("L")
        g=Image.fromarray(arr[1]).convert("L")
        b=Image.fromarray(arr[2]).convert("L")
        #合并三通道
        img=Image.merge("RGB",(r,g,b))
        img.save(img_path+str(count)+".png")
        count=count+1



def plot_histogram(arr,head_path):
    for epoch in range(200):
        for x in range(3):
            n = arr[epoch][x]
            x_lever=arr[epoch]
            #返回折叠后的一维数组
            x0 = np.array(n).flatten()
            #histogram
            #求平均值
            x_m=np.array(x0).mean(0)
            #求方差
            x_n=np.var(x0)
            #x轴坐标名
            ddd = 'mean value = '+str(x_m)+'variance = '+str(x_n)
            plt.xlabel(ddd)
            plt.ylabel('Frequency')
            #plt.xlim(-1000, 800)
            #x坐标范围
            plt.xlim(x_lever.min()-0.35, x_lever.max()+0.35)

            params = dict(histtype='stepfilled', alpha=0.5, density=True, bins=50)
            plt.hist(x0,**params)

            #plt.legend(['simulate data training epoch={} \n \n SB={:.3f} \n SW.mean={:.3f} \n class_var_var={:.3f}'.format(epoch, SB, SW_m, class_var_var)])
            
            if not os.path.exists(head_path):
                os.mkdir(head_path)

            # dir=head_path+''+str(epoch)+'channel:'+str(x) +'.png'
            dir=head_path+'epoch_'+str(epoch)+'_channel_'+str(x) +'.png'
            plt.savefig(dir)


def relitu(output,label,img_path):
    cos=np.zeros((len(output),len(output[0]),len(output[0][0])))
    #1600*32*32*3

    time1=time.time()
    for i in range(len(output)):#1600
        for j in range(len(output[0])):#32
            for k in range(len(output[0][0])):#32
                #print(output[i][j][k].dot(label[i][j][k])/(np.linalg.norm(output[i][j][k])*np.linalg.norm(label[i][j][k])))
                cos[i][j][k]=output[i][j][k].dot(label[i][j][k])/(np.linalg.norm(output[i][j][k])*np.linalg.norm(label[i][j][k]))
                #
                cos[i][j][k]=np.arccos(cos[i][j][k])
                cos[i][j][k]=cos[i][j][k]*180/np.pi



    for _,i in enumerate(tqdm(range(50))):
        uniform_data=cos[i]
        sns.set()
        plt.rcParams['font.sans-serif']='SimHei'#设置中文显示，必须放在sns.set之后
        f, ax = plt.subplots(figsize=(9, 6))

        #heatmap后第一个参数是显示值,vmin和vmax可设置右侧刻度条的范围,
        #参数annot=True表示在对应模块中注释值
        # 参数linewidths是控制网格间间隔
        #参数cbar是否显示右侧颜色条，默认显示，设置为None时不显示
        #参数cmap可调控热图颜色，具体颜色种类参考：https://blog.csdn.net/ztf312/article/details/102474190

        sns.heatmap(uniform_data, ax=ax,vmin=0,vmax=180,cmap='YlOrRd',annot=False,linewidths=2,cbar=True)

        #求平均值
        x_m=np.array(uniform_data).mean()
        #求方差
        x_n=np.var(uniform_data)
        #x轴坐标名
        ddd = 'mean value = '+str(x_m)+'variance = '+str(x_n)
        plt.xlabel(ddd)    

        ax.set_title('heatmap') #plt.title('热图'),均可设置图片标题
        #ax.set_ylabel('y')  #设置纵轴标签
        ax.set_ylabel('Y-axis (degrees)')


        ax.set_xlabel(ddd)  #设置横轴标签

        #设置坐标字体方向，通过rotation参数可以调节旋转角度
        #
        label_y = ax.get_yticklabels()
        plt.setp(label_y, rotation=360, horizontalalignment='right')
        label_x = ax.get_xticklabels()
        plt.setp(label_x, rotation=45, horizontalalignment='right')

        
        # plt.show()
        imgout_path=img_path+'relitu_01'
        if not os.path.exists(imgout_path):
            os.makedirs(imgout_path)  
        dir= imgout_path +'\\'+ str(i) +'.png'
        plt.savefig(dir)
        

def vector(arr,head_path):
    #传入四维数组，(number,img_size,img_size,3)
     
    if not os.path.exists(head_path):
            os.makedirs(head_path)  
    arr=arr.transpose(0,2,1,3)
    fig = plt.figure(dpi=1000)
    for i in range(50):
        
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        
        X = np.arange(32) 
        Y = np.arange(32) 
        ax.cla()
        ax.quiver( X, Y, arr[i,:,:,0], arr[i,:,:,1], arr[i,:,:,2],
                  clim=[np.min(arr), np.max(arr)] )
        ax.set_title('Vector graphics')
        path=head_path+str(i)+".png"
        fig.savefig(path)
        matplotlib.pyplot.close()
        # plt.pause(0.8)

if __name__ == "__main__":
    arr_np_path=r"output\size_32\mixcase\test\label.npy"
    head_path=r"output\size_32\mixcase\test\exam\\"
    # arr_np_path=r"output\size_32\mixcase\train\output.npy"
    # head_path=r"output\size_32\mixcase\train\output_vector_img\\"
    arr_np=np.load(arr_np_path)
    vector(arr_np,head_path)
