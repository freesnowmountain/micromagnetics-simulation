import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

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
    for epoch in range(100):
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
            # plt.show()
            # plt.close()


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
