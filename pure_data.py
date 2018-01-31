import numpy as np
import scipy.io as sio
GD= sio.loadmat ('D:\\challenge\\Cleaned_Data_Crop\\Genetic_Data.mat')
GD=GD['TData']
GD=np.float32(GD)
num_lable=np.arange(0,GD.shape[0]).reshape(GD.shape[0],1)
GD=np.column_stack((num_lable,GD))
def clean_delete(raw_data, pct_fea=0.3,pct_sam=0.6):
    num_sam =raw_data.shape[0]
    num_fea =raw_data.shape[1]
    print('number of samples:', num_sam)
    print('number of genetic features:', num_fea-1)
    num_miss_sam = np.sum(raw_data == -111, 1)
    num_miss_fea = np.sum(raw_data == -111, 0)
    del_index_fea=np.where(num_miss_fea/num_sam>pct_fea)
    del_index_sam= np.where(num_miss_sam / num_fea> pct_sam)
    print('Delete %i samples'% len(del_index_sam[0]))
    print('Delete %i features'% len(del_index_fea[0]))
    del_data=np.delete(raw_data,del_index_fea,1)
    del_data = np.delete(del_data, del_index_sam,0)
    print('After deletinion, there are %i samples and %i features '% (del_data.shape[0],del_data.shape[1]))
    return del_data
def clean_knn(del_data,pct=0.001):
    knn_data=del_data
    num_k=int(del_data.shape[0]*pct)

    for i in range(del_data.shape[0]):
        print(i)
        index_nn = 1000000 * np.ones((num_k, 2))

        for j in range(del_data.shape[0]):
            dist = 0
            if i!=j:
                for k in range(1,del_data.shape[1]):
                  if (del_data[i,k]==-111) or(del_data[j,k]==-111):
                     dist=dist+4
                  else:
                    dist=dist+np.square((del_data[i,k]-del_data[j,k]))
                    if np.square((del_data[i,k]-del_data[j,k]))>4:
                        print('error')
                index_nn=index_nn[np.argsort(-index_nn[:,1])]
                np.argsort
                for m in range(num_k):
                    if dist<index_nn[m,1]:
                        index_nn[m,1]=dist
                        index_nn[m,0]=j
                        break
        print(index_nn)
        index_nn=index_nn[:,0].astype(int)
        for l in range(1,del_data.shape[1]):
            data_inn= del_data[index_nn,l]
            index_aver=np.where(data_inn!=-111)
            knn_data[i,l]=np.mean(data_inn[index_aver])
    return knn_data




def clean_aver(del_data):
    for i in range(del_data.shape[1]):
        if i!=0:
            index_miss=np.where(del_data[:,i]==-111)
            index_aver=np.where(del_data[:,i]!=-111)
            del_data[index_miss,i]=np.mean(del_data[index_aver,i])
    print('max value:',np.max(del_data[:,1:]),'min value:',np.min(del_data))
    return del_data
A=clean_delete(GD,0.3,0.6)
B=clean_aver(A)
C=clean_knn(A[1:100,:] ,pct=0.05)