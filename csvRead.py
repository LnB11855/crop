import pandas as pd
import glob
import os
import numpy as np
PATH = r'D:\challenge\review\crop'
EXT = "*.csv"
minist_ga_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob.glob(path + "/*.csv")]
rawData= []
trainFinal=[]
testAll=[]
for filename in minist_ga_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    if "ga" in filename:
        a=df['test_acc'].max()
        testAll.append(a)
    else:
        a = df['test_acc'][0:800].max()
        testAll.append(a)
    rawData.append(df)
df_all = pd.DataFrame({'name':minist_ga_files,'testAcc':testAll})
df_all=df_all.sort_values(by='testAcc', ascending=False)


path1 = r'D:\challenge\review\crop\minist_keras_ga_non_break2_60000_global'
path2 = r'D:\challenge\review\crop\minist_keras_ga_non_break_60000'
path3 = r'D:\challenge\review\crop\minist_keras_ga_60000'
path4 = r'D:\challenge\review\crop\minist_keras_ga_non_break2_60000_global'
path5 = r'D:\challenge\review\crop\minist_keras'
minist_ga_files= glob.glob(path1 + "/*.csv")
rawData= []
trainFinal=[]
testAll=[]
for filename in minist_ga_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    a=df['test_acc'].max()
    testAll.append(a)
    rawData.append(df)
tuneMaxIndex=np.argmax(testAll)
print("The max test accuracy file is ",minist_ga_files[tuneMaxIndex] )
print("The max test accuracy is ",testAll[tuneMaxIndex] )
