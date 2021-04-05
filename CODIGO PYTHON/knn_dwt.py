# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:05:05 2019

@author: ediberto
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix,confusion_matrix, accuracy_score,recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import statistics 
import os
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.robust.scale as robust
import scipy.stats as sc
import pywt
import scipy.signal as signal
from sklearn import preprocessing

#%%
DataBase= np.load('C:/10khz-10s_filtrado/corrente3.npy')
DataBase1= np.load('C:/10khz-10s_filtrado/corrente1.npy')
DataBase2= np.load('C:/10khz-10s_filtrado/corrente2.npy')

Datateste= np.load('C:/database_atrib/normal/corrente3.npy')
Datateste1= np.load('C:/database_atrib/normal/corrente1.npy')
Datateste2= np.load('C:/database_atrib/normal/corrente2.npy')

Datatestef= np.load('C:/database_atrib/falha/corrente3.npy')
Datatestef1= np.load('C:/database_atrib/falha/corrente1.npy')
Datatestef2= np.load('C:/database_atrib/falha/corrente2.npy')

DataBase=np.concatenate((DataBase,DataBase1,DataBase2),axis=0)
Datateste=np.concatenate((Datateste,Datateste1,Datateste2),axis=0)
Datatestef=np.concatenate((Datatestef,Datatestef1,Datatestef2),axis=0)
E2 = pd.DataFrame(Datateste)
E = pd.DataFrame(DataBase)
E3 = pd.DataFrame(Datatestef)
E.columns =[              'DesvioD1','DesvioD2','DesvioD3','DesvioD4','DesvioD5','DesvioA5',
                          'MeanAD1','MeanAD2','MeanAD3','MeanAD4','MeanAD5','MeanADA5',
                          'MedianAD1','MedianAD2','MedianAD3','MedianAD4','MedianAD5','MedianADA5',
                          'KurtosisD1','KurtosisD2','KurtosisD3','KurtosisD4','KurtosisD5','KurtosisA5',
                          'SkewnessD1','SkewnessD2','SkewnessD3','SkewnessD4','SkewnessD5','SkewnessA5',
                          'EnergiaD1','EnergiaD2','EnergiaD3','EnergiaD4','EnergiaD5','EnergiaA5',
                          'frequencia','classe','carga']

E2.columns =[              'DesvioD1','DesvioD2','DesvioD3','DesvioD4','DesvioD5','DesvioA5',
                          'MeanAD1','MeanAD2','MeanAD3','MeanAD4','MeanAD5','MeanADA5',
                          'MedianAD1','MedianAD2','MedianAD3','MedianAD4','MedianAD5','MedianADA5',
                          'KurtosisD1','KurtosisD2','KurtosisD3','KurtosisD4','KurtosisD5','KurtosisA5',
                          'SkewnessD1','SkewnessD2','SkewnessD3','SkewnessD4','SkewnessD5','SkewnessA5',
                          'EnergiaD1','EnergiaD2','EnergiaD3','EnergiaD4','EnergiaD5','EnergiaA5',
                          'frequencia','classe','carga']
E3.columns =[              'DesvioD1','DesvioD2','DesvioD3','DesvioD4','DesvioD5','DesvioA5',
                          'MeanAD1','MeanAD2','MeanAD3','MeanAD4','MeanAD5','MeanADA5',
                          'MedianAD1','MedianAD2','MedianAD3','MedianAD4','MedianAD5','MedianADA5',
                          'KurtosisD1','KurtosisD2','KurtosisD3','KurtosisD4','KurtosisD5','KurtosisA5',
                          'SkewnessD1','SkewnessD2','SkewnessD3','SkewnessD4','SkewnessD5','SkewnessA5',
                          'EnergiaD1','EnergiaD2','EnergiaD3','EnergiaD4','EnergiaD5','EnergiaA5',
                          'frequencia','classe','carga']
E=E.fillna(0)       
E2=E2.fillna(0) 
E3=E3.fillna(0) 

Media=pd.DataFrame()
Media['MeanAD3']=[E.MeanAD3.mean()]
Media['MeanAD4']=[E.MeanAD4.mean()]
Media['MeanAD5']=[E.MeanAD5.mean()]
Media['MedianAD3']=[E.MedianAD3.mean()]
Media['MedianAD4']=[E.MedianAD4.mean()]
        #X['MedianAD5']=teste[1].MedianAD5
Media['MedianADA5']=[E.MedianADA5.mean()]
Media['KurtosisD5']=[E.KurtosisD5.mean()]
Media['KurtosisA5']=[E.KurtosisA5.mean()]
Media['EnergiaD3']=[E.EnergiaD3.mean()]
Media['EnergiaD4']=[E.EnergiaD4.mean()]
Media['EnergiaD5']=[E.EnergiaD5.mean()]
Media['EnergiaA5']=[E.EnergiaA5.mean()]
Media['SkewnessD4']=[E.SkewnessD4.mean()]
Media['frequencia']=[E.frequencia.mean()]

def std(amt):
    Desvio=pd.DataFrame()
    Desvio['MeanAD3']=[amt.MeanAD3.std()]
    Desvio['MeanAD4']=[amt.MeanAD4.std()]
    Desvio['MeanAD5']=[amt.MeanAD5.std()]
    Desvio['MedianAD3']=[amt.MedianAD3.std()]
    Desvio['MedianAD4']=[amt.MedianAD4.std()]
            #X['MedianAD5']=teste[1].MedianAD5
    Desvio['MedianADA5']=[amt.MedianADA5.std()]
    Desvio['KurtosisD5']=[amt.KurtosisD5.std()]
    Desvio['KurtosisA5']=[amt.KurtosisA5.std()]
    Desvio['EnergiaD3']=[amt.EnergiaD3.std()]
    Desvio['EnergiaD4']=[amt.EnergiaD4.std()]
    Desvio['EnergiaD5']=[amt.EnergiaD5.std()]
    Desvio['EnergiaA5']=[amt.EnergiaA5.std()]
    Desvio['SkewnessD4']=[amt.SkewnessD4.std()]
    Desvio['frequencia']=[amt.frequencia.std()]
    
    return Desvio
def split_teste():
    g= E2.groupby(['classe','frequencia','carga'])
    X_test=pd.DataFrame()
    y_test=pd.DataFrame()
    for teste in g:
        #A=np.zeros((len(g),14),float)
        
        X=pd.DataFrame()
        X=pd.DataFrame()
        #X['MeanAD3']=teste[1].MeanAD3
        X['MeanAD4']=teste[1].MeanAD4
        #X['MeanAD5']=teste[1].MeanAD5
        X['MeanADA5']=teste[1].MeanADA5
        X['MedianAD3']=teste[1].MedianAD3
        X['MedianAD4']=teste[1].MedianAD4
        
        X['MedianAD5']=teste[1].MedianAD5
        X['MedianADA5']=teste[1].MedianADA5
        X['KurtosisD5']=teste[1].KurtosisD5
        X['KurtosisA5']=teste[1].KurtosisA5
        X['EnergiaD3']=teste[1].EnergiaD3
        X['EnergiaD4']=teste[1].EnergiaD4
        X['EnergiaD5']=teste[1].EnergiaD5
        X['EnergiaA5']=teste[1].EnergiaA5
        X['EnergiaD2']=teste[1].EnergiaD2

        X['SkewnessD4']=teste[1].SkewnessD4
        #X['frequencia']=teste[1].frequencia
        y=pd.DataFrame(teste[1].classe)
        y.classe=0
        
        X_test=X
        y_test=y
        
    return  X_test, y_test
def split_teste2():
    g= E3.groupby(['classe','frequencia','carga'])
    X_test=pd.DataFrame()
    y_test=pd.DataFrame()
    for teste in g:
        #A=np.zeros((len(g),14),float)
        
        X=pd.DataFrame()
        X=pd.DataFrame()
        #X['MeanAD3']=teste[1].MeanAD3
        X['MeanAD4']=teste[1].MeanAD4
        #X['MeanAD5']=teste[1].MeanAD5
        X['MeanADA5']=teste[1].MeanADA5
        X['MedianAD3']=teste[1].MedianAD3
        X['MedianAD4']=teste[1].MedianAD4
        
        X['MedianAD5']=teste[1].MedianAD5
        X['MedianADA5']=teste[1].MedianADA5
        X['KurtosisD5']=teste[1].KurtosisD5
        X['KurtosisA5']=teste[1].KurtosisA5
        X['EnergiaD3']=teste[1].EnergiaD3
        X['EnergiaD4']=teste[1].EnergiaD4
        X['EnergiaD5']=teste[1].EnergiaD5
        X['EnergiaA5']=teste[1].EnergiaA5
        X['EnergiaD2']=teste[1].EnergiaD2

        X['SkewnessD4']=teste[1].SkewnessD4
        #X['frequencia']=teste[1].frequencia
        y=pd.DataFrame(teste[1].classe)
        y.classe=6
        
        X_test=X
        y_test=y
        
    return  X_test, y_test

def split_corrent(nteste,ntrain,cl):
    
    g= E.groupby(['classe','frequencia','carga'])
    X_train=pd.DataFrame()
    X_test=pd.DataFrame()
    y_train=pd.DataFrame()
    y_test=pd.DataFrame()
    for teste in g:
        #A=np.zeros((len(g),14),float)
        
        X=pd.DataFrame()
        #X['MeanAD3']=teste[1].MeanAD3
        X['MeanAD4']=teste[1].MeanAD4
        #X['MeanAD5']=teste[1].MeanAD5
        X['MeanADA5']=teste[1].MeanADA5
        X['MedianAD3']=teste[1].MedianAD3
        X['MedianAD4']=teste[1].MedianAD4
        
        X['MedianAD5']=teste[1].MedianAD5
        X['MedianADA5']=teste[1].MedianADA5
        X['KurtosisD5']=teste[1].KurtosisD5
        X['KurtosisA5']=teste[1].KurtosisA5
        X['EnergiaD3']=teste[1].EnergiaD3
        X['EnergiaD4']=teste[1].EnergiaD4
        X['EnergiaD5']=teste[1].EnergiaD5
        X['EnergiaA5']=teste[1].EnergiaA5
        X['EnergiaD2']=teste[1].EnergiaD2

        X['SkewnessD4']=teste[1].SkewnessD4
        #X['frequencia']=teste[1].frequencia
        y=pd.DataFrame(teste[1].classe)
        
        clss=int(y.iat[0,0])
        '''
        if(clss== 1 or clss== 2 or clss== 3 or clss== 4 or clss== 5 or clss== 6):
            y.classe=1
        #if(clss== 4 or clss== 5 or clss== 6):
          # y.classe=2
        if(cl==0) : 
            if(clss==0):
            
        #if(clss==0 or clss==6):
                X_trainAux, X_testAux, y_trainAux, y_testAux = train_test_split(X, y, test_size=nteste,train_size=ntrain)
                X_train=X_train.append(X_trainAux)
                X_test=X_test.append(X_testAux)
                y_train=y_train.append(y_trainAux)
                y_test=y_test.append(y_testAux)
        if(cl==1) : 
            if(  clss==2 or clss ==3 ):
            
        #if(clss==0 or clss==6):
                X_trainAux, X_testAux, y_trainAux, y_testAux = train_test_split(X, y, test_size=nteste,train_size=ntrain)
                X_train=X_train.append(X_trainAux)
                X_test=X_test.append(X_testAux)
                y_train=y_train.append(y_trainAux)
                y_test=y_test.append(y_testAux)
        if(cl==2) : 
            if(clss==4 or clss==5 or clss ==6):
            
        #if(clss==0 or clss==6):
                X_trainAux, X_testAux, y_trainAux, y_testAux = train_test_split(X, y, test_size=nteste,train_size=ntrain)
                X_train=X_train.append(X_trainAux)
                X_test=X_test.append(X_testAux)
                y_train=y_train.append(y_trainAux)
                y_test=y_test.append(y_testAux)
        if(cl==-1) : 
            
        #if(clss==0 or clss==6):
                X_trainAux, X_testAux, y_trainAux, y_testAux = train_test_split(X, y, test_size=nteste,train_size=ntrain)
                X_train=X_train.append(X_trainAux)
                X_test=X_test.append(X_testAux)
                y_train=y_train.append(y_trainAux)
                y_test=y_test.append(y_testAux) 
   # print(teste)
    '''
        X_trainAux, X_testAux, y_trainAux, y_testAux = train_test_split(X, y, test_size=nteste,train_size=ntrain)
        X_train=X_train.append(X_trainAux)
        X_test=X_test.append(X_testAux)
        y_train=y_train.append(y_trainAux)
        y_test=y_test.append(y_testAux)
           
    return X_train, X_test, y_train, y_test


#aac=X.Index(0)

def binary(ypred,yteste):
    #ypred=ypred.to_numpy()
    for  x in range(len(ypred)): 
        '''   
        if (ypred[x]==1) :
           ypred[x]=0 
          '''
        if (ypred[x]==2 or ypred[x]==3 or ypred[x]==4 or ypred[x]== 5 or ypred[x]==6) :
           ypred[x]=1 
          
    yteste=yteste.astype(int)       
    for  x in range(len(yteste)):
        '''
        if (yteste[x]==1) :
           yteste[x]=0 
          '''
        if( yteste[x]==2 or yteste[x]==3 or yteste[x]==4 or yteste[x]== 5 or yteste[x]==6) :
           yteste[x]=1   
            
    return yteste,ypred

mean_acc = []
mean_sen = []



target_names = ['normal', 'falha']
#classifier.predict(X)
#%%
for i in range(1, 20):
    
    
    print("rodada ->", i)
    #alterar para numeros em vez de %
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=504,train_size=200)
    X_train, X_test, y_train, y_test = split_corrent(9,30,0)

   #DataBase[i,13]=[X_train]
    #teste = std(X_train)
    '''
    X_trainAux, X_testAux, y_trainAux, y_testAux = split_corrent(9,30,1)
    X_train=X_train.append(X_trainAux)
    X_test=X_test.append(X_testAux)
    y_train= y_train.append(y_trainAux)
    y_test=y_test.append(y_testAux)
    X_trainAux, X_testAux, y_trainAux, y_testAux = split_corrent(9,30,2)
    X_train=X_train.append(X_trainAux)
    X_test=X_test.append(X_testAux)
    y_train= y_train.append(y_trainAux)
    y_test=y_test.append(y_testAux)
    '''
    #para teste de novos dados
     
    X_test1,y_test1 = split_teste()
    X_test1a,y_test1a = split_teste2()
    X_test1=X_test1.append(X_test1a)
    y_test1=y_test1.append(y_test1a)
    #precisam ser vetores para adicionar todos os grupos
    #print(X_train)
    
    
    #normalizaçao
    #X_train=normalize(X_train)
    #X_test=normalize(X_test)
    #Xt = X_train.iloc[:, :-1].values
    #Yt = y_train.iloc[:, 0].values
    
    scaler = preprocessing.MinMaxScaler()
    #scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(X_test1)
    X_test = scaler.transform(X_test1)
    y_train=np.ravel(y_train)
    y_test=np.ravel(y_test1)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred,labels=[0,1,2,3,4,5,6]))
   # print(classification_report(y_test, y_pred,target_names=target_names))
    #print(classification_report(y_test1, y_pred))

    #sen=recall_score(y_test, y_pred,average='macro')
    #print(sen)
    #print(confusion_matrix(y_test, y_pred))
    acc=accuracy_score(y_test, y_pred)
    #sen=accuracy_score(y_test, y_pred)
    sen=recall_score(y_test, y_pred,average='macro')
    print("Acuracia 7 classes ", acc)
    
    print("sensibilidade " , sen)
     
    print("matriz de confusao binarizada")
    y_test1, y_pred=binary(y_test, y_pred)
    print(confusion_matrix(y_test1, y_pred,labels=[0,1]))
    accb=accuracy_score(y_test1, y_pred)
    senb=recall_score(y_test1, y_pred,average='macro')
    print("Acuracia binarizada " , accb)
    print("sensibilidade binarizada " , senb)
    
    mean_acc.append(accb)
    mean_sen.append(senb)
print("")
print("A accuracia media binariza é :" , statistics.mean(mean_acc))
print("desvio padrao da  accuracia media binariza é :" , statistics.pstdev(mean_acc))

print("A sensibilidade media binariza é :" , statistics.mean(mean_sen))
print("desvio padrao da  sensibilidade media binariza é :" , statistics.pstdev(mean_sen))

#%%