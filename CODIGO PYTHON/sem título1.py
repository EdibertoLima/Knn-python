# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:35:12 2019

@author: ediberto
"""

#%%
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles      
######################################################################################

def EnergyPercent(Ac,Dc,Original):
    
    E=np.empty((1,np.size(Dc,1)+1),float)
    Ep=np.zeros((1,np.size(Dc,1)+1),float)
    
    Et=np.sum(np.square(Original))
    
    for i in range(0,np.size(Dc,1)):
        E[0,i]=np.sum(np.square(Dc[0,i]))
        
    E[0,i+1]=np.sum(np.square(Ac))


    Ep=E/Et;
    
    return Ep;

    

#######################################################################################
#Ler amostra do arquivo
path='C:\corrente_10khz_10s_filtrado'
 #Confgurar caminho para pasta contendo as amostras

Files=getListOfFiles(path);  

amostra=np.empty((100000), float);

N_decomp_levels=5 #Configurar níveis de decomposição aqui

n=N_decomp_levels+2

nCols=(n-1)*6+3

DataBase=np.zeros((len(Files),nCols),float);

x_array=[10] #Configurar quantas amostras devem ser "saltadas" para subamostragem



for x_idx in range(0,len(x_array)):

    #for i in range(0, len(Files)):
     for i in range(0, len(Files)):
        fp=open(Files[i])

        content=fp.readlines();
        x = np.array(content[0:])

        for j in range(0,100000):
            #amostra_strings=x[j].split(',');
            amostra[j]=np.array(x[j]);

    #####################################################################################
    
        #  Remover a media
        #mean=np.mean(amostra)    
        #amostra=amostra-mean;    

        x=x_array[x_idx];
        #Filtragem digital
        fs=10000;
        fc =fs/(2*x)-300 ;  # Cut-off frequency of the filter   
        w = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(5, w, 'low')
        output = amostra

        #Subamostragem
        t=10                             # Configurar tempo total em segundos da amostra cortada
        n_pontos=int(t*10000)

        fluxo=output[0:n_pontos:x]
        #print(len(fluxo))

        input=fluxo;
    
        Desvio=np.zeros((1,n-1),float)
        MeanAD=np.zeros((1,n-1),float)
        MedianAD=np.zeros((1,n-1),float)
        Energia=np.zeros((1,n-1),float)
        Kurtosis=np.zeros((1,n-1),float)
        Skewness=np.zeros((1,n-1),float)
        cD=np.empty((1,n-2),object)
    
        for m in range(0,n-2):
            cA, cD[0,m]=pywt.dwt(input,'db2')
            Desvio[0,m]=np.std(cD[0,m])
            MeanAD[0,m]=pd.DataFrame(cD[0,m]).mad()
            MedianAD[0,m]=robust.mad(cD[0,m])
            Kurtosis[0,m]=sc.kurtosis(cD[0,m])
            Skewness[0,m]=sc.skew(cD[0,m])
            input=cA;
        #print(MeanAD)    
        #print(cA)    
        Desvio[0,m+1]=np.std(cA)
        MeanAD[0,m+1]=pd.DataFrame(cA).mad()
        MedianAD[0,m+1]=robust.mad(cA)
        Kurtosis[0,m+1]=sc.kurtosis(cA)
        Skewness[0,m+1]=sc.skew(cA)
        
        Energia=EnergyPercent(cA,cD,fluxo)
        
        sObject=slice(39,41);
       
        Freq=int(Files[i][sObject]);

        sObject=slice(42,44);

        Load=int(Files[i][sObject]);
    
        sObject=slice(31,32);
    
        Classe=int(Files[i][sObject]);
    
        sObject=slice(34,38);
    
        Index=int(Files[i][sObject]);
        
        #print(Freq,Load,Classe,Index)

        print(i)
        Pattern=np.zeros((1,nCols-3),float)
        np.concatenate((Desvio, MeanAD, MedianAD, Kurtosis, Skewness, Energia),out=Pattern,axis=1)
    
        
    
        DataBase[i,0:nCols-3]=Pattern;
    
        DataBase[i,nCols-3:]= [Freq, Classe,Load]
    
        fp.close()