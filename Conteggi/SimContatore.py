import numpy as npy
import math
import sys
import random

#set global random seed
#random_Gseed = 1111
#npy.random.seed(random_Gseed)

class SimContatore:
    
  def __init__(self):
    #Parametri fisici    
    self.rA = int('0b110',2)*10**-1 #rate ambiente 
    self.rC = int('0b100',2)*10**-1 #rate cosmici  
    self.rS = int('0b1000',2)*10**-1 #rate sorgente  
    #Caratteristiche strumento
    self.res = 0.0001 #risoluzione temporale geiger [s]

  def MisuraConteggi(self,T,n_misure,mode):
    T = round(T,1)
    
    if T<1 or T>600:
        print ("La durata di una singola presa dati deve essere un numero tra 1s e 600s")
        sys.exit()  

    n = int(T / self.res)

    r = 0
    if mode=="O":
        r = self.rA + self.rC
        p = r*self.res
    elif mode=="V":
        r = self.rA + self.rC/2
    elif mode=="S":
        r = self.rA + self.rC + self.rS
    else:
        print ("Opzione errata! Usa O (orizzontale), V (verticale), S (orizzontale+sorgente)")
        sys.exit()
    
    p = r*self.res
    
    counts_list = []
    time_counts_list = []

    for mis in range(1,n_misure+1):
        tmp_time_counts_list = []        
        tmp_count = 0
        for i in range(1,n+1):               
            x = random.random()
            if x<p:
                tmp_count = tmp_count + 1
                hit_time = round(i*self.res,4)
                tmp_time_counts_list.append(hit_time)                
        counts_list.append(tmp_count)
        time_counts_list.append(npy.asarray(tmp_time_counts_list))      

    time_counts_np = npy.asarray(time_counts_list)
    counts_np = npy.asarray(counts_list)    

    print ("{0:d} misure ripetute di conteggio in un intervallo temporale di {1:.1f} s =".format(n_misure,T))   
    print (counts_np)
    
    return  counts_np, time_counts_np
