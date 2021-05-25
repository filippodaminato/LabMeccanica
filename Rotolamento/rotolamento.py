import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats
import pandas as pd
import math

import my_lib_santanastasio as my

def PrintResult(name,mean,sigma,digits,unit):
    mean = round(mean,digits)
    sigma = round(sigma,digits)
    nu = sigma / mean
    result = (name+"= {0} ± {1} ".format(mean,sigma)+unit+" [{0:.2f}%]".format(nu*100))
    print (result)
    
# Data = array di tempi
# time = tempo interessato
# ritorna l'indice dell'array di tempi combaciante con il tempo passato
def time_to_index(data,time):
    
    for x in range(1,len(data)):
        if time <= data[x]  and time >= data[x-1]:
            return x
        
    return 0

# data_time = array di tempi
# data gyro = array sensore giroscopio
# time start = tempo [s] di inizio analisi
# time finish = tempo [s] di finie analisi
# with_chi2 = eseguire il test del chi2
# save_fig = "immagine.png" se non None salva immagine con nome file passato
def m_from_fit(data_time ,data_gyro ,time_start ,time_finish, with_chi2=False, with_plot=False,save_fig=None):
    
    fig_name = None
    
    s_index = time_to_index(data_time,time_start)
    f_index = time_to_index(data_time,time_finish)
    
    x = data_time[s_index:f_index]-time_start # sottraggo il tempo di inizio per far combaciare con 0
    y = data_gyro[s_index:f_index]
    
    #ricavo sigma y dato inizialmente uguale a 0
    std_camp = np.std(y,ddof=1) #std campionaria
    
    uy = np.array([ std_camp for i in y])
    
    #eseguire fit
    if save_fig is not None:
        plt.cla()
        fig_name = "1-"+save_fig 
    m0, sm0, c0, sc0, cov0, rho0 = my.lin_fit(x, y, uy, "x [ux]", "y [uy]",  0, x.max()+0.01, 0, y.max()+0.1, plot=False, setrange=True,verbose=False,save_fig=fig_name)
    

    ux = 0
    uy_new = np.sqrt(uy**2+(m0*ux)**2)
    

    
    if save_fig is not None:
        plt.cla()
        fig_name = "nuove-y-"+save_fig 
    m, sm, c, sc, cov, rho = my.lin_fit(x, y, uy_new, "x [ux]", "y [uy]", 0, x.max()+0.01, 0, y.max()+0.1, plot=False, setrange=True,verbose=False,save_fig=fig_name)
    
    y_atteso = m*x + c
    d = y - y_atteso
    d_norm = d / uy_new
    
    if save_fig is not None:
        plt.cla()
        fig_name = "residui-"+save_fig 
        plt.errorbar(x,d,uy_new,marker='.',linestyle="")
        plt.ylabel("Residui $d=y-y_{atteso}$ [uy]")
        plt.xlabel("$x$ [ux]")
        plt.grid()
        plt.savefig(fig_name)
    
    # Studio dei residui        
    if save_fig is not None:
        plt.cla()
        fig_name = "residui-norm-"+save_fig 
        plt.errorbar(x,d_norm,uy_new/uy_new,marker='.',linestyle="")
        plt.ylabel("Residui normalizzati $d/\sigma_y=(y-y_{atteso})/\sigma_y$")
        plt.xlabel("$x$ [ux]")
        plt.grid()
        plt.savefig(fig_name)
    

    
    # Incertezze a posteriori
    sigmy_post = math.sqrt( np.sum(d**2)/(d.size-2) )
    uy_post = np.repeat(sigmy_post,y.size)

    # Nuovo fit con incertezze a posteriori sulle y
    if save_fig is not None:
        plt.cla()
        fig_name = "posteriori-"+save_fig 
    m1, sm1, c1, sc1, cov1, rho1 = my.lin_fit(x, y, uy_post, "x [ux]", "y [uy]", 0, x.max()+0.1, 0, y.max()+1, plot=with_plot, setrange=True,verbose=False,save_fig=fig_name)
    
    if with_chi2:
        chi2(x,y,uy,m,c)
    
    return m1, sm1

def chi2(x,y,uy,m,c):
    
    from scipy.stats import chi2
    residui = y - (m*x+c)
    residuiNorm = residui / uy

    chi2_data = np.sum(residuiNorm**2)
    ndf = len(x) - 2
    redChi2_data = chi2_data / ndf
    print("\n")
    print ("chi2 misurato: ", chi2_data.round(5))
    print ("ndf: ",  ndf)
    print ("chi2 mis./ndf: ",  (chi2_data/ndf).round(5))
    print("\n")
    
    p_value = 1-chi2.cdf(chi2_data, ndf)
    alpha = 0.05
    if p_value>alpha:
        print ("p_value={0} > {1}".format(p_value,alpha))
        print ("Test del Chi2 al livello di significatività alpha={0} superato".format(alpha))
    else:
        print ("p_value={0} < {1}".format(p_value,alpha))
        print ("Test del Chi2 al livello di significatività alpha={0} non superato".format(alpha))
    
    print("\n\n")
    
    
def plot_gyro(data,time):
    plt.plot(time,data)


def plot_gyro_phyphox(file_name, index_time=None):
    df = pd.read_csv(file_name)
    
    y = df['Gyroscope y (rad/s)'].to_numpy()
    x = df['Time (s)'].to_numpy()
    
    if index_time:
        ind_start = time_to_index(x,index_time[0])
        ind_finish = time_to_index(x,index_time[1])
        plt.plot(x[ind_start:ind_finish],y[ind_start:ind_finish])
    else:
        plt.plot(x,y)
    
    
    return plt

# ricavare angolo (in rad) dal dataset di phyphox (in deg) (basta passare il nome del file .csv)
def get_angle_from_data(file_name,nmis,save_fig=False,name=None):
    df_angle = pd.read_csv(file_name)
    

    
    angle = df_angle['Inclination (deg)'].to_numpy()
    time = df_angle['t (s)'].to_numpy()
    
    angle = angle * np.pi/180 # conversion deg to rad
    
    plt.plot(time,angle)
    plt.xlabel("Tempo [s]")
    plt.ylabel(r"Angolo $\theta$")
    if save_fig:
        plt.savefig(name)
    plt.show()
    
    
    
    
    # media e deviazione standard campionaria
    mean_angle = angle.mean()
    smean_angle = np.std(angle,ddof=1)/np.sqrt(nmis) #divido per il numero di misurazioni
    
    PrintResult("Angolo",mean_angle,smean_angle,10,"[Rad]" )
    
    convert_factor = 180/np.pi
    PrintResult("Angolo",mean_angle*convert_factor ,smean_angle*convert_factor,3,"[Deg]" )
    print("")
    

    
    
    return mean_angle,smean_angle

# ricavare angolo (in rad) dal dataset di phyphox (in deg)
def get_angle(angle_data):
    
    angle = angle_data * np.pi/180 # conversion deg to rad
    
    # media e deviazione standard campionaria
    mean_angle = angle.mean()
    smean_angle = np.std(angle,ddof=1) #divido per il numero di misurazioni
    
    PrintResult("Angolo",mean_angle,smean_angle,3,"[Rad]" )
    
    convert_factor = 180/np.pi
    PrintResult("Angolo",mean_angle*convert_factor ,smean_angle*convert_factor,3,"[Deg]" )
    
    
    return mean_angle,smean_angle



# fit passando il nome del file .csv e l'intervallo di tempo interval=[start,finish]
def fit_from_gyro_phyphox(file_name, interval,plot=True):
    
    df = pd.read_csv(file_name)

    data_time = df['Time (s)'].to_numpy()
    data_y = df['Gyroscope y (rad/s)'].to_numpy()
    
    aplha = m_from_fit(data_time,data_y,interval[0],interval[1],plot)
    
    return aplha

def get_intervals(time, data, bottom,top):
    
    intervals = []
    indexes = []
    x = 0
    while x < len(time)-5:
        temp_bt = None
        
        #over bottom
        if data[x] >= bottom and data[x] <= top:
            temp_bt = x
            
            while data[x] >= bottom and data[x] <= top and x < len(time)-5 :
                x += 5
            
            #if over top
            if data[x] >= top:
                intervals.append([time[temp_bt],time[x]])
                indexes.append([temp_bt,x])
                
                while x < len(time)-5 and data[x] >= bottom:
                    x += 5
                    
        x += 5
                             
    return intervals,indexes

def easy_linfit(x,ux,y,uy,save_fig=None):
    
    fig_name = None
   
    #eseguire fit
    if save_fig is not None:
        fig_name = save_fig 
     
    print("prima iterazione (assumo sigma x=0)\n")
    m0, sm0, c0, sc0, cov0, rho0 = my.lin_fit(x, y, uy, "x [ux]", "y [uy]",  0, x.max()+0.01, 0, y.max()+0.1, setrange=True,save_fig=fig_name)    
    plt.cla()
    
    print("\n nuove y \n")
    uy_new = np.sqrt(uy**2+(m0*ux)**2)
    m, sm, c, sc, cov, rho = my.lin_fit(x, y, uy_new, "x [ux]", "y [uy]", 0, x.max()+0.01, 0, y.max()+0.1, setrange=True,save_fig=fig_name)
    
    # Studio dei residui
    if save_fig is not None:
        fig_name = "residui-"+save_fig 
        
    y_atteso = m*x + c
    d = y - y_atteso
    d_norm = d / uy_new
    
    plt.cla()

    if save_fig is not None:
        fig_name = "residui-"+save_fig 
        plt.errorbar(x,d_norm,uy_new/uy_new,marker='.',linestyle="")
        plt.ylabel("Residui normalizzati $d/\sigma_y=(y-y_{atteso})/\sigma_y$")
        plt.xlabel("$x$ [ux]")
        plt.grid()
        plt.savefig(fig_name)
    
    # Incertezze a posteriori
    sigmy_post = math.sqrt( np.sum(d**2)/(d.size-2) )
    uy_post = np.repeat(sigmy_post,y.size)

    # Nuovo fit con incertezze a posteriori sulle y
    print("\n Nuovo fit con incertezze a posteriori sulle y \n")
    if save_fig is not None:
        fig_name = "posteriori-"+save_fig 
       
   
    plt.cla()
    m1, sm1, c1, sc1, cov1, rho1 = my.lin_fit(x, y, uy_post, "x [ux]", "y [uy]", 0, x.max()+0.1, 0, y.max()+1, setrange=True,save_fig=fig_name)
    

    chi2(x,y,uy,m1,c1)
    
    return [ 
        [ m,sm ],
        [ m1,sm1 ],
    ]

