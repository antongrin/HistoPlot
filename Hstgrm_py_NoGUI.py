# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 16:53:23 2016

@author: GrinevskiyAS
"""

from __future__ import division
import numpy as np
#from numpy import sin,cos,tan,pi,sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import norm
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tkr


mpl.rc("figure", figsize=(9, 4))
mpl.rc('font',**{'family':'Liberation'})
sns.set(rc={'axes.facecolor':[0.97,0.97,0.98],"axes.edgecolor":[0,0,0],"grid.linewidth": 1,
            "axes.titlesize":16,"xtick.labelsize":14,"ytick.labelsize":14})
#sns.set_context("paper")   





def plothisto(data_c,data_nc,binsize,ax):
    u"""Строим гистограмму для коллектора и неколлектора с заданным размером бина """
    
#    Расчёт координат краёв бинов
    min_c=np.min(data_c)
    min_nc=np.min(data_nc)
    max_c=np.max(data_c)
    max_nc=np.max(data_nc)
    minedge=np.min((min_c,min_nc))
    minedge=np.floor(minedge/binsize)*binsize
    maxedge=np.max((max_c,max_nc))
    maxedge=np.floor(maxedge/binsize + 1)*binsize
    binedges=np.arange(minedge,maxedge+binsize,binsize)

#    Задайм свойства отображения гистограмм и линий для коллектора и неколлектора    
    coll_props={"linewidth": 1,"alpha": 0.5, "color": [1, 0.6, 0.2],'edgecolor':'w'}
    nc_props={"linewidth": 1,"alpha": 0.5, "color": [0.41, 0.24, 0.98],'edgecolor':'w'}
    
    coll_fit_props={"linewidth": 1,"alpha": 1, 'ls':'--', "color": [0.9, 0.39, 0.05]}
    nc_fit_props={"linewidth": 1,"alpha": 1, 'ls':'--', "color": [0.41, 0.24, 0.98]}

#    Веса, чтобы сумма высот столбцов равнялась 1
    weights_c = np.ones_like(data_c)/len(data_c)
    weights_nc = np.ones_like(data_nc)/len(data_nc)
    
#    Гистограммы
    ax.hist(data_nc, bins=binedges, normed=False,weights=weights_nc, **nc_props )
    ax.hist(data_c, bins=binedges, normed=False,weights=weights_c, **coll_props )
#    Подобранные pdf-ки
    ax.plot(binedges,binsize*norm.pdf(binedges,np.mean(data_c),np.std(data_c)),**coll_fit_props)
    ax.plot(binedges,binsize*norm.pdf(binedges,np.mean(data_nc),np.std(data_nc)),**nc_fit_props)

#    Это чтобы точки заменить на запятые
    ax.get_yaxis().set_major_formatter( tkr.FuncFormatter(lambda x, pos: str(x)[:str(x).index('.')]+','+str(x)[(str(x).index('.')+1):]) )
    
    return ax
    
    
def stats_for_table(data_c,data_nc,isint=True):
    """ Расчёт среднего и стандартного отклонения для выборок и преобразование их в текст """
    if isint:
        #округляем скорости до целого
        tbl_data=np.rint([[np.mean(data_c),np.mean(data_nc)],[np.std(data_c),np.std(data_nc)]]).astype(int) 
        tbl_flat=tbl_data.ravel()    
        for i,el in enumerate(tbl_flat):
            tbl_flat[i]='{0}'.format(el)
        tbl_data=tbl_flat.reshape(2,2)
    else:
        #округляем плотности до сотых
        tbl_data=np.round([[np.mean(data_c),np.mean(data_nc)],[np.std(data_c),np.std(data_nc)]],decimals=2)
        tbl_flat=tbl_data.ravel().astype('object')
        for i,el in enumerate(tbl_flat):
            tbl_flat[i]='{:.2f}'.format(el).replace('.',',')
        tbl_data=tbl_flat.reshape(2,2)
    
    return tbl_data    
    
    
def add_table_to_ax(ax,table_data):
    """ Помещение таблицы со статистикой на оси """
        
#   Создание таблицы, в которой будут эти честыре параметра
    NewTbl=[['',u'Коллектор',u'Неколлектор'],  [u'Среднее:','',''],  [u'Ст.откл.:',[''],['']]]
    NewTbl[1][1]=table_data[0,0]    
    NewTbl[1][2]=table_data[0,1]
    NewTbl[2][1]=table_data[1,0]
    NewTbl[2][2]=table_data[1,1]
    
#   Размещаем на осях таблицу на соответствущих координатах (bbox)
    htbl=ax.table(cellText=NewTbl,bbox=[0.05, 0.2, 0.8, 0.6],cellLoc='center',colWidths=[1,0.91,1])
    
#    Задаём формат каждой ячейки
    for icell in htbl.get_children():        
        icell.set_facecolor('w')
        icell.set_edgecolor('w')
        icell.set_linewidth(4)

#    Задаём формат заголовков (цветной фон)
    cell_coll=htbl.get_celld()[(0,1)]
    cell_coll.set_facecolor([1, 0.6, 0.2,0.6])
    cell_nc=htbl.get_celld()[(0,2)]
    cell_nc.set_facecolor([0.41, 0.24, 0.98,0.5])
    
    htbl.auto_set_font_size(False)    
    htbl.set_fontsize(14)

#    Выделяем жирным цифры
    htbl.get_celld()[(1,1)].set_text_props(fontname='Liberation Serif',fontsize=18) #weight='semibold',
    htbl.get_celld()[(1,2)].set_text_props(fontname='Liberation Serif',fontsize=18)
    htbl.get_celld()[(2,1)].set_text_props(fontname='Liberation Serif',fontsize=18)
    htbl.get_celld()[(2,2)].set_text_props(fontname='Liberation Serif',fontsize=18)

#    Отключам оси-подложку    
    ax.axis('off')
    
    return htbl
    


#           формат файла: 
#   <глубина> <Vp> <Vs> <Density> <Coll>

data=np.loadtxt(r'D:\Projects\Komandirshor\AVO_Model\ready_Sk4_forhisto_zd_kolMerged2.txt')

dd=data[:,0]
vp=data[:,1]
vs=data[:,2]
den=data[:,3]
coll=data[:,4]


#Выделяем индексы коллекторов и неколлекторов
ind_c=np.flatnonzero(coll>0.5)
ind_nc=np.flatnonzero(coll<0.5)


#Сделать True, чтобы считать несколько пропластков единым коллектором с некоторыми осредненными свойствами
single_plast_flag=False

if single_plast_flag:
    ind_c2=np.arange(ind_c[0],ind_c[-1]+1)
    ind_nc2=np.hstack((np.arange(0,ind_c[0]),np.arange(ind_c[-1]+1,ind_nc[-1]+1)))
    ind_c=ind_c2
    ind_nc=ind_nc2


vp_c=vp[ind_c]
vp_nc=vp[ind_nc]
vs_c=vs[ind_c]
vs_nc=vs[ind_nc]
den_c=den[ind_c]
den_nc=den[ind_nc]

#Размеры карманов для гистограммы
vp_bin_size=100
vs_bin_size=50
den_bin_size=0.02


f3=plt.figure(facecolor='w',figsize=(18,7))


#Грид лдя осей, в верхнем ряду (ширина 2) - гистограммы, в нижнем ряду - таблички с цифрами (ширина 1)
gs = gridspec.GridSpec(2,3, height_ratios=[2,1])
ax_vp=f3.add_subplot(gs[0,0])
ax_vs=f3.add_subplot(gs[0,1])
ax_den=f3.add_subplot(gs[0,2])



ax_vp=plothisto(vp_c,vp_nc,vp_bin_size,ax_vp)
ax_vp.set_title(u'$\mathregular{V_P}$, м/с')

ax_vs=plothisto(vs_c,vs_nc,vs_bin_size,ax_vs)
ax_vs.set_title(u'$\mathregular{V_S}$, м/с')
ax_den=plothisto(den_c,den_nc,den_bin_size,ax_den)
ax_den.set_title(u'Плотность, $\mathregular{г/см^3}$')
ax_den.get_xaxis().set_major_formatter( tkr.FuncFormatter(lambda x, pos: str(x)[:str(x).index('.')]+','+str(x)[(str(x).index('.')+1):]) )


#Добавляем таблицы 
ax_vp_tbl=f3.add_subplot(gs[1,0])
htbl_vp=add_table_to_ax(ax_vp_tbl, stats_for_table(vp_c,vp_nc)  )
ax_vs_tbl=f3.add_subplot(gs[1,1])
htbl_vs=add_table_to_ax(ax_vs_tbl, stats_for_table(vs_c,vs_nc)  )
ax_den_tbl=f3.add_subplot(gs[1,2])
htbl_den=add_table_to_ax(ax_den_tbl, stats_for_table(den_c,den_nc,isint=False)  )




f3.tight_layout()

f3.savefig(r'fig1.png',bbox_inches='tight')