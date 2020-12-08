#!/usr/bin/env python
# -*- coding: utf-8 -*-
###
#see my_plinn.c
###

#Written by Hiromu Kitayama
#20190724
#ブロードキャストとnumpyによるベクトル演算を使用しかなり高速化されたが、まだ精度と計算時間はC言語に劣る
#20181220
#高速化するための工夫を考えた方がいいかも
#全体的に学習精度がそこまで良くはならなかった。どこが悪いのか探した方が良い

######
#import numpy as np
#original
#import numpy as xp
#cupy
#import cupy as xp
import switchCupy
xp = switchCupy.xp_factory()
xpfloat=switchCupy.xpfloat(xp)
####
import numpy
######
import math
import copy
import am #am_kitayama
#import can2 #can2_kitayama
import my_function
import scipy as sp
#from scipy.spatial.distance import euclidean
#import numba
import time
#from itertools import product
#from joblib import Parallel,delayed 
#import cython
#
ZERO=1e-20
#
#define LA(x,y1,x1,y2,x2) ((y1)+((x)-(x1))*((y2)-(y1))/((x2)-(x1))) //Linear Approximation
def LA(x,y1,x1,y2,x2):
  return ((y1)+((x)-(x1))*((y2)-(y1))/((x2)-(x1))) #//Linear Approximation




#####################w計算におけるソートの初期化##########################
#def init_sort_weights(net,s,v,x,_x):
##def init_sort_weights(net,s,v,x):
#  n_cells = net['n_cells']
#  n_channels = net['k']
#  
#  #初期化
#  n_fires=0
#  s=xp.arange(n_cells)
#
#  #最小距離見つける --> modify to broadcast soon
#  for i in xrange(n_cells):
#    v[i] = 0.0
#    if net['v'][i] > 0:
#      n_fires+=1
#      for k in xrange(n_channels):
#        v[i] += (net['w'][i,k]-x[k])**2
#    else:
#      v[i] =1e+12
#
#  i_v_min = xp.argmin(v)		#最小となるインデックスを格納
##  v_min = v[i_v_min]			#最小距離
#
#  net['n_fires'] = n_fires
#  s[i_v_min]=0
#  s[0]=i_v_min
#
#  return s,v
#
def init_sort_weights(net,x,_x):
#def init_sort_weights(net,s,v,x):
  n_cells = net['n_cells']
  n_channels = net['k']
  
  #初期化
  n_fires=0
  #最小距離見つける
#  z=x[:n_channels]-net['w']  #broadcast 20190723
#  v=xp.sum(z**2,axis=1)
  v=xp.linalg.norm(_x-net['w'],axis=1)
#  v=xp.linalg.norm(x[:n_channels]-net['w'],axis=1)
#  import pdb;pdb.set_trace() #for debug 
#  v[net['None_fire_number']]=1e+1
#
#slower bloadcast than xrange?
  I=xp.where(net['v']<=ZERO)
  n_fires=n_cells-len(I)
  v[I]=1e+12
#faster xrange than broadcast?
#  for i in xrange(n_cells):
#    if net['v'][i] > ZERO: 
#      n_fires+=1
#    else:
#      v[i] =1e+12
  s=v.argsort(axis=0,kind='quicksort') #  s=v.argsort(axis=0,kind=sortkind) #
  i_v_min = xp.argmin(v)		#最小となるインデックスを格納
#  v_min = v[i_v_min]			#最小距離
  net['n_fires'] = n_fires
  s[i_v_min]=0
  s[0]=i_v_min
  #import pdb;pdb.set_trace() #for debug
  return s,v
#
#####################どのボロノイ領域に近いかソートする関数################
def sort_weights(net,s,v,start,end):
  n_fires = net['n_fires']
 
  if n_fires < end:
    return s,v
  #ソート
  for i in xrange(start,end):
    for j in xrange(i+1,n_fires):
      if v[int(s[i])] > v[int(s[j])]:
        s[i],s[j]=s[j],s[i] 
  return s,v

##############

#####################出力計算#############################
def calc_output(net,x,_x):
#def calc_output(net,x,yr,y,Y):
  n_cells=net['n_cells']
  n_channels=net['n_channels']
#  import pdb;pdb.set_trace() #for debug 
  s,v=init_sort_weights(net,x,_x)
#  s=xp.zeros((n_cells),dtype=xp.int32)
#  v=xp.zeros((n_cells,),dtype=xpfloat)
#  s,v=init_sort_weights(net,s,v,x)

  n_fires=net['n_fires']
#一番近いセルを探す
  for i in xrange(n_fires):
    if(net['v'][s[i]]>ZERO):	#セルが消滅しているとき２個目に近いところのセルを使う
      break;
    i+=1
    if i>=n_fires:
      i=0
      break;
#    s,v=sort_weights(net,s,v,i,i+1) #20190727

  net['c']=[s[i]]
#  y=xp.sum(net['am']['M'][s[i],0]*x) #slower?
#  y=xp.dot(net['am']['M'][s[i],0],x) #slower?
  y=0
  for j in xrange(n_channels+1):
    y+= net['am']['M'][s[i]][0,j]*x[j]

  ymin=net['ymin']
  ymax=net['ymax']
  if y < ymin: y=ymin
  if y > ymax: y=ymax
  yr=y
  if net['r1']>0:
    if yr<net['r'][0]:
      yr=net['r'][0]
    elif yr>net['r'][net['nr']]:
      yr=net['r'][net['nr']]
  if net['r1']>0:
    ii=floor((y-ymin)/net['r12']+0.5)
    if ii > net['nr']:
      ii=net['nr']
    elif ii < 0:
      ii=0
    y=net['r'][ii]
    Y=net['R'][ii]
  else:
#    import pdb;pdb.set_trace() #for debug 
    Y=my_function.moverange(y,net['ymin'],net['ymax'],net['ymin0'],net['ymax0'])
  return yr,y,Y #def calc_output(net,x,_x):

###modify from calc_output2 
def calc_output2_(net,x,_x):
#def calc_output2(net,x,givendata):#x=test['x']
  n_cells=net['n_cells']
  n_channels=net['n_channels']
  n_x=x.shape[0] #  n_total=givendata['n_total']
  n_fires=net['n_fires']
  ymin=net['ymin']
  ymax=net['ymax']
#  s=xp.zeros((n_cells),dtype=xp.int32)
#  v=xp.zeros((n_cells,),dtype=xpfloat)
#  import pdb;pdb.set_trace() #for debug 
  zz = xp.sum((x[:n_channels].reshape(1,1,n_channels) - net['w'].reshape(1,n_cells,n_channels))**2, axis=2)
#  zz = xp.sum((x[:,:n_channels].reshape(n_x,1,n_channels) - net['w'].reshape(1,n_cells,n_channels))**2, axis=2)
#  import pdb;pdb.set_trace() #for debug 
  zz[:,net['i_firezero']]=1e+12 #20190730 kuro #i_firezero->i of v<=0
#  for k in xrange(len(net['i_firezero'])):
#    zz[:,net['i_firezero'][k]]=1e+12
  zzs=zz.argmin(axis=1)		#一番近いセル探し
  net['c']=zzs

#  y=xp.zeros((n_x),dtype=xpfloat)
#  yr=xp.zeros((n_x),dtype=xpfloat)
  Y=xp.zeros((n_x),dtype=xpfloat)
#  import pdb;pdb.set_trace() #for debug 
  y=xp.sum(net['am']['M'][zzs,0]*x,axis=1) # broadcast for inner product (np.dot for many x[:,:]) 
  y=xp.where(y<ymin,ymin,y)
  y=xp.where(y>ymax,ymax,y)
  yr=copy.deepcopy(y)
#  import pdb;pdb.set_trace() #for debug 
  if net['r1']>0: #discrete output
    yr=xp.where(yr<net['r'][0],net['r'][0],yr)
    yr=xp.where(yr>net['r'][net['nr']],net['r'][net['nr']],yr)
#    import pdb;pdb.set_trace() #for debug 
    ii=xp.floor((y-ymin)/net['r12']+0.5).astype(xp.int32)
    y=net['r'][ii] #not checked yet
    Y=net['R'][ii] #not checked yet
  else:
#    import pdb;pdb.set_trace() #for debug 
    Y=my_function.moverange(y,net['ymin'],net['ymax'],net['ymin0'],net['ymax0'])

  return yr,y,Y

###############################ブロードキャストによる出力計算#######################
#@numba.jit
def calc_output2(net,x,givendata):#x=test['x']
  n_cells=net['n_cells']
  n_channels=net['n_channels']
  n_x=x.shape[0] #  n_total=givendata['n_total']
  n_fires=net['n_fires']
  ymin=net['ymin']
  ymax=net['ymax']
#  s=xp.zeros((n_cells),dtype=xp.int32)
#  v=xp.zeros((n_cells,),dtype=xpfloat)

  zz = xp.sum((x[:,:n_channels].reshape(n_x,1,n_channels) - net['w'].reshape(1,n_cells,n_channels))**2, axis=2)
###  z = _x.reshape(n_x,1,n_channels) - net['w'].reshape(1,n_cells,n_channels)
####  z = _x[:,:].reshape(n_total,1,n_channels) - net['w'].reshape(1,n_cells,n_channels)
####  z = x[:,:n_channels].reshape(n_x,1,n_channels) - net['w'].reshape(1,n_cells,n_channels)
###  zz = xp.sum(z**2, axis=2)
#  import pdb;pdb.set_trace() #for debug 
  zz[:,net['i_firezero']]=1e+12 #20190730 kuro #i_firezero->i of v<=0
#  for k in xrange(len(net['i_firezero'])):
#    zz[:,net['i_firezero'][k]]=1e+12
  zzs=zz.argmin(axis=1)		#一番近いセル探し
  net['c']=zzs

#  y=xp.zeros((n_x),dtype=xpfloat)
#  yr=xp.zeros((n_x),dtype=xpfloat)
  Y=xp.zeros((n_x),dtype=xpfloat)
#  import pdb;pdb.set_trace() #for debug 
  y=xp.sum(net['am']['M'][zzs,0]*x,axis=1) # broadcast for inner product (np.dot for many x[:,:]) 
  y=xp.where(y<ymin,ymin,y)
  y=xp.where(y>ymax,ymax,y)
  yr=copy.deepcopy(y)
#  import pdb;pdb.set_trace() #for debug 
  if net['r1']>0: #discrete output
    yr=xp.where(yr<net['r'][0],net['r'][0],yr)
    yr=xp.where(yr>net['r'][net['nr']],net['r'][net['nr']],yr)
#    import pdb;pdb.set_trace() #for debug 
    ii=xp.floor((y-ymin)/net['r12']+0.5).astype(xp.int32)
    y=net['r'][ii] #not checked yet
    Y=net['R'][ii] #not checked yet
  else:
#    import pdb;pdb.set_trace() #for debug 
    Y=my_function.moverange(y,net['ymin'],net['ymax'],net['ymin0'],net['ymax0'])
#    for t in xrange(n_x):
#      Y[t]=can2.moverange(y[t],net['ymin'],net['ymax'],net['ymin0'],net['ymax0'])
#  for t in xrange(n_x):
##    y[t]=xp.dot(net['am']['M'][int(zzs[t]),0],x[t]) 
##    if y[t] < ymin: y[t]=ymin
##    if y[t] > ymax: y[t]=ymax
##    yr[t]=y[t]
##    if net['r1']>0:
##      if yr[t]<net['r'][0]:
##        yr[t]=net['r'][0]
##      elif yr[t]>net['r'][net['nr']]:
##        yr[t]=net['r'][net['nr']]
#    if net['r1']>0:
#      ii=floor((y[t]-ymin)/net['r12']+0.5)
#      if ii > net['nr']:
#        ii=net['nr']
#      elif ii < 0:
#        ii=0
#      y[t]=net['r'][ii]
#      Y[t]=net['R'][ii]
#    else:
#      Y[t]=can2.moverange(y[t],net['ymin'],net['ymax'],net['ymin0'],net['ymax0'])

  return yr,y,Y



#########################境界計算###################################
def in_window(x1,x2,x,width2,n_channels):
#def in_window(x1,x2,x,width1,n_channels):
#def in_window(x1,x2,x,width,n_channels):
# this broadcast is slow why? low dimension (n_channels is small) ?
#(1) slower broadcast than xrange below
#  x12=x1-x2
#  l2=xp.sum((x12)**2)
#  l3=xp.sum((x-x2)*(x12))
#(2) soso 
#  x12=x1-x2
#  l2=xp.dot(x12,x12)
#  l3=xp.dot(x-x2,x12)
##  l2=xp.sum(x12**2)
##  l3=xp.sum((x-x2)*x12)
#(3) faster xrange than broadcast! why?
  l2 =l3=0.0 ##  l3 = 0.0
  for i in xrange(n_channels):
    x12i=x1[i]-x2[i]
    l2 += x12i**2 #    l2 += float(x1[i] - x2[i])**2
    l3 += (x[i] - x2[i]) * x12i  
#  for i in xrange(n_channels):
#    l2 += (x1[i] - x2[i])**2 #    l2 += float(x1[i] - x2[i])**2
#    l3 += (x[i] - x2[i]) * (x1[i] - x2[i])  
#####
  if l2 <= 0.0:
    print('### Error: l2={}'.format(l2))
#    import pdb;pdb.set_trace() #for debug 
    return False
  else:
    return (width2-abs(l3/l2-0.5)>1e-16) #same as the result for withd1 or www
#    ret=width2-abs(l3/l2-0.5)>1e-17 #different from the result for width1 or www
#
#    width1 = (1.-width)/2.  #www = (1.-width)/2.
#    width1=0.5-width2
#    l3l2 = l3/l2
#    ret1=(l3l2-width1>0) and (1.-width1-l3l2 >0)
#    if ret!=ret1:
#      import pdb;pdb.set_trace() #for debug 
#    ret = (width1 < l3l2) and (l3l2 < 1.-width1)
#    ret = int((www < l3l2) and (l3l2 < 1.-www))
#  return ret 

#in_window = xp.ElementwiseKernel(
#        'T x1, T x2, T width2, T n_channels',    #１番目の引数:in_params:入力を定義します。(複数あればカンマ区切りのリストで)
#        'int32 z',               #２番目の引数:out_params:出力を定義します。
#        'z = (x - y) * (x - y);',   #３番目の引数:operation: CUDA-C/C++の表記で計算内容を記載します。
#        'squared_diff')            #４番目の引数:name:内部で使用する関数の識別です。

def checkwM(cmt,net):
  if 1==0: #change if 1==1: for check
    net['print']('start modify_w_batch')
    for i in xrange(net['n_cells']):net['print']('{} w{} M{}'.format(i,net['w'][i],net['am']['M'][i]))
  return
######################荷重ベクトルWの更新を行う関数########################
def modify_w_batch(net,x,y,n_train,GlobalTime,_x):
#  width = net['width']			#wの更新を行うボロノイの境界幅
#  width1 = net['width1']			#wの更新を行うボロノイの境界幅
  width2 = net['width2']			#width2=width/2
#  if width1 != (1.-width)/2.:
#    import pdb;pdb.set_trace() #for debug 
  n_cells = net['n_cells']
  n_channels = net['k']
  n_compare = net['n_compare']		#n_compare個まで境界幅にあるか見る
  dwNorm = 0

#  checkwM('start modify_w_batch',net)
  V = net['V']
  #sとd2の初期化
#  s=xp.zeros((n_cells),dtype=xp.int64)
#  d2=xp.zeros((n_cells,),dtype=xpfloat)

  #dwの初期化
  net['dw']=xp.zeros((n_cells,n_channels),dtype=xpfloat)
#  for i,t in product(xrange(n_cells),xrange(n_cells)): #canot be used because L375 slower?
  for i in xrange(n_cells):
#    for t in V['ij2t'][i]:
    for j in xrange(V['i2v0'][i]):
      t = V['ij2t'][i][j]

      d2=xp.sum((_x[t,:] - net['w'])**2,axis=1)
#      d2=xp.sum((x[t,:n_channels] - net['w'])**2,axis=1)
#      z=(x[t,:n_channels] - net['w'])**2
#      d2=xp.sum(z,axis=1)
#      import pdb;pdb.set_trace() #for debug 
      d2[net['i_firezero']]=1e+12
#      for k in xrange(len(net['i_firezero'])):
#        d2[net['i_firezero'][k]]=1e+12
      s=d2.argsort(axis=-1)			#最小距離

#      s,d2=init_sort_weights(net,s,d2,x[t])						#ソートの初期化
      for xi in xrange(1,n_compare):					#そのxの周辺n_compare分のセルの境界を見る 
#        s,d2=sort_weights(net,s,d2,xi,xi+1)						#どの近傍セルを探し、ソート
        sxi = s[xi]								#2番目以降の近いセル
        if sxi == i:								#もし自分自身のセルだったら無視
          continue
#        print('### check i{} j{} sxi{}'.format(i,j,sxi))
#        print('### check i{} {} w{} x{} w{} {} {}'.format(i,sxi,net['w'][i],_x[t,:],net['w'][sxi], (_x[t,:]-net['w'][sxi])/(net['w'][sxi]-net['w'][i]), (net['w'][i]-_x[t,:])/(net['w'][sxi]-net['w'][i])))
        if in_window(net['w'][sxi],net['w'][i],_x[t,:],width2,n_channels):
#          print('### check i{} {} w{} x{} w{} {} {}'.format(i,sxi,net['w'][i],_x[t,:],net['w'][sxi], (_x[t,:]-net['w'][sxi])/(net['w'][sxi]-net['w'][i]), (net['w'][i]-_x[t,:])/(net['w'][sxi]-net['w'][i])))
#          print('### check i{} {} w{} x{} w{} {} {} w45={} dw45={}'.format(i,sxi,net['w'][i],_x[t,:],net['w'][sxi], (_x[t,:]-net['w'][sxi])/(net['w'][sxi]-net['w'][i]), (net['w'][i]-_x[t,:])/(net['w'][sxi]-net['w'][i]),net['w'][45],net['dw'][45]))
#          import pdb;pdb.set_trace() #for debug 
#        if in_window(net['w'][sxi],net['w'][i],x[t,:n_channels],width2,n_channels):
#        if in_window(net['w'][sxi],net['w'][i],x[t],width2,n_channels):
#        if in_window(net['w'][sxi],net['w'][i],x[t],width,n_channels):
#        if in_window(net['w'][sxi],net['w'][i],x[t],width,n_channels):
#        if in_window(net['w'][sxi],net['w'][i],x[t],width,n_channels) > 0:
          y_hat_i=xp.dot(net['am']['M'][i,0],x[t])
          y_hat_xi=xp.dot(net['am']['M'][int(sxi),0],x[t])
          delta_w_ic = 0.0
          for k in xrange(n_channels):
            delta_w_ic += float(net['w'][sxi,k] - net['w'][i,k])**2
            #delta_w_ic += float(x[t,k] - net['w'][i,k])**2
          delta_w_ic = float(math.sqrt(delta_w_ic))
#          err_xi = abs(y_hat_xi - y[t]) / net['ywidth']
#          err_i  = abs(y_hat_i  - y[t]) / net['ywidth']
          err_xi = (y_hat_xi - y[t]) / net['ywidth']
          err_i  = (y_hat_i  - y[t]) / net['ywidth']

          if  delta_w_ic < 1e-20:
            continue
#          alpha = 0.001 / delta_w_ic
          alpha = ((float((err_xi)**2)) - (float((err_i)**2))) / delta_w_ic

#          if GlobalTime == 53: 
#            print('Globaltime=53')
#            import pdb;pdb.set_trace() #for debug  check 20191030

          if alpha>0: #1==1: #
            net['dw'][i]   += alpha*(_x[t,:]-net['w'][i])
            net['dw'][sxi] -= alpha*(_x[t,:]-net['w'][sxi])
#            print("### check i={},{} alpha={:.3e} dwi={} dwxi={}".format(i,sxi,alpha,net['dw'][i],net['dw'][sxi]))
#          import pdb;pdb.set_trace() #for debug 
#          net['dw'][i] += alpha*(x[t,0:n_channels]-net['w'][i])
#          net['dw'][sxi] -= alpha*(x[t,0:n_channels]-net['w'][sxi])
##          for k in xrange(n_channels):
##            net['dw'][i,k]   += alpha * (x[t][k] - net['w'][i,k])
##            net['dw'][sxi,k] -= alpha * (x[t][k] - net['w'][sxi,k])
##            print('t{} x{} w{} wx{}'.format(t,x[t][k],net['w'][i,k],net['w'][sxi,k]))
##            net['cell'][i]['dw'][k] += alpha * (x[t][k] - net['cell'][i]['w'][k])
##            net['cell'][sxi]['dw'][k] -= alpha * (x[t][k] - net['cell'][sxi]['w'][k]          
          break	  #modified20180111
    
    for k in xrange(n_channels):
      if dwNorm < abs(net['dw'][i,k]):   dwNorm = abs(net['dw'][i,k])  #max dw?
#  dwNorm=xp.max(abs(net['dw']))
#  import pdb;pdb.set_trace() #for debug 
###########
#  if GlobalTime >= 53 and GlobalTime <= 63:  
#   alpha *=0.01
#
  if dwNorm>0:
    alpha = net['gamma0'] / (1.+float(GlobalTime)/net['Tgamma'])*(net['xwidth'])/(dwNorm) #float!!!!20190115
#    alpha = net['gamma0'] / (1.+float(GlobalTime)/net['Tgamma'])*(net['xwidth'])/(dwNorm+1.0e-10) #float!!!!20190115
    net['w']+=alpha*net['dw']
    #for print
    if net['nop']==0:#print=not no-print
      max_dw=dwNorm #  max_dw=max(abs(net['dw']))[0]
      Iw=xp.where(net['dw']!=0)[0]
      wl=net['w'][Iw].reshape(1,-1)[0].tolist()
      wls=''
      Iws=''
      for i,e in enumerate(Iw): 
        Iws+='{} '.format(e)
        if i>=10: 
          break
      for i,e in enumerate(wl): 
        wls+='{:.2g} '.format(e)
        if i>=10: 
          break
  #    import pdb;pdb.set_trace() #for debug  check 20191030
      net['print']('{} #modify_w_batch alpha={:.3g} max_dw={:.3g} w[{}...]=[{}...]'.format(net['GlobalTime'],alpha,max_dw,Iws,wls))
#    net['print']('{} #modify_w_batch alpha={:.3g} max_dw={:.3g} w{}...=[{}...]'.format(net['GlobalTime'],alpha,max_dw,Iw[0:10],wls[0:10]))
#     print('Globaltime=53')
#    import pdb;pdb.set_trace() #for debug  check 20191030
###########

#  import pdb;pdb.set_trace() #for debug 
#  for i in xrange(n_cells):
#    print('[{}]='.format(i)),
#    for k in xrange(n_channels):
#      net['w'][i,k] += (net['dw'][i,k]*alpha)
#      net['w'][i,k] = net['w'][i,k] + net['dw'][i,k]*alpha
#      print('w{} += dw{} * alpha{}'.format(net['w'][i,k], net['dw'][i,k],alpha)),
#    print(' ')
#      net['cell'][i]['w'][k] = net['cell'][i]['w'][k] + net['cell'][i]['dw'][k]*alpha
      #n=3
#  del s
#  del d2 

  return #def modify_w_batch(net,x_train,y_train,n_train):


###############αiの計算を行う関数(ret=1なら再初期化する)######################
#@numba.jit
def calc_alpha(net,x,y,n_train,GlobalTime):
  n_cells = net['n_cells']
  n_channels = net['k']
  k1=n_channels+1
  V = []
  V = net['V']
  S = 0.0
  n_alphai = 1e-20
  alpha_sum = 1e-20
  H = 0
#  S0 = 0.0
  #各セルの平均二乗和を求める
  for i in xp.where(V['i2v']>0)[0]: #??V['i2v0'][i]=len(V['ij2t'][i]?#  for i in xp.where(V['i2v0']>0)[0].astype(xp.int32): 
#  for i in xp.where(V['i2v0']>0)[0]: #??V['i2v0'][i]=len(V['ij2t'][i]?#  for i in xp.where(V['i2v0']>0)[0].astype(xp.int32): 
    ii=int(i)
    y_m=xp.sum((net['am']['M'][ii,0])*x[V['ij2t'][ii]],axis=1) #broadcast
    net['S'][ii]=xp.sum((y[V['ij2t'][ii],0]-y_m)**2,axis=0)
    S+=net['S'][ii]
#  print 'S={}'.format(S);import pdb;pdb.set_trace() #for debug 20191028checking
#??print V['i2v0'][1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
#    print('S={} e={}=y{}-ym{}'.format(S,net['S'][ii],y[V['ij2t'][ii],0],y_m))#    print('S0={},S={}'.format(S0,S))
    #for broadcast
#    ii=int(i)
#    y_m=xp.sum((net['am']['M'][int(ii),0])*x[V['ij2t'][int(ii)]],axis=1) #broadcast
#    net['S'][ii]=xp.sum((y[V['ij2t'][ii],0]-y_m)**2,axis=0)
#    S0+=net['S'][ii]
#
#  print('S={},Si={}'.format(S,net['S']))
#  import pdb;pdb.set_trace() #for debug 

#
#  for i in xrange(n_cells): 
#    #original: not for broadcast
#    net['S'][i] = 0.0
#    for j in xrange(V['i2v0'][i]):
#      t = V['ij2t'][i][j]
#      y_m = 0
#      y_m = xp.dot(net['am']['M'][i,0],x[t])
#      net['S'][i]+=(y[t][0]-y_m)**2			#各セルの予測値と真値との2乗誤差(S[i]) #20190806 y[t]->y[t][0]
#    S+=net['S'][i]					#全セルの二乗誤差の総和
   
#  import pdb;pdb.set_trace() #for debug 
  net['St'][int(GlobalTime)] = S
  NDS = S/net['St'][int(GlobalTime-1)]-1.			#NDSの計算
  net['NDS']=NDS
###  print '{} NDS=S/S(t-1)-1={}/{}-1={} '.format(GlobalTime,S,net['St'][int(GlobalTime-1)],NDS) #DISP or not
#  print('{}'.format(GlobalTime)),
  #観測雑音σ^2の計算
  I=xp.where(net['v']>net['v_thresh'])[0]
  SvMin=min(net['S'][I]/net['v'][I]) #element wise division?
  n_alphai=len(I)+1e-20
#  import pdb;pdb.set_trace() #for debug 

#  SvMin=1e30
#  for i in xrange(n_cells):
#    if net['v'][i] < net['v_thresh']:			#発火比率がv_thresh未満だったら計算しない
#      continue
#    n_alphai+=1    
#    if SvMin > (net['S'][i] / net['v'][i]):
#      SvMin = (net['S'][i] / net['v'][i])	#Sとvの比率の最小値を求める
    
  SvMin/=V['vmax']						#SvMin/vmax
  net['sigma2_hat'] = SvMin					#観測雑音σ^2(論文式(20))若干違う?
  sigma2_hat = SvMin

  #推定歪αiと総和αの計算
#broadcast
#  import pdb;pdb.set_trace() #for debug 
  net['alpha'] = (net['S']-sigma2_hat*(net['v']*V['vmax']))/n_alphai	#推定歪αi(論文式(21))  
  if 'ElementwiseKernel'!= dir(xp)[0]: #xp=numpy
    Iv=xp.where(net['v']>=net['v_thresh'])[0]
    Ia=xp.where(net['alpha']>=1.0e-35)[0]
#    Ia=xp.where(net['alpha']>=1.0e-10)[0]
    Iva=xp.intersect1d(Iv,Ia)
  else: #use xp=cupy
    Vv=(xp.where(net['v']>=net['v_thresh'],1,0)*xp.where(net['alpha']>=1.0e-10,1,0))
    Iva=xp.where(Vv==1)[0]
  alpha_sum=xp.sum(net['alpha'][Iva])
#  import pdb;pdb.set_trace() #for debug 
#original not for broadcast
#  for i in xrange(n_cells):
#    net['alpha'][i] = (net['S'][i]-sigma2_hat*(net['v'][i]*V['vmax']))/n_alphai	#推定歪αi(論文式(21))  
#
#    if net['v'][i] < net['v_thresh']:  
#      continue
#
#    if net['alpha'][i] < 1.0e-10:
#      continue
#
#    alpha_sum+=net['alpha'][i]				#推定歪αiの総和

  #エントロピーHの計算
#broadcast
  if alpha_sum>0:
    aa = net['alpha'][Iva]/alpha_sum 
    H=-xp.sum(aa*xp.log(aa))
  else:
    H=1
##original not for broadcast
#  for i in xrange(n_cells):
#    if net['v'][i] < net['v_thresh']: 
#      continue
#    if net['alpha'][i] < 1.0e-10:
#      continue
#    aa = net['alpha'][i]/alpha_sum			#αi/Σαi
#    H -= aa*float(math.log(aa)) 				#H=-Σ(aa*ln(aa))エントロピー計算(論文式(23))
   
  #アルファの平均
  if int(n_alphai) >= 2:
    net['nentropy'] = H/float(math.log(n_alphai))		#H/ln(N)の計算(論文式(22))
  else:
    net['nentropy'] = 1

  alpha_bar = alpha_sum / float(n_alphai)			#<αi>(推定歪の平均値)
  if alpha_bar < 1e-20:						#アルファNGです
    alpha_NG = 1
  else:
    alpha_NG = 0

  net['alpha_bar'] = alpha_bar
  net['alpha_NG'] = alpha_NG

##  print 'sigma2hat={},alphai{},<alphai>{},NOrmalizedEntropy={}'.format(sigma2_hat,n_alphai,alpha_bar,net['nentropy']) #DISP or not

  NDS_thresh=0.0 #-0.01 #NDS_thresh=0.0
#  if GlobalTime == 53:
#    import pdb;pdb.set_trace() #for debug 
  reinit_flag1=NDS>=NDS_thresh
  reinit_flag2=net['nentropy']<net['nentropy_thresh']
  reinit_flag=reinit_flag1 and reinit_flag2
  NDSmes="(NG!)" if NDS>1 else ""
  net['print']('#check S={:.3g} ent{:.3g} NDS{}{:.3g}>={:.3g}:{} entropy{:.3g}<{:.3g}:{}-> Reinit={}'.format(S,net['nentropy'],NDSmes,NDS,NDS_thresh,reinit_flag1,net['nentropy'],net['nentropy_thresh'],reinit_flag2,reinit_flag))
#  import pdb;pdb.set_trace() #for debug 
  return reinit_flag
#  if NDS < NDS_thresh: #  if NDS < -0.01: 
#    return 0 #return 0 if S is decreasing
#  return 0 if net['nentropy'] >= net['nentropy_thresh'] else 1
#  if net['nentropy'] < net['nentropy_thresh']:
#    return 1
#  else:
#    return 0
#  return #def calc_alpha(net,x,y,n_train) #in effective?



############################連想行列Mの更新を行う関数###########################
def modify_M_batch_org(net,x,y):
  n_channels=net['k']
#  V=[]
  V=net['V']
  
#  checkwM('start modify_M_batch',net)
#      import pdb;pdb.set_trace() #for debug 
  if net['pinv']>0:
###(2)
    for i in xrange(net['n_cells']):
      if len(V['ij2t'][i])>0: 
        net['am']['M'][i]=xp.dot(xp.linalg.pinv(x[V['ij2t'][i],:]),y[V['ij2t'][i],:]).T #擬似逆行列 same as GSL in 
  else:
####(1)
    for i in xrange(net['n_cells']):
      for t in V['ij2t'][i]: #different result from the below: why?
####(1-2) faster
        am.calc_AMxy(net['am'],i,x[t],y[t])		#連想行列Mの更新(am_kitayama.py->kuro190809)
####(1-1) slower
####      net['am']['x'][i]=x[t]		#xの取り出し 20180113
####      net['am']['y'][i]=y[t]		#yの取り出し #      net['cell'][i]['am']['y'][0]=y[t]			#yの取り出し 
####      am.calc_AM(net['am'],i)		#連想行列Mの更新(am_kitayama.py)
######
####    if V['i2v'][i]!=len(V['ij2t'][i]):
####      print('{}!={}'.format(V['i2v'][i],len(V['ij2t'][i])))
####      import pdb;pdb.set_trace() #for debug 
####    for j in xrange(V['i2v'][i]):			#そのセルにあるxの個数分まで
####      t=V['ij2t'][i][j]					#訓練データ番号
####      import pdb;pdb.set_trace() #for debug 
####      if i<7:
#####        print('t={} x={} y={} M={} e={}'.format(t,net['cell'][i]['am']['x'], net['cell'][i]['am']['y'], net['cell'][i]['am']['M'], y[t]-xp.dot(net['cell'][i]['am']['M'],x[t])))
####        print('i{} j{} t{} x={} y={} M={} e={}'.format(i,j,t,net['cell'][i]['am']['x'], net['cell'][i]['am']['y'], net['cell'][i]['am']['M'], y[t]-xp.dot(net['cell'][i]['am']['M'],x[t])))
####  import pdb;pdb.set_trace(); #for debug 
####  checkwM('finish modify_w_batch',net)

  return #def modify_M_batch(net,x_train,y_train):

def modify_M_batch_RLS(net,x,y):#RLS (Recursive Least Square) method
  n_channels=net['k']
  V=net['V']
  
####(1)
  for i in xrange(net['n_cells']):
    for t in V['ij2t'][i]: #different result from the below: why?
####(1-2) faster
      am.calc_AMxy(net['am'],i,x[t],y[t])		#連想行列Mの更新(am_kitayama.py->kuro190809)
####(1-1) slower
####      net['am']['x'][i]=x[t]		#xの取り出し 20180113
####      net['am']['y'][i]=y[t]		#yの取り出し #      net['cell'][i]['am']['y'][0]=y[t]			#yの取り出し 
####      am.calc_AM(net['am'],i)		#連想行列Mの更新(am_kitayama.py)
######
  return #def modify_M_batch(net,x_train,y_train):

def modify_M_batch_pinv(net,x,y):
  n_channels=net['k']
#  V=[]
  V=net['V']
###(2)   if net['pinv']>0:
  rpinv=0.7 if net['Tpinv']>=0 else 1.0
  rpinv1=1.0-rpinv
  for i in xrange(net['n_cells']):
    if len(V['ij2t'][i])>0: 
      Mi=xp.dot(xp.linalg.pinv(x[V['ij2t'][i],:]),y[V['ij2t'][i],:]).T #擬似逆行列 same as GSL in 
      net['am']['M'][i]=rpinv*Mi+rpinv1*net['am']['M'][i]
#      net['am']['M'][i]=xp.dot(xp.linalg.pinv(x[V['ij2t'][i],:]),y[V['ij2t'][i],:]).T #擬似逆行列 same as GSL in 
  return #def modify_M_batch(net,x_train,y_train):


#######################################################################################
#def init_batch_wvector(NET *net, FLOAT **x, int n_train):
def init_batch_wvector(net, x, n_train): #first call by init_net_batch() beloow
  n_cells=net['n_cells']
  n_channels=net['k']
  #copy x[j] to w[i] randomely
  perm = numpy.random.permutation(n_train) 

  for i in xrange(n_cells):
    net['w'][i,:]=copy.deepcopy(x[perm[i%n_train]][0:n_channels])
#    net['w'][i,:]=x[perm[i%n_train]][0:n_channels]
#    net['cell'][i]['w']=net['w'][i,:]=x[perm[i%n_train]][0:n_channels]
#    net['w'][i,:]=x[perm[i%n_train]][0:n_channels]
  #import pdb;pdb.set_trace() #for debug 
  #check 20190115
# use below  for check4 for fn=3 with export ncells=7;it=100  
#  net['w'][0:]=x[84][0:n_channels]
#  net['w'][1:]=x[39][0:n_channels]
#  net['w'][2:]=x[79][0:n_channels]
#  net['w'][3:]=x[80][0:n_channels]
#  net['w'][4:]=x[92][0:n_channels]
#  net['w'][5:]=x[19][0:n_channels]
#  net['w'][6:]=x[33][0:n_channels]
  return 


############################再初期化を行う関数##############################
def reinit_cell_batch(net,x,y,n_train,GlobalTime,_x):
  n_cells = net['n_cells']
  n_channels = net['k']
  xi_j = 0
  reinit = 0
  n_cells=net['n_cells']
  iii=0
#  V=net['V']	 
#  checkwM('start reinit_cell_batch',net)

#アルファを大きい順に並べる
  rho=sorted(enumerate(net['alpha']),key=lambda x:x[1],reverse=True)		#アルファを大きい順に並べる。インデント番号も並
#  rho=xp.array([])
#  for i in xrange(n_cells):
#    rho=xp.append(rho,xp.linalg.norm(net['cell'][i]['alpha']))
#  rho=sorted(enumerate(rho),key=lambda x:x[1],reverse=True)		#アルファを大きい順に並べる。インデント番号も並び替える

#歪がj番目に大きいユニットとj番目に小さいユニットを用いて再初期化
  N_i = n_cells-1
  i=0
  alpha_max=max(net['alpha'])
  while i<n_cells:
    rho_i = rho[i][0]							#大きい順に並べたセル番号を代入
    i+=1
#    if net['alpha'][rho_i]> net['v_ratio'] * alpha_max:	#再初期化条件を満たすかどうか
    if net['alpha'][rho_i]> net['v_ratio'] * net['alpha_bar']:	#再初期化条件を満たすかどうか
#      import pdb;pdb.set_trace() #for debug 
      while N_i >= 0:							#セルがなくなるまで回す
        if i >= N_i:							#すべてのセルを見終わったら抜け出す
          break
        rho_N_i = rho[N_i][0]						#アルファの値が小さいものから格納
        N_i-=1
        #check whether the following line is necessary
        if net['v'][rho_N_i] >= net['v_thresh']:continue		#発火比率より多いなら無視
        iii+=1
        if iii>=net['N_reinit_max'] and iii>=2:	break			#再初期化回数MAXに達するとbreak iii>=2?不安
        #print省略 activate if necessary for check
#        import pdb;pdb.set_trace() #for debug 
        net['print']('{} #reinit w[{}]=[{:.2g},]a{:.2g} near-to w[{}]=[{:.2g},]a{:.2g};a_{:.2g}'.format(net['GlobalTime'],rho_N_i,net['w'][rho_N_i,0],net['alpha'][rho_N_i],rho_i,net['w'][rho_i,0],net['alpha'][rho_i],net['alpha_bar']))
#        print('{} #check reinit w[{}]={:.2g} a{:.3e} <= w[{}]={:.2g} a{} ;a_b{:.3e}'.format(net['GlobalTime'],rho_N_i,net['w'][rho_N_i],net['alpha'][rho_N_i],rho_i,net['w'][rho_i],net['alpha'][rho_i],net['alpha_bar']))
#        print('{}({}:{})reinit w{:%3.2f},{:%3.2f}falp{:%3.2ealp_{:%3.2e},[{:%d}]w}}'.format(GlobalTime,i,rho_N_i,net['cell'][rho_N_i]['w'][0],net['cell'][rho_N_i]['w'][1],net['cell'][rho_N_i]['alpha'],rho_i,net['cell'][rho_i]['w'][0],net['cell'][rho_i]['w'][1],net['cell'][rho_i]['alpha'],net['alpha_bar'],xi_j))
#move2x   
        V=net['V']
        min_length = 1e+20
#        if V['i2v'][i]!=len(V['ij2t'][i]):
#          print('{}!={}'.format(V['i2v'][i],len(V['ij2t'][i])))
#          import pdb;pdb.set_trace() #for debug 
        for t in V['ij2t'][rho_i]:						#訓練番号
#        for j in xrange(V['i2v0'][rho_i]):#					#セル番号rho_i番目のxの個数まで
#          t = V['ij2t'][rho_i][j]						#訓練番号
#          length=xp.linalg.norm(x[t][:n_channels]-net['w'][rho_i])
          length=xp.sum((_x[t]-net['w'][rho_i])**2)
#          length=xp.sum((x[t][:n_channels]-net['w'][rho_i])**2)
###          length=0.0
###          for k in xrange(n_channels):
###            length += (x[t][k]-net['w'][rho_i][k])**2 
###check later which is better 1e-5 or 1e-10
#          if length > 1e-5 and length < min_length:			#最小となる長さと訓練データ番号を保存(ちょうど０は考慮しない)
          if length > 1e-10 and length < min_length:			#最小となる長さと訓練データ番号を保存(ちょうど０は考慮しない)
            min_length = length
            xi_j = t          
#          if t>70:
#            print('t{} length{}=|{}-{}|,minl{},xi_j{}'.format(t,length,x[t][0],net['w'][rho_i][0],min_length,xi_j))
# 1e-5??
        
        #再初期化の更新式 
#        net['w'][rho_N_i,:] = net['w'][rho_i,:] + 1.99 * (_x[xi_j] - net['w'][rho_i,:]) 	#wの再初期化式(論文式(24))
        net['w'][rho_N_i,:] = net['w'][rho_i,:] + 1.1 * (_x[xi_j] - net['w'][rho_i,:]) 	#wの再初期化式(論文式(24))
#        net['w'][rho_N_i,:] = net['w'][rho_i,:] + 1.9 * (_x[xi_j] - net['w'][rho_i,:]) 	#wの再初期化式(論文式(24))
#        net['w'][rho_N_i,:] = net['w'][rho_i,:] + 1.8 * (_x[xi_j] - net['w'][rho_i,:]) 	#wの再初期化式(論文式(24))
#        net['w'][rho_N_i,:] = net['w'][rho_i,:] + 0.9 * (_x[xi_j] - net['w'][rho_i,:]) 	#wの再初期化式(論文式(24))
#        if GlobalTime >= 54:
#          print('x{}={}'.format(xi_j,_x[xi_j]))
#          import pdb;pdb.set_trace() #for debug 
#        net['w'][rho_N_i,:] = net['w'][rho_i,:] + 1.9 * (x[xi_j][:n_channels] - net['w'][rho_i,:]) 	#wの再初期化式(論文式(24))
#        for k in xrange(n_channels):
#          net['w'][rho_N_i,k] = net['w'][rho_i,k] + 1.9 * (x[xi_j][k] - net['w'][rho_i,k]) 	#wの再初期化式(論文式(24))

#REINITMODE==2   
#original 
#        for k in xrange(n_channels+1):
#          net['cell'][rho_N_i]['am']['M'][0][k] = net['cell'][rho_i]['am']['M'][0][k]			#Mの再初期化式)(論文式(25))
#          for j in xrange(n_channels+1):
#            if k == j:								#対角成分10000
#              net['cell'][rho_N_i]['am']['P'][k,j] = 1e4
#              net['cell'][rho_i]['am']['P'][k,j] = 1e4
#            else:
#              net['cell'][rho_N_i]['am']['P'][k,j] = 0
#              net['cell'][rho_i]['am']['P'][k,j] = 0

        net['am']['M'][rho_N_i,0] = net['am']['M'][rho_i,0]			#Mの再初期化式(論文式(25))
#        net['am']['M'][rho_N_i][0] = net['am']['M'][rho_i][0]			#Mの再初期化式(論文式(25))
        net['am']['P'][rho_N_i]=xp.identity(n_channels+1,dtype=xpfloat)*1e4		#対角成分10000
        net['am']['P'][rho_i]=xp.identity(n_channels+1,dtype=xpfloat)*1e4
        net['v0'][rho_N_i] = 0
        net['v0'][rho_i] = 0

        reinit = 1   
        break

#  checkwM('finish reinit_cell_batch',net)
  return reinit

#def reinit_cell_batch_kitayama(net,x,y,n_train,GlobalTime):
#  n_cells = net['n_cells']
#  n_channels = net['k']
#  xi_j = 0
#  reinit = 0
#  n_cells=net['n_cells']
#  iii=0
#  V=[]
#  V=net['V']	  
#
#  #アルファを大きい順に並べる
#  rho=xp.array([])
#  for i in xrange(n_cells):
#    rho=xp.append(rho,xp.linalg.norm(net['alpha'][i]))
#  rho=sorted(enumerate(rho),key=lambda x:x[1],reverse=True)		#アルファを大きい順に並べる。インデント番号も並び替える
##  import pdb;pdb.set_trace() #for debug   
#  #歪がj番目に大きいユニットとj番目に小さいユニットを用いて再初期化
#  N_i = n_cells-1
#  for i in xrange(1,n_cells):
#    rho_i = rho[i-1][0]							#大きい順に並べたセル番号を代入
#    #print 'rho{}={}'.format(i,rho[i][0])				#確認用
#    if net['alpha'][rho_i] > net['v_ratio'] * net['alpha_bar']:	#再初期化条件を満たすかどうか
#    #if n_channels==2: #テストのため一時的
#      while N_i >= 0:							#セルがなくなるまで回す
#        if i >= N_i:							#すべてのセルを見終わったら抜け出す
#          break
#        rho_N_i = rho[N_i][0]						#アルファの値が小さいものから格納
#        N_i-=1
#        if net['v'][rho_N_i] >= net['v_thresh']:		#発火比率より多いなら無視
#          continue         
#        iii+=1
#        if iii>=net['N_reinit_max'] and iii>=3:				#再初期化回数MAXに達するとbreak iii>=2?不安
#          break
#        #print省略
#
#        min_length = 1e+20
#        for j in xrange(V['i2v0'][rho_i]):					#セル番号rho_i番目のxの個数まで
#          t = V['ij2t'][rho_i][j]						#訓練番号
#          length = xp.linalg.norm(x[t][:n_channels]-net['w'][rho_i])		#wとxの距離を計算
##          length = xp.linalg.norm(x[t][:n_channels]-net['cell'][rho_i]['w'])		#wとxの距離を計算
##          if length > 1e-5 and length < min_length:				#最小となる長さと訓練データ番号を保存(ちょうど０は考慮しない)
#          if length > 1e-10 and length < min_length:				#最小となる長さと訓練データ番号を保存(ちょうど０は考慮しない)
#            min_length = length
#            xi_j = t          
#        
#        #再初期化の更新式 
#        for k in xrange(n_channels):
#          net['w'][rho_N_i,k] = net['w'][rho_i,k] + 1.9 * (x[xi_j][k] - net['w'][rho_i,k]) 	#wの再初期化式(論文式(24))
##          net['cell'][rho_N_i]['w'][k] = net['cell'][rho_i]['w'][k] + 1.9 * (x[xi_j][k] - net['cell'][rho_i]['w'][k]) 	#wの再初期化式(論文式(24))
#        
#        for k in xrange(n_channels+1):
#          net['am']['M'][rho_N_i,0,k] = net['am']['M'][rho_i,0][k]			#Mの再初期化式)(論文式(25))
#          for j in xrange(n_channels+1):
#            if k == j:								#対角成分10000
#              net['am']['P'][rho_N_i,k,j] = 1e4
#              net['am']['P'][rho_i,k,j] = 1e4
#            else:
#              net['am']['P'][rho_N_i,k,j] = 0
#              net['am']['P'][rho_i,k,j] = 0
#
#
#        net['cell'][rho_N_i]['v0'] = 0
#        net['cell'][rho_i]['v0'] = 0
#
#        reinit = 1
#        break
#
##  del rho
#  #ReinitTime = GlobalTime
#  #import pdb;pdb.set_trace() #for debug 
#  return reinit



#######################ボロノイ領域の初期化###############################
def init_Voronoi(V, n_cells, n_channels, n_train):
#void init_Voronoi(VORONOI *V, int n_cells, int n_channels, int n_train)
  V['i2v']=xp.zeros(n_cells,xp.int32)
  V['i2v0']=xp.zeros(n_cells,xp.int32)
  V['t2i']=xp.zeros(n_train,xp.int32)
  V['ij2t']={}

#  for i in xrange(n_cells):
#   V['ij2t'][i]=[]
#  V['ij2t']=xp.zeros((n_cells,n_train),xp.int32)	
#  import pdb;pdb.set_trace() #for debug 
  return #def init_Voronoi(V, n_cells, n_channels, n_train):


#######################ボロノイ領域計算を行う関数#############################
def calc_Voronoi(net,x,y,n_train,GlobalTime,_x):
  w=net['w']  #wkuro= [net['cell'][i]['w'] for i in net['cell'].keys()]
  n_cells=net['n_cells']
  n_channels=net['k']
  V=net['V']							
#発火回数を初期化  V内の訓練データ数を初期化
#original
#  c2py_ver=0 # 0 for original my_plinn.c, 1 for version of modification
#  if c2py_ver==0: #original my_plinn.c
#    for i in xrange(n_cells):
#      V['ij2t'][i]=[]
#      if V['i2v'][i] < 0:			#removed unit i2v<vmin2
#        continue
#      V['i2v'][i] = 0
#  else: #This isnot same as original my_plinn.c, but may be effective 20180113
#    V['i2v']=xp.zeros(n_cells,xp.int32) 
#修正
  V['ij2t'] = [[] for i in xrange(n_cells)]
#  c2py_ver=0 					# 0 for original my_plinn.c, 1 for version of modification
#  if c2py_ver==0: 				#original my_plinn.c
#    V['ij2t'] = [[] for i in xrange(n_cells)]
##    V['i2v'] = xp.where(V['i2v']>=0,0,V['i2v'])
#  else: 					#This isnot same as original my_plinn.c, but may be effective 20180113
#    V['i2v']=xp.zeros(n_cells,xp.int32) 

#################################################
#ボロノイ領域計算
#oroginal 
##  search_Volonoi(V,net,n_cells,n_channels,x,w,n_train)
##  Parallel(n_jobs=-1)([delayed(search_Volonoi)(V,net,n_cells,n_channels,t,x,w) for t in xrange(n_train)]) 
#  for t in xrange(n_train):
#    min_length =1e30
#    i_min_length=0
#    for i in xrange(n_cells):
#      if V['i2v'][i] < 0:        
#        continue
##      length = xp.linalg.norm(x[t][:n_channels]-w[i])				#2点間の距離
##      length = xp.sum((x[t,:n_channels]-w[i])**2)	
##      length = euclidean(x[t][:n_channels],w[i])			
#      length=0.0
#      for j in xrange(n_channels):
#        length+=(x[t][j]-w[i][j])**2
#      if length < min_length:
#       min_length =length
#       i_min_length = i    
#    V['t2i'][t]=i_min_length							#最小となるインデックス(セル番号)を格納
#    V['ij2t'][i_min_length].append(t)						#訓練データ番号を格納
##    V['ij2t'][i_min_length][V['i2v'][i_min_length]]=t				#訓練データ番号を格納
#    V['i2v'][i_min_length]+=1							#各セルの発火回数をカウント V内の訓練データ数
  
#ブロードキャスト使用 broadcast
#  import pdb;pdb.set_trace(); #for debug 
#  z=xp.asarray(xp.asnumpy(x)[:n_train,:n_channels].reshape(n_train,1,n_channels)-xp.asnumpy(w).reshape(1,n_cells,n_channels))
#  z = x[:n_train,:n_channels].reshape(n_train,1,n_channels) - w.reshape(1,n_cells,n_channels)
#  zz = xp.sum(z**2, axis=2)
  zz=xp.linalg.norm(_x.reshape(n_train,1,n_channels)-w.reshape(1,n_cells,n_channels),axis=2)
#  zz=xp.linalg.norm(_x[:n_train,:].reshape(n_train,1,n_channels)-w.reshape(1,n_cells,n_channels),axis=2)
#  zz=xp.linalg.norm(x[:n_train,:n_channels].reshape(n_train,1,n_channels)-w.reshape(1,n_cells,n_channels),axis=2)
  V['t2i']=zz.argmin(axis=1) #nearest各訓練データの最も近いセル番号記憶
  for t in xrange(n_train):
    V['ij2t'][int(V['t2i'][t])].append(t)
#  V['i2v']=xp.array([int(len(xp.where(V['t2i']==i)[0])) for i in xrange(n_cells)],xp.int32)
  V['i2v']=xp.array([len(V['ij2t'][i]) for i in xrange(n_cells)],xp.int32)
#  if min(V['i2v'])==0:
#    print('#min(V[i2v])=0')
#  import pdb;pdb.set_trace(); #for debug 
##############################################
#  prevtime = time.time()#
##remove units with v<=vmin2 from here
#  c2py_ver=1 #old 20190724kitayama using linalg.norm 
#  c2py_ver=2 # 20190727 kuro using zz
#  c2py_ver=3 # 20190727 kuro using zz
  c2py_ver=0 #20190724kitayama fast best doesnt work small n/N with vmin2
  if c2py_ver==0:#kitayama-190724 from here 
##############20190724orig==>20190724orig->new
#20190724orig->new
    Is=xp.where(V['i2v']<=net['vmin2'])[0] #indices
    Ib=xp.where(V['i2v']>net['vmin2'])[0] #indices
#??    if len(Ib)>0:      Ib=Is
#    Ib=xp.array(xp.where(V['i2v']>net['vmin2'])[0],dtype=int) #indices
#    Is=xp.where(V['i2v']<=net['vmin2'])[0].astype(xp.int64) #indices
#    Ib=xp.where(V['i2v']>net['vmin2'])[0].astype(xp.int32) #indices
#    import pdb;pdb.set_trace(); #for debug 
    if len(Ib)==0: Is=[] #do not remove cells with v<vmin2 when no-cells with v>=vmin2
    for ii in Is:
#20190724orig
#    for ii in xrange(n_cells):#20190724orig
##############
##remove unit with v<=vmin2
      iii=int(ii)
      net['n_cells2']=n_cells
      if V['i2v'][iii] <= net['vmin2']:
###compare broadcast and non-broadcast
#100 3.250e-05 2.191e-04 2.659e-04 1.793e-03 #MSEtr,MSE,NMSEtr,NMSE Learning Time = 6.12351298332  broadcast n=51 N=90
#100 3.250e-05 2.191e-04 2.659e-04 1.793e-03 #MSEtr,MSE,NMSEtr,NMSE Learning Time = 6.20609998703  no-broadcast n=51 N=90
#100 5.264e-05 6.795e-05 4.390e-04 5.668e-04 #MSEtr,MSE,NMSEtr,NMSE Learning Time = 24.414386034 broadcast n=101 N=120
#100 5.264e-05 6.795e-05 4.390e-04 5.668e-04 #MSEtr,MSE,NMSEtr,NMSE Learning Time = 23.9028010368 no-broadcast n=101 N=120
######broadcast from here
###        use_broadcast=True #False #True
###        if use_broadcast: #broadcast for vmin2
#        lcheck0=[]
#        import pdb;pdb.set_trace(); #for debug 
        lengths=xp.linalg.norm(_x[V['ij2t'][iii]].reshape(len(V['ij2t'][iii]),1,n_channels)-w[Ib].reshape(1,len(Ib),n_channels),axis=2)
        i_min_lengths=lengths.argmin(axis=1)
        for i,t in enumerate(V['ij2t'][iii]):
          i_min_length=Ib[i_min_lengths[i]]
          V['t2i'][t]=i_min_length			#２番目に近いセル番号を記憶
          V['ij2t'][i_min_length].append(t)		#そのセルに訓練データ番号を記憶
          V['i2v'][i_min_length]+=1			#各セルの発火回数をカウント
#          lcheck0.append([t,i_min_length,lengths[i,i_min_length]])
#        lcheck1=[]
###  ######broadcast to here
###  ###no-broadcast from here
###        else:
###          for t in V['ij2t'][iii]:
###  #        for j in xrange(V['i2v'][ii]):
###  #          t=V['ij2t'][int(ii)][int(j)]	
###  ##############20190724orig==>20190724orig->new
###  #20190724orig->new
###            min_length=1e30
###            i_min_length=0
###            for i in Ib:
###  #          for i in xrange(n_cells):
###  #            if i in Is: continue #modified by kuro?
###  #20190724orig
###  #            if ii==i: continue #20190724orig
###  ##############
###  #            length=xp.linalg.norm(x[t][:n_channels]-w[i],ord=2)
###  #            length=xp.linalg.norm(_x[t]-w[i],ord=2)
###  #            length = sp.sum((x[t][:n_channels]-w[i])**2) #slower?
###  ##            length = sp.sum((_x[t]-w[i])**2) #slower?
###  #
###              length=0.0 #faster?
###              for j in xrange(n_channels):
###                length+=(x[t][j]-w[i][j])**2            
###  ##            if length!=zz[t,i]:
###              if length < min_length:
###                min_length,i_min_length=length,int(i)
###            V['t2i'][t]=i_min_length					#２番目に近いセル番号を記憶
###  #          import pdb;pdb.set_trace(); #for debug 
###  #          print('i_min_length.dtype={}'.format(i_min_length.dtype))
###            V['ij2t'][i_min_length].append(t)		#そのセルに訓練データ番号を記憶
###            V['i2v'][i_min_length]+=1			#各セルの発火回数をカウント
###  #          lcheck1.append([t,i_min_length,min_length**0.5])
###no-broadcast to here
  
#        import pdb;pdb.set_trace(); #for debug 
#       print lcheck0,lcheck1
        V['i2v'][iii]=-1e3
        V['ij2t'][iii]=[] #??
        net['n_cells2']-=1
        w[iii,:]=0 #    w[ii]=0 #??
#        w[ii,:]=xp.zeros((n_channels),xpfloat) #    w[ii]=0 #??
#        net['n_cells2']-=1
#        import pdb;pdb.set_trace(); #for debug 
#kitayama-190724 to here 
  elif c2py_ver==3:#kuro 190726 from here 
#export T=100 N=100;python can2.py -fn tmp/train.csv,tmp/test.csv -k $k,0 -in $N,6,0.2,3,2,0.5,0.2 -ex 1,0.05,0.5,$T,5,50,350
#N100
#MSE=2.9126406e-04 NMSE=2.3834362e-03 MSEtr=8.2443259e-05 NMSEtr=6.7463953e-04 calc_output Time = 8.08890199661 ver=0,3
#Fake#MSE=2.1533624e-04 NMSE=1.7621130e-03 MSEtr=5.9779286e-05 NMSEtr=4.8917849e-04 calc_output Time = 7.72242307663 ver=0
#Fake#MSE=2.9126406e-04 NMSE=2.3834362e-03 MSEtr=8.2443259e-05 NMSEtr=6.7463953e-04 calc_output Time = 7.95553708076 ver=3
#Fake#MSE=2.1533624e-04 NMSE=1.7621130e-03 MSEtr=5.9779286e-05 NMSEtr=4.8917849e-04calc_output Time = 7.73744297028 ver=1
#Fake#N=200
#Fake#MSE=2.3237957e-04 NMSE=1.9015798e-03 MSEtr=4.8068622e-06 NMSEtr=3.9334923e-05 calc_output Time = 10.4000759125 ver=0
#Fake#MSE=3.0567694e-04 NMSE=2.5013779e-03 MSEtr=6.6432201e-06 NMSEtr=5.4361981e-05 calc_output Time = 10.8238310814 ver=3
#Fake#MSE=3.4607013e-04 NMSE=2.8319184e-03 MSEtr=1.1902454e-05 NMSEtr=9.7398697e-05calc_output Time = 11.0224900246 ver=2
#Fake#MSE=2.3237957e-04 NMSE=1.9015798e-03 MSEtr=4.8068622e-06 NMSEtr=3.9334923e-05 calc_output Time = 11.0686039925 ver=1
#kuro 190726 from here 
#    import pdb;pdb.set_trace(); #for debug 
#    vmin2=net['vmin2']
    Is=xp.where(V['i2v']<=net['vmin2'])[0] #indices
    if len(Is)>0:
      Ib=xp.delete(xp.array([i for i in xrange(n_cells)]),Is)
#    Ib=xp.where(V['i2v']>net['vmin2'])[0] #indices
#    import pdb;pdb.set_trace(); #for debug 
#      import pdb;pdb.set_trace(); #for debug 
      for i in Is:
#        import pdb;pdb.set_trace(); #for debug 
        for t in V['ij2t'][i]:
          ii=Ib[zz[t,Ib].argmin()]
          V['t2i'][t]=ii
          V['ij2t'][ii].append(t)
          V['i2v'][ii]+=1
#          V['ij2t'][i].remove(t) #??
        V['i2v'][i]=-1e3
        V['ij2t'][i]=[]
        w[i,:]=0
#        w[i,:]=xp.zeros((n_channels),xpfloat)
#        w[i,:]=xp.ones((n_channels),xpfloat)*1e10
#        import pdb;pdb.set_trace(); #for debug 
      net['n_cells2']=len(Ib)
#kuro 190726 to here 
#  elif c2py_ver==2:#kuro 190726 from here 
##kuro 190726 from here 
##    import pdb;pdb.set_trace(); #for debug 
#    vmin2=net['vmin2']
#    IVi2v=xp.where(V['i2v']<=vmin2)[0] #indices
##    import pdb;pdb.set_trace(); #for debug 
#    if len(IVi2v>0):
#      zzs=zz.argsort(axis=1,kind='quicksort')
#      for i in IVi2v:
#  #      import pdb;pdb.set_trace(); #for debug 
#  #      for j in xrange(len(V['ij2t'][i])):
#        for t in V['ij2t'][i]:
#  #      for j,t in enumerate(V['ij2t'][i]):
##          for i_ in xrange(len(IVi2v)+1):
#          for i_ in xrange(n_cells):
#            if not zzs[t,i_] in IVi2v:
#              ii=zzs[t,i_]
#              break
#          V['t2i'][t]=ii
#  #        print('i,ii={},{}'.format(i,ii))
#  #        import pdb;pdb.set_trace(); #for debug 
#          V['ij2t'][ii].append(t)
#          V['i2v'][ii]+=1
#          V['ij2t'][i].remove(t)
#          V['i2v'][i]=-1e3
#        w[i,:]=xp.zeros((n_channels),xpfloat)
#      net['n_cells2']=n_cells-len(IVi2v)
##kuro 190726 to here 
#kitayama-190724 from here 
  elif c2py_ver==1: #by kitayama
    for ii in xrange(n_cells):
      net['n_cells2']=n_cells
      if V['i2v'][ii] <= net['vmin2']:
        for j in xrange(V['i2v'][ii]):
          t=V['ij2t'][ii][j]	
  #        L=[]        
          L=[xp.linalg.norm(_x[t]-w[i]) for i in xrange(n_cells)]	#2点間の距離のリストを作る
#          L=[xp.linalg.norm(x[t][:n_channels]-w[i]) for i in xrange(n_cells)]	#2点間の距離のリストを作る
          L1=sorted(enumerate(L),key=lambda x:x[1])		#Lを小さい順に並べる.[(1番目に近いセル番号,Lの値),(2番目に近いセル番号,Lの値)]
  #        L1=sorted(L)
  #        L1_index = xp.argsort(L)
          
          V['t2i'][t]=L1[1][0]					#２番目に近いセル番号を記憶
          V['ij2t'][L1[1][0]].append(t)		#そのセルに訓練データ番号を記憶
  #        V['ij2t'][L1[1][0]][V['i2v'][L1[1][0]]]=t		#そのセルに訓練データ番号を記憶
          V['i2v'][L1[1][0]]+=1					#各セルの発火回数をカウント
  #        V['t2i'][t]=L1_index[1]				#２番目に近いセル番号を記憶
  #        V['ij2t'][L1_index[1]][V['i2v'][L1_index[1]]=t		#そのセルに訓練データ番号を記憶
  #        V['i2v'][L1_index[1]]+=1
  
        V['i2v'][ii]=-1e3
        V['ij2t'][ii]=[]
        net['n_cells2']-=1
        for j in xrange(n_channels): w[ii][j]=0 #    w[ii]=0 #??
        net['n_cells2']-=1
##remove units with v<=vmin2 to here
###################################################
#Complete the Input Vectors(訓練データ補充)
#  prevtime = time.time()#
  V['vmax']=0
  c2py_ver=1 #0 for original, 1 for kitayama
#  if c2py_ver==0:
#    pass
#    V['i2v0'][i]=copy.deepcopy(V['i2v'][i]) #??i is not determined
#    n1=n_channels+1
#    if V['i2v'][i] >= n1:
#      net['v0'][i]=1
#    elif net['v0'][i]==0:
#      n1=net['vmin']
#    for i in xrange(n_cells):
#      V['i2v0'][i]=copy.deepcopy(V['i2v'][i])
#      if V['i2v'][i]<= net['vmin2']:
#        pass
#      elif V['i2v'][i]<n1:
#        net['n_cells2'] -=1
#        net['v0'][i]=1
#        n0=V['i2v'][i]
#        n2=n1-n0
#        tt=0
#        d2=[xp.linalg.norm(x[t][:n_channels]-w[i]) for t in xrange(n_train)]
#        s=[t for t in xrange(n_train)]
#        for t in xrange(n2):
#          for j in xrange(t+1,n2-1):
#            if d2(s[t]) > d2[s2[j]]:
#              s[j],s[t]=s[t],s[j]
#
#        if GlobalTime > 1:
#          for j in xrange(n2):
#            ltemp=y[s[j]]+0.0
#            for k in xrange(n_channels+1):
#              ltemp -= net['am']['M'][i,0,k] * x[s[j]][k]
#            d2[s[j]]=abs(ltemp)
#
#        #補充を行う
#        for j in xrange(n10):
#          for k in xrange(j+1,n2):
#            if d2[s[j]]>d2[s[k]]:
#              s[j],s[k]=s[k],s[j]
#
#        for j in xrange(n10):
#          V['ij2t'][i].append(s[j])			#訓練データ補充 +1??
#        V['i2v'][i] =n1
#  
#      if  V['vmax'] < V['i2v'][i]:				#発火の回数が最大のセルの発火回数をVmax
#        V['vmax']=V['i2v'][i]
        
  if c2py_ver==1: #by kitayama
    V['i2v0']=copy.deepcopy(V['i2v'])	#参照渡しでいい？？？？？？	
#    import pdb;pdb.set_trace() #for debug 
    n1=net['vmin']
    for i in xrange(n_cells):
      if V['i2v'][i] <= net['vmin2']:				#vmin2以下だったら無視(何もしない)
        pass
      elif V['i2v'][i] < n1:					#セルの中のxの個数がvmin以下だったら補充
#        net['n_cells2']-=1					#いる?
        net['v0'][i]=1
        n0=V['i2v'][i]						#n0=セルの中に入っているxの個数
#kuro 190726 from here
        n2=n1 #?n1-n0?							#n2=最後の要素？not?補充しなければならない数
#original
#        n2=n1-n0							#n2=補充しなければならない数
#kuro 190726 to here
        n10=n1-n0
#        L2_index=[]
#        L=[xp.linalg.norm(x[t][:n_channels]-w[i]) for t in xrange(n_train)]	#2点間の距離のリストを作る(リスト内包表記)
#kuro 190726 broadcast
#        L_ = x[:n_train,:n_channels]-w[i] #broadcast
#        L = xp.sum(L_**2, axis=1)
#        L=xp.sum((x[:n_train,:n_channels]-w[i])**2,axis=1) #broadcast
        L=xp.linalg.norm(_x-w[i],axis=1) #broadcast
#        L=xp.linalg.norm(_x[:n_train,:]-w[i],axis=1) #broadcast
#        L=xp.linalg.norm(x[:n_train,:n_channels]-w[i],axis=1) #broadcast
        L_sort_index=L.argsort(axis=0,kind='quicksort')
#export T=100 N=100;python can2.py -fn tmp/train.csv,tmp/test.csv -k $k,0 -in $N,6,0.2,3,0,0.5,0.2 -ex 1,0.05,0.5,$T,5,50,350
#MSE=2.4461908e-04 NMSE=2.0017367e-03 MSEtr=6.4084099e-05 NMSEtr=5.2440511e-04 calc_output Time = 8.91912102699
##kitayama 190724 from here
#        L = [xp.sum((x[t][:n_channels]-w[i])**2) for t in xrange(n_train)]  
#        L_sort_index=xp.argsort(L)						#Lの小さい順のインデックスを取得      
#MSE=2.4461908e-04 NMSE=2.0017367e-03 MSEtr=6.4084099e-05 NMSEtr=5.2440511e-04 #Learning Time = 11.627918005
##kitayama 190724 from to
##        import pdb;pdb.set_trace() #for debug 
#        if xp.all(L_==L)==False:
#          import pdb;pdb.set_trace() #for debug 
#        if xp.all(L_sort_index_==L_sort_index)==False:
#          import pdb;pdb.set_trace() #for debug 
  #      L5=sorted(enumerate(L),key=lambda x:x[1])			#Lを小さい順に並べる.
  #      L2=sorted(L)							#Lを小さい順にソート
  
        #近似誤差による入力ベクトルのソート
        if GlobalTime > 1:
          for j in xrange(n2):      
#            ltemp = copy.deepcopy(y[L_sort_index[j]])
#            ltemp = y[L_sort_index[j]]+0.0 #deepcopy
#            for k in xrange(n_channels+1):
#              ltemp -= net['cell'][i]['am']['M'][0][k] * x[L_sort_index[j]][k]
            ltemp = y[L_sort_index[j]]-xp.dot(net['am']['M'][i,0],x[L_sort_index[j]])
            L[L_sort_index[j]] = abs(ltemp)
        
        for j in xrange(n10):	#ソート
          for k in xrange(j+1,n2):
            if L[L_sort_index[j]] > L[L_sort_index[k]]:
              L_sort_index[j],L_sort_index[k] = L_sort_index[k],L_sort_index[j]	#入れ替え
  
        #補充を行う
        c2py_ver=0
        if c2py_ver==0: #c2py_ver=1 #20190114 modified by kuro 違うデータを補充
          for j in xrange(n_train):   
            if len(V['ij2t'][i])>=n1:
              break
            if not L_sort_index[j] in V['ij2t'][i]:
              V['ij2t'][i].append(L_sort_index[j])			#訓練データ補充 +1??
        else: #if c2py_ver==1:#original
          for j in xrange(n10): 
            V['ij2t'][i].append(L_sort_index[j])			#訓練データ補充 +1??
    #        V['ij2t'][i][j+n0]=L_sort_index[j]			#訓練データ補充


#        V['i2v'][i] =len(V['ij2t'][i])
        V['i2v'][i] =n1
  
      if  V['vmax'] < V['i2v'][i]:				#発火の回数が最大のセルの発火回数をVmax
        V['vmax']=V['i2v'][i]

#  print V['ij2t']; import pdb;pdb.set_trace() #for debug 20191028checking 
#############################################
#  import pdb;pdb.set_trace() #for debug 
#kuro from here 190726
  net['v']=V['i2v'].astype(xpfloat)/V['vmax']
#i_firezero->i of v<=0
#  net['i_firezero']=xp.where(net['v']<=net['vmin2'])[0] #??
  net['i_firezero']=xp.where(net['v']<=0)[0]
  n_fires=n_cells-len(net['i_firezero'])
#kuro to here 190726
#  n_fires_bak=n_fires
#  netNfnbak=net['i_firezero']
###  import pdb;pdb.set_trace() #for debug 
#kitayama-orig
##orig  net['i_firezero']=[]
##orig  n_fires=0
##orig  for i in xrange(n_cells):
##orig    net['v'][i]=float(V['i2v'][i])/V['vmax']		#vmaxとの比率を求める(発火比率)
##orig    if net['v'][i] > 0:
##orig      n_fires+=1#ここで計算させる
##orig    else:
##orig      net['i_firezero'].append(i)		#発火比率が低い番号を記憶
##  import pdb;pdb.set_trace() #for debug 
#  if n_fires<n_cells:
#    import pdb;pdb.set_trace() #for debug 
#  if netNfnbak!=net['i_firezero']:
#    import pdb;pdb.set_trace() #for debug 
#  if n_fires!=n_fires_bak:
#    import pdb;pdb.set_trace() #for debug 

  net['vmax']=copy.copy(V['vmax'])						#左辺を変更すると右辺も変わる?
  net['n_fires']=n_fires

  return #def calc_Voronoi(net,x,y,n_train):




#################def init_net_batch(net,givendata['x'],n_train):###########################
def init_net_batch(net,x,n_train): #first call by exec_sim() in sim.py
  if net['winit']==0:
    init_batch_wvector(net,x,n_train)
    net['winit']=1

  if net['Vinit']==0:
    net['V']={}
    init_Voronoi(net['V'],net['n_cells'],net['k'],n_train);
#    for  i in xrange(net['n_cells']):
#      net['v0'][i]=0;
    net['v0'][:]=0 #slower?
    net['Vinit']=1

  if net['nentropy_thresh']<0:
    net['nentropy_thresh']=LA(net['n_cells'],0.75,100,0.90,500)
  net['N_reinit_max']=int(LA(net['n_cells'],0.10,100,0.30,500)*net['n_cells'])
  net['Tgamma']=5  #my_plinn.c:4385
#  import pdb;pdb.set_trace() #for debug 
  return #def init_net_batch(net,x,n_train): #first call by exec_sim() in sim.py





###################ネットの初期化を行う関数###################################
def init_net(net,args): # init_net() in my_plinn.c
  n_channels=net['k']
  inet=args.inet.split(',')
  net['n_cells']=n_cells=int(inet[0])
  net['n_compare']=int(inet[1])
  net['v_thresh']=float(inet[2])
  net['vmin']=int(inet[3])
  net['vmin2']=int(inet[4])
  net['v_ratio']=float(inet[5])
  net['width']=float(inet[6])
  net['width1']=(1.-net['width'])/2.
  net['width2']=net['width']/2.
#
#  n_cells,n_compare,v_thresh,vmin,vmin2,v_ratio,width=map(float,args.in.split(','))
##  n_cells,n_compare,v_thresh,vmin,vmin2,v_ratio,width=map(float,args.vt.split(','))
#  net['n_cells']=n_cells=int(n_cells)
#  net['n_compare']=n_compare=int(n_compare)
#  net['v_thresh']=v_thresh
#  net['vmin']=vmin=int(vmin)
#  net['vmin2']=vmin2=int(vmin2)
#  net['v_ratio']=v_ratio
#  net['width']=width
#
#  n_cells=net['n_cells']=args.n_cells
#  net['n_compare']=args.n_compare
#  net['v_thresh'],net['vmin'],net['vmin2']=map(float,args.vt.split(','))
#  net['v_ratio']=args.v_ratio
#  net['width']=args.width
  if net['n_compare'] > net['n_cells']: net['n_compare']=net['n_cells'] #add by kuro
  net['t_ri']=0 # 再初期化のための学習回数?
  net['w']=xp.zeros((n_cells,n_channels),dtype=xpfloat)
#  import pdb;pdb.set_trace(); #for debug 
#  net['dw']=xp.zeros((n_cells,n_channels),dtype=xpfloat) #--> modify_w_batch(net,x,y,n_train,GlobalTime):で定義
#  net['am']=xp.zeros((n_cells,n_channels),dtype=xpfloat)
#  net['cell']={}
  net['v']=xp.zeros((n_cells),dtype=xpfloat)
  net['v0']=xp.zeros((n_cells),dtype=xpfloat)
  net['S']=xp.zeros((n_cells),dtype=xpfloat)
  net['alpha']=xp.zeros((n_cells),dtype=xpfloat)
  net['am']={}
  am.init_AM(net['am'],n_cells,n_channels+1,1) 
#  for i in xrange(n_cells):
#    net['cell'][i]={}
#    net['cell'][i]['w']=net['w'][i] #=xp.zeros((n_cells,n_channels),dtype=xpfloat)	#?????参照渡し
#    net['cell'][i]['dw']=net['dw'][i] #=xp.zeros((n_cells,n_channels),dtype=xpfloat)	##?????参照渡し
#    net['cell'][i]['w']=net['w'][i]=xp.zeros((n_channels),dtype=xpfloat)#=xp.zeros((n_cells,n_channels),dtype=xpfloat)		#?????参照渡し
#    net['cell'][i]['dw']=net['dw'][i]=xp.zeros((n_channels),dtype=xpfloat)#=xp.zeros((n_cells,n_channels),dtype=xpfloat)	##?????参照渡し
#    net['cell'][i]['v']=0.0#=xp.zeros((n_cells,n_channels),dtype=xpfloat)
#    net['cell'][i]['S']=0.0#=xp.zeros((n_cells,n_channels),dtype=xpfloat)
#    net['cell'][i]['alpha']=0.0#xp.zeros((n_cells,n_channels),dtype=xpfloat)
#    net['cell'][i]['enough']=1.0#=xp.ones((n_cells,n_channels),dtype=xpfloat)
#    net['cell'][i]['am']={}
#    am.init_AM(net['cell'][i]['am'],n_channels+1,1)  #see am.py
  net['tau_E']=n_cells*800
  net['r_E']= math.exp(-1./net['tau_E'])
  net['nEns']=1

#  net_cell['am']=
#init_AM
  return net #def init_net(net): # init_net() in my_plinn.c


#void store_vector_batch(NET *net, FLOAT **x_train, FLOAT *y_train,int n_train, int phase){
def store_vector_batch(net, x_train, y_train,n_train, GlobalTime, phase,_x_train):#
  #coding 20180829
#  import pdb;pdb.set_trace() #for debug 
  if phase==0:
#    prevtime = time.time()
    calc_Voronoi(net,x_train,y_train,n_train,GlobalTime,_x_train)			#ボロノイ計算
#(1) simple is best
#    if GlobalTime>=50 and GlobalTime<70: 
    if net['Tpinv']>=0 and GlobalTime>=net['Tpinv']: 
      modify_M_batch_pinv(net,x_train,y_train)		#連想行列Mの更新 
      net['print']('{} #modify_M_bach_pinv'.format(GlobalTime))
    else:
      modify_M_batch_RLS(net,x_train,y_train)		#連想行列Mの更新 
      net['print']('{} #modify_M_bach_RLS'.format(GlobalTime))

#(2) NDS 
#    if net['NDS']>-0.001 and net['NDS']<0:
#      modify_M_batch_RLS(net,x_train,y_train)		#連想行列Mの更新 
#      print('{} #modify_M_bach_RLS'.format(GlobalTime))
#    else: 
#      modify_M_batch_pinv(net,x_train,y_train)		#連想行列Mの更新 
#      print('{} #modify_M_bach_pinv'.format(GlobalTime))
###(3)Tpinv80-NDS combo
##    if (net['pinvflag']==1) or (GlobalTime>=net['Tpinv']) or (GlobalTime<net['Tpinv'] and net['NDS']>6): 
##      net['pinvflag']=1
##      modify_M_batch_pinv(net,x_train,y_train)		#連想行列Mの更新 
##      print('{} #modify_M_bach_pinv'.format(GlobalTime))
##    else:
##      modify_M_batch_RLS(net,x_train,y_train)		#連想行列Mの更新 
##      print('{} #modify_M_bach_RLS'.format(GlobalTime))
###(0) original?
#    net['modify_M_batch'](net,x_train,y_train)	#連想行列Mの更新 
#    modify_M_batch(net,x_train,y_train)						#連想行列Mの更新 
#    elapsed_time = time.time() - prevtime		#計算時間
#    print 'modify_M_batch Time = {}'.format(elapsed_time)
#    prevtime = time.time()
    net['ret1']=calc_alpha(net,x_train,y_train,n_train,GlobalTime)		#αiの計算(ret=1なら再初期化すべき)
#    elapsed_time = time.time() - prevtime		#計算時間
#    print 'calc_alpha Time = {}'.format(elapsed_time)

  if phase==1:
    ret2=0
#    import pdb;pdb.set_trace() #for debug 
    if net['ret1']!=0:
#      prevtime = time.time()								#再初期化すべき
### check
##      ret2=0; print('###check the action without no_reinit') #for check no-reinit
      ret2=reinit_cell_batch(net,x_train,y_train,n_train,GlobalTime,_x_train)			#再初期化を行う
#      print('Reinit,'),
#      elapsed_time = time.time() - prevtime		#計算時間
#      print 'reinit_cell_batch Time = {}'.format(elapsed_time)
#    import pdb;pdb.set_trace() #for debug 
#    if 1==1: #net['ret1']==0 or ret2==0:
    if net['ret1']==0 or ret2==0:
#      prevtime = time.time()
#      print('##check withot modify_w') check by reducing 'width'
      modify_w_batch(net,x_train,y_train,n_train,GlobalTime,_x_train)			#再初期化しないならWの更新

#      print('Modify w,'),
#      elapsed_time = time.time() - prevtime		#計算時間
#      print 'modify_w_batch Time = {}'.format(elapsed_time)


  return #def store_vector_batch(net, x_train, y_train,n_train, phase):






