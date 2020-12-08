#!/usr/bin/env python
# -*- coding: utf-8 -*-
###
#see sim.c
#AM_VER == 1
#Written by Hiromu Kitayama

######
#import numpy as np
#import cupy as cp
#import numpy as xp
#import cupy as xp
import switchCupy
xp=switchCupy.xp_factory()
xpfloat=switchCupy.xpfloat(xp)
######

import math
import time
import datetime
import pandas as pd #for pandas see http://keisanbutsuriya.hateblo.jp/entry/2015/12/16/161410
import copy
import my_plinn #my_plinn_kitayama
#from joblib import Parallel,delayed 

####################学習結果をチェックするための条件########################
def coud_check_learn(i_times,max_i_times,d_times):
  ret =1

#  ret = (i_times == 1 || i_times < 100 || (i_times%(max_i_times/d_times) == 0));

  return ret
#####################誤差計算############################################
def calc_MSE(t,test,net,givendata):
  test['y'][t],y,test['Y'][t]=my_plinn.calc_output(net, test['x'][t])
  test['e'][t] = (test['Y'][t]-givendata['Y'][t])
  e2=(test['e'][t])**2
  return e2

def exec_ssp_test(net, givendata, test):
  n_total=givendata['n_total']
  n_train=givendata['n_train']
  n_channels=givendata['k']
  n_channels1=n_channels+1
  
  test['x']=givendata['x']
  test['ymax']=givendata['ymax']
  test['ymin']=givendata['ymin']
################################################
#original
##  for t in xrange(n_train):
##    test['y'][t],y,test['Y'][t]=my_plinn.calc_output(net, test['x'][t])
##    test['e'][t] = (test['Y'][t]-givendata['Y'][t])
##    e2=(test['e'][t])**2
##    MSE +=e2
#  e2 = Parallel(n_jobs=4)([delayed(calc_MSE)(t,test,net,givendata) for t in xrange(n_train)])	#マルチコア化
#  MSE=xp.sum(e2)
##  for t in xrange(n_train,n_total):
#    my_plinn_kitayama.calc_output(net, test['x'][t],test[y][t],y,test['Y'][t])
##    test['y'][t],y,test['Y'][t]=my_plinn.calc_output(net, test['x'][t])
##    test['e'][t] = (test['Y'][t]-givendata['Y'][t])
##    e2=(test['e'][t])**2
##    MSE +=e2
##    fp.write('%.7e %d %.7e %.7e %.7e %.7e #Y^,t,Y,y,c,e2\n' %  
##            (test['Y'][t],t-n_train+1,givendata['Y'][t],test['y'][t], net['c'],e2))
#    fp.write('%.7e %d %.7e %.7e %.7e %2d %2d %.7e %.7e #Y^,t,Y,y,var=0,c,v,e2--#1\n' %  (test['Y'][t],t-n_train+1,givendata['Y'][t],test['y'][t], 0., net['c'], net['cell'][net['c']]['vv'],e2,0.))

##ブロードキャスト
  test['y'],y,test['Y']=my_plinn.calc_output2(net,test['x'],givendata) #test['x']=givendata['x']
  test['e'] = (test['Y'].reshape(-1,1)-givendata['Y'])
  MSE=xp.sum((test['e'][:n_train])**2) #MSEtrain
  MSEtr= MSE/givendata['n_train']
  NMSEtr= MSEtr/givendata['VARtrain']
  MSE=xp.sum((test['e'][n_train:n_total])**2) #MSEtest
  #execute the last 
  if net['savepred']==1:
    fp=open('predict.dat','w')
    for t in xrange(n_train,n_total):
      fp.write('%.7e %d %.7e %.7e %.7e %.7e #Y^,t,Y,y,c,e2\n' %  
              (test['Y'][t],t-n_train+1,givendata['Y'][t],test['y'][t], net['c'][t],(test['e'][t])**2))
    fp.close()

#  import pdb;pdb.set_trace() #for debug 
###################################
  MSE= MSE/givendata['n_test']
  NMSE= MSE/givendata['VARtrain'] 
  test['MSE'].append([MSEtr,MSE,NMSEtr,NMSE])
#  net['mes']='{:d} {:.3e} {:.3e} #MSEtr,MSE'.format(net['GlobalTime'],elapsed,MSEtr,MSE)
#  mes='{:d} {:.3e} {:.3e} {:.3e} {:.3e} #MSEtr,MSE,NMSEtr,NMSE'.format(net['GlobalTime'],MSEtr,MSE,NMSEtr,NMSE)
#  net['print'](mes)
  return #net['mes'] #def exec_ssp_test(net, givendata, test):
#  return MSEtr,MSE #net['mes'] #def exec_ssp_test(net, givendata, test):

def check_learn(simdata, net, givendata, testdata, msebank): #testdata is test originary
  exec_ssp_test(net, givendata, testdata)

#####################シミュレーションを行う##########################
def exec_sim(net, givendata,test,args):
  ex=args.ex.split(',')
  net['l_mode']=int(ex[0])
  net['gamma0']=float(ex[1])
  net['nentropy_thresh']=float(ex[2])
  net['T']=net['i_times']=int(ex[3])
  net['d_times']=int(ex[4])
  givendata['rot_x']=int(ex[5])
  givendata['rot_z']=int(ex[6])

  GlobalTimeMax =0 #??
  GlobalTimeMax += net['i_times']
  c_times = 0
  #net['cell'][i]['S']と区別のためnet['S']でなくnet['St']とする
  net['St']=xp.zeros(GlobalTimeMax+1) #sim.c:5758    
#-->learn_net(net, givendata, test) #called by exec_sim() sim.c:5769
#-->learn_net_base(NET *net, DATA *givendata, DATA *test, int mode)  #sim.c:5300 called at sim.c:5581 in lean_net() # 
  max_i_times=net['i_times']
  lbuff=max_i_times+10

#--> BATCH_MODE
##Here, ignore simdata_create() for display? ignore here, consider later
  net['St'][0]=1e+12 #sim.c:5420
#init_net_batch
  net['winit']=0
  net['Vinit']=0
  n_train=givendata['n_train']
  my_plinn.init_net_batch(net,givendata['x'],n_train) #sim.c:5423 calls init_net_batch(NET *net, FLOAT **x, int n_train) in my_plinn.c:4358
  i_times=0
  GlobalTime=0
  #初期時間
  prevtime = time.time()
  test['MSE']=[]
###
  x_train= givendata['x'][:n_train,:]
  _x_train= givendata['x'][:n_train,:givendata['k']] #_x_train= givendata['_x'][:n_train,:]
  y_train=givendata['y'][:n_train,:]
#
#CAN2による学習 learning
#  print('Start learning and prediction.')
  while True:
    GlobalTime+=1    
    i_times+=1
    net['GlobalTime']=GlobalTime
#    net['print']('{}'.format(GlobalTime)),)
#    import pdb;pdb.set_trace() #for debug 
    my_plinn.store_vector_batch(net, x_train, y_train, n_train,GlobalTime,0,_x_train) #coding 20180829 ボロノイ計算、Mの更新、エントロピー
#    my_plinn.store_vector_batch(net, givendata['x'][:n_train,:], givendata['y'][:n_train,:], n_train,GlobalTime,0,givendata['_x'][:n_train]) #coding 20180829 ボロノイ計算、Mの更新、エントロピー
#    my_plinn.store_vector_batch(net, givendata['x'], givendata['y'], n_train,GlobalTime,0,givendata['_x']) #coding 20180829 ボロノイ計算、Mの更新、エントロピー
    #学習状況のチェック
    #if coud_check_learn(i_times,max_i_times,d_times):
    #  check_learn(simdata, net, givendata, test, msebank)
      #c_times+=1
    simdata={}  #no implementation
    msebank={}
    prevtime2 = time.time()
#    net['savepred']=0 # if i_times==max_i_times else 0 
    net['savepred']=1 if i_times==max_i_times else 0 
    exec_ssp_test(net, givendata, test) #call calc_output calc MSE
#    elapsed_time2 = time.time() - prevtime		#計算時間
#    print 'calc_output Time = {}'.format(elapsed_time2)
    if i_times >= max_i_times: break
    my_plinn.store_vector_batch(net, x_train, y_train, n_train,GlobalTime,1,_x_train) #再初期化、wの更新
#    my_plinn.store_vector_batch(net, givendata['x'], givendata['y'], n_train,GlobalTime,1,givendata['_x']) #再初期化、wの更新
   
  elapsed_time = time.time() - prevtime		#計算時間
#  import pdb;pdb.set_trace() #for debug  #test['n_train'],test['n_test']
  MSEtr,MSE=test['MSE'][-1][0],test['MSE'][-1][1]
  net['mes']='{:d}({:.3f}s) {:.3e} {:.3e} #ep(time),MSEtr,MSE n{},{} k{} N{} T{},{} seed{} nop{}'.format(net['GlobalTime'],elapsed_time,MSEtr,MSE,givendata['n_train'],givendata['n_test'],net['k'],net['n_cells'],net['T'],net['Tpinv'],net['seed'], net['nop'])

#  net['mes']='{} n{},{} k{} N{:} T{},{} Time{:.3f}s({:s}) seed{} nop{}'.format(net['mes'],givendata['n_train'],givendata['n_test'],net['k'],net['n_cells'],net['T'],net['Tpinv'],elapsed_time,str(datetime.timedelta(seconds=elapsed_time))[:-3],net['seed'], net['nop'])
#  print(mes)
#  print('{} n{},{} N{:} Time{:.3f}s({:s}) seed{} nop{}'.format(mes,givendata['n_train'],givendata['n_test'],net['n_cells'],elapsed_time,str(datetime.timedelta(seconds=elapsed_time))[:-3],net['seed'], net['nop']))
  return #def exec_sim(net, givendata,test,args):

#BATCH_MODE learn_net_base(NET *net, DATA *givendata, DATA *test, int mode) 
  #my_plinn.c:  GlobalTimeMax=GlobalTime; #in my_plinn.c
#  GlbalTime=0 #learn_net_base(NET *net, DATA *givendata, DATA *test, int mode) 
#  ++GlobalTime #

def exec_msp_test(net, givendata,test,args): 
  k=n_channels=net['k']
  k1=k+1 #net['k1'] #
  n_total=givendata['n_total']
  n_train=givendata['n_train']
  n_test=givendata['n_test']
#  tr0=givendata['tr0']
  tr1=givendata['tr1']
  tp0=givendata['tp0']
  tp1=givendata['tp1']
  tpD=givendata['tpD']
  tpG=givendata['tpG']

  test=copy.deepcopy(givendata)
  t0=n_train=givendata['n_train'] #t0=n_train indicates the first time for prediction
  test['x'][t0,:]=givendata['x'][t0,:] #no-need ? already set in load_data ?
#  test['x'][t0,:]=copy.deepcopy(givendata['x'][t0,:]) #no-need ? already set in load_data ?
#  test['x'][t0-1,0:k]=givendata['x'][t0,0:k] #no-need ? already set in load_data ?
#  test['x'][t0,0]=givendata['x'][t0,0] #no-need ? already set in load_data ?
  netc=xp.zeros((n_total),dtype=xp.int32)
  starttime = time.time()
  for t in range(t0,n_total):
#    import pdb;pdb.set_trace() #for debug 
#    test['x'][t,1:k]=test['x'][t-1,0:k-1]
    test['y'][t,0],y,test['Y'][t,0]=my_plinn.calc_output(net,test['x'][t,:],test['x'][t,:k]) 
#    yrt,yt,Yt=my_plinn.calc_output2_(net,test['x'][t,:],test['x'][t,:k]) 
#    yrt,yt,Yt=yrt[0],yt[0],Yt[0]
#    test['y'][t],y,test['Y'][t]=yrt[0],yt[0],Yt[0]
#    netc[t]=net['c']
    netc[t]=net['c'][0] #for calc_output2_
#    import pdb;pdb.set_trace() #for debug 
    # 次の時刻での入力データを準備
    if tpG != 0 and t+1 < n_total: #non recursive one-step ahead prediction
        test['x'][t+1,0]=givendata['x'][t+1,0]
    elif t+tpD+1<n_total:#recursive tpD-step ahead prediction
      if t-n_train < tr1-tp0:#use given data if exists
#    import pdb;pdb.set_trace() #for debug 
        test['x'][t+tpD+1,0]=test['y'][t,0] #for check
#        test['x'][t+tpD+1,0]=givendata['x'][t+tpD+1,0]
      else: #elif t+tpD+1<n_total:
        test['x'][t+tpD+1,0]=test['y'][t,0]
#        print test['x'][t+tpD+1,0],test['y'][t,0], test['x'][t+tpD+1,0]==test['y'][t,0]
#        import pdb;pdb.set_trace() #for debug 
#        test['x'][t+tpD+1,0]=givendata['x'][t+tpD+1,0] #for check
    if t+1 < n_total:
      test['x'][t+1,1:k]=test['x'][t,0:k-1]

#    test['x'][t+1,1:k1]=BIAS #no-need ? already set in load_data ?
  elapsed_time = time.time() - starttime
  testerr=abs(test['Y'][n_train:n_total]-givendata['Y'][n_train:n_total])
  net['hpred']=len(testerr)
  for t in range(len(testerr)):
    if testerr[t]>net['tpEy']:
      net['hpred']=t
      break
#  import pdb;pdb.set_trace() #for debug 
  net['mes']='{} H{}(Ey{}) predTime{:.3f}s'.format(net['mes'],net['hpred'],net['tpEy'],elapsed_time)
  mo=xp.concatenate((test['Y'][n_train:n_total].astype(str).reshape((-1,1)),xp.array([t for t in range(tp0+tpD,tp1+tpD)],dtype=str).reshape((-1,1))),axis=1)
  mo=xp.concatenate((mo,givendata['Y'][n_train:n_total].astype(str).reshape((-1,1))),axis=1)#pred ts, time
  mo=xp.concatenate((mo,test['y'][n_train:n_total].astype(str).reshape((-1,1))),axis=1) #given ts
  mo=xp.concatenate((mo,(test['Y'][n_train:n_total]-givendata['Y'][n_train:n_total]).astype(str).reshape((-1,1))),axis=1)#err
#  mo=xp.concatenate((mo,abs(test['Y'][n_train:n_total]-givendata['Y'][n_train:n_total]).astype(str).reshape((-1,1))),axis=1)#err
#  mo=xp.concatenate((mo,xp.zeros((n_test),dtype=str).reshape((-1,1))),axis=1) #zero ??
  mo=xp.concatenate((mo,netc[n_train:n_total].astype(str).reshape((-1,1))),axis=1)#unit number selected
  df=pd.DataFrame(mo)
  df.to_csv("msp.dat",index=False,sep=' ',header=None)
#  df=pd.DataFrame(test['x'][n_train:n_total])
#  df.to_csv("xtest.dat",index=False,sep=' ',header=None)

  
  
