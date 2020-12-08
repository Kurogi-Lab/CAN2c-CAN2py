#!/usr/bin/env python
# -*- coding: utf-8 -*-
###
#chainer 3.1.0
##



######
import switchCupy
xp = switchCupy.xp_factory()
xpfloat=switchCupy.xpfloat(xp)
#import numpy as xp
#import cupy as xp
######

import argparse
from sklearn.datasets import load_diabetes
import random

###
import csv
#import numpy as np
import pandas as pd #for pandas see http://keisanbutsuriya.hateblo.jp/entry/2015/12/16/161410
import matplotlib.pyplot as plt
import math
import time
import datetime
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import copy
import os
#
import matplotlib.pyplot as plt
import matplotlib.ticker as tick # 目盛り操作に必要なライブラリを読み込みます
import matplotlib.cm as cm
import sys
#import statistics
##
import am #am_kitayama  #importing am.py
import sim #sim_kitayama
import my_plinn #my_plinn_kitayama
import my_misc
import my_function
#import can2_kitayama
####
ZERO=1e-20

def myshell(cmd): #no stop even when error occured
  try:
    retcode=subprocess.Popen(cmd, shell=True)
#    import pdb;pdb.set_trace(); #for debug 
    if retcode < 0:
      print('{} {}'.format("my Child was terminated by signal", retcode))
#    else:
#      pass
#      print "my Child returned", retcode
  except OSError as e:
    print("Execution failed:{} {}".format(cmd, e))
  return retcode.wait()
 #   pass
#def make_sample_data(msd):
##  nx=n
##  ny=n
#
#  if msd[0]=='ax+b' or msd[0]=='1':
#    k=1
#    nx=int(msd[1])
#    a=float(msd[2])
#    b=float(msd[3])
#    xtrain=[]
#    ytrain=[]
#    xtest=[]
#    ytest=[]
#    for x in xrange(nx):
#      y=a*x+b
#      if x%2 ==0:
#        xtrain.append([x])
#        ytrain.append(y)
#      else:
#        xtest.append([x])
#        ytest.append(y)
#    xtrain=xp.array(xtrain)
#    ytrain=xp.array(ytrain).reshape(-1,1)
#    xtest=xp.array(xtest)
#    ytest=xp.array(ytest).reshape(-1,1)
#    
#  elif msd[0]=='axx+b' or msd[0]=='2':
#    k=1
#    nx=int(msd[1]) 
#    a=float(msd[2])
#    b=float(msd[3])
#    xtrain=[]
#    ytrain=[]
#    xtest=[]
#    ytest=[]
#    for x in xrange(nx):
#      y=a*x*x+b
#      if x%2 ==0:
#        xtrain.append([x])
#        ytrain.append(y)
#      else:
#        xtest.append([x])
#        ytest.append(y)
#    xtrain=xp.array(xtrain)
#    ytrain=xp.array(ytrain).reshape(-1,1)
#    xtest=xp.array(xtest)
#    ytest=xp.array(ytest).reshape(-1,1)
#
#  elif msd[0]=='geo1d' or msd[0]=='3':
#    k=1
#    nx=int(msd[1])
#    xtrain=[]
#    ytrain=[]
#    xtest=[]
#    ytest=[]
#
#
#    for X in xrange(nx):
#      x=X/float(nx)
#      if x<0.2:
#        y=1.0
#      elif x<0.4:
#        y=1.0-(x-0.2)/0.2
#      elif x<0.6:
#        y=0
#      elif x<0.8:
#        y=math.cos((x-0.7)*math.pi*5.)/2.
#      else:
#        y=0
#      if X%2 ==0:
#        xtrain.append([x])
#        ytrain.append(y)
#      else:
#        xtest.append([x])
#        ytest.append(y)
#
#
#    xtrain=xp.array(xtrain)
#    ytrain=xp.array(ytrain).reshape(-1,1)
#    xtest=xp.array(xtest)
#    ytest=xp.array(ytest).reshape(-1,1)
#
#  elif msd[0]=='uxa' or msd[0]=='4':
#    k=1
#    nx=int(msd[1])
#    ax=float(msd[2])
#    xtrain=[]
#    ytrain=[]
#    xtest=[]
#    ytest=[]
#    for X in xrange(nx):
#      x=X/float(nx)
#      if x<ax:
#        y=0.0
#      else:
#        y=1.0
#      if X%2 ==0:
#        xtrain.append([x])
#        ytrain.append(y)
#      else:
#        xtest.append([x])
#        ytest.append(y)
#    xtrain=xp.array(xtrain)
#    ytrain=xp.array(ytrain).reshape(-1,1)
#    xtest=xp.array(xtest)
#    ytest=xp.array(ytest).reshape(-1,1)
#
#  elif msd[0]=='geo2d' or msd[0]=='13':
#    k=2
#    nx1=int(msd[1])
#    nx2=nx1 #nx2=int(msd[2])
#    xtrain=[]
#    ytrain=[]
#    xtest=[]
#    ytest=[]
#    for X2 in xrange(nx2):
#      x2=float(X2)/nx2
#      for X1 in xrange(nx1):
#        x1=float(X1)/nx1
#        xi=2.1*x1-0.1
#        r=math.sqrt((xi-3./2.)**2+(x2-0.5)**2)
#        if x2-xi > 0.5:
#          y=1.0
#        elif x2-xi > 0 and x2-xi <= 0.5:
#          y=2.0*(x2-xi)
#        elif r <= 0.25:
#          y=(math.cos(4.*math.pi*r)+1.)/2.
#        else:
#          y=0.0
#        if X1%2 ==0:
#          xtrain.append([x1,x2])
#          ytrain.append(y)
#        else:
#          xtest.append([x1,x2])
#          ytest.append(y)
#    xtrain=xp.array(xtrain)
#    ytrain=xp.array(ytrain).reshape(-1,1)
#    xtest=xp.array(xtest)
#    ytest=xp.array(ytest).reshape(-1,1)
#
#  elif msd[0]=='axx+b2d' or msd[0]=='10':
#    k=2
#    nx1=int(msd[1])
#    nx2=nx1
#    a=float(msd[2])
#    b=float(msd[3])
#    xtrain=[]
#    ytrain=[]
#    xtest=[]
#    ytest=[]
#    for x1 in xrange(nx1):
#      for x2 in xrange(nx2):
#        y=a*x1*x1+b
#        if x1%2 ==0:
#          xtrain.append([x1,x2])
#          ytrain.append(y)
#        else:
#          xtest.append([x1,x2])
#          ytest.append(y)
#    xtrain=xp.array(xtrain)
#    ytrain=xp.array(ytrain).reshape(-1,1)
#    xtest=xp.array(xtest)
#    ytest=xp.array(ytest).reshape(-1,1)
#  #  import pdb;pdb.set_trace(); #for debug 
#  #  n_train=(nx/2)*(ny/2)
#  #  n_test=(nx/2)*(ny/2)
#  #  xtrain=xp.zeros((n_train,k),dtype=xpfloat)
#  #  ytrain=xp.zeros((n_train,1),dtype=xpfloat)
#  #  xtest=xp.zeros((n_test,k),dtype=xpfloat)
#  #  ytest=xp.zeros((n_test,1),dtype=xpfloat)
#  #  n_train=0
#  #  n_test=0
#  #  for y in xrange(ny):
#  #    for x in xrange(nx):
#  #      if x%2 ==0:
#  #        xtrain[n_train,0]=x #*0.01
#  #        xtrain[n_train,1]=y #*0.01
#  #        ytrain[n_train]=x*x
#  #        n_train+=1
#  #      else:
#  #        xtest[n_test,0]=x #*0.01
#  #        xtest[n_test,1]=y #*0.01
#  #        ytest[n_test]=x*x
#  #        n_test+=1
#  d='./tmp'
#  if not os.path.exists(d): os.mkdir(d)
#  xy=xp.concatenate((xtrain,ytrain),axis=1)
#  df=pd.DataFrame(xy)
#  df.to_csv("tmp/train.csv",index=False,sep=' ',header=None, float_format='%.7e')
#  xy=xp.concatenate((xtest,ytest),axis=1)
#  df=pd.DataFrame(xy)
#  df.to_csv("tmp/test.csv",index=False,sep=' ',header=None, float_format='%.7e')
#  print '######################################################'
#  print '#make_sample_data() finish! File:tmp/train.csv tmp/test.csv are saved.'
#  print 'ytrain in [{},{}], ytest in [{},{}]'.format(xp.min(ytrain),xp.max(ytrain),xp.min(ytest),xp.max(ytest))
#  print 'n_train={} n_test={}'.format(len(ytrain),len(ytest))
#  print '######################################################'
#  fp=open('tmp/traintest.plt','w')
#  if k==1 or k==2:
#    if k==1:
#      fp.write('plot "tmp/train.csv" using 1:2, "tmp/test.csv" using 1:2\n')
#    if k==2:
#      fp.write('splot "tmp/train.csv" using 1:2:3, "tmp/test.csv" using 1:2:3\n')
#    fp.write('pause -1 "Hit a key to quit.\n')
#    fp.close()
#    if args.DISP>0:
#      myshell('xterm -geometry 40x5-0-100 -T traintest -e gnuplot -geometry 300x240 tmp/traintest.plt&')
##  myshell('export fn1=tmp/train.csv fn2=tmp/test.dat;show{}d.sh'.format(givendata['k']))
#
#  #  fp.write('set term postscript eps enhanced color;set output "tmp/traintest.eps"\n')
#  #  fp.write('splot "tmp/train.csv" using 1:2:3, "tmp/test.csv" using 1:2:3;quit\n')
#  #  fp.close()
#  #  myshell('gnuplot tmp/traintest.plt;gv tmp/traintest.eps&')
#  #  import pdb;pdb.set_trace(); #for debug 
#  return #finish def make_sample_data():

def set_random_seed(seed): 
  # set Python random seed
  random.seed(seed)
  # set NumPy random seed
  xp.random.seed(seed)
  return

#def moverange(y1,y1min,y1max,y0min,y0max): #in my_plinn.c
#  div1=y1max-y1min
#  if div1>-ZERO and div1<ZERO:
#    return (y0max+y0min)/2.
#  y0=((y1)-(y1min))*((y0max)-(y0min))/div1+(y0min)
##  return max(y0min,min(y0,y0max))
#  if y0>y0max:
#    return y0max
#  elif y0<y0min:
#    return y0min
#  else:
#    return y0

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Chainer example: MNIST')
  parser.add_argument('--gpu', '-g', default=-1, type=int,
                      help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--epochs', '-e', default=100, type=int,
                      help='number of epochs to learn')
### main() in main.c
  parser.add_argument('-BIAS', '-B', default=1, type=int,
                      help='Bias')
### load_data() in my_function.c
  parser.add_argument('--data_class', '-dc', default='reg', type=str,
                      help='data_class ts:timeseries,reg:regression')
  parser.add_argument('-k', type=str, default="2,0",
                      help='k1 and k2: number of input channels k=k1,k2=0')
  ##function_approximation
  parser.add_argument('-fn', type=str, default="",
                      help='fntrain,fntest')
  parser.add_argument('-t', type=str, default='',
                      help='null for regression, t:tr0-tr1:tp0-tp1:tpD:tpG:Ey for recursive tpD-step ahead prediction with tpG=0, non-recursive one step ahead pred with tpG=1')
#                      help='t:tr0-tr1:tp0-tp1 ')
### normalize_data(DATA *givendata, NET *net)
  parser.add_argument('--ytrans', '-y', type=str, default="0,0,0,0", #default='-18.5,18.5,0.0,1.0', #
                      help='ymin0,ymax0,ymin1,ymax1 to transform y in [ymin0,ymax0] to y in [ymin1,ymax1]')
  parser.add_argument('--xtrans','-x', type=str, default="0,0,0,0",
                      help='xmin0,xmax0,xmin1,xmax1 to transform x in [xmin0,xmax0] to x in [xmin1,xmax1]')
  parser.add_argument('-r', type=str, default="0,0,0",
                      help='r1,r2 for integers r1 and r2 is the resolution, no digitization if r1=0')
  parser.add_argument('-nop', type=int, default=0,
                      help='1 for noprint 0 for print')
 ###init_net() in my_plinn.c
  parser.add_argument('--inet', '-in', type=str, default='400,6,0.2,3,0,0.5,0.2',
                      help='n-units,n_compare, v_thresh,vmin,vmin2,v_ratio,width')
#  parser.add_argument('-nc', type=str, default='10,6',
#                      help='number of cells (units)')
#  parser.add_argument('-n_cells', type=int, default=1, 
#                      help='number of cells (units)')
#  parser.add_argument('-n_compare', type=int, default=6, 
#                      help='number of neighbouring cells compare ')
#  parser.add_argument('-vt', type=str, default="0.2,3,0", 
#                      help='v_thresh,vmin,vmin2')
#  parser.add_argument('--v_ratio', '-vr', type=float, default=0.5, 
#                      help='v_ratio')
#  parser.add_argument('-width', type=float, default=0.2, 
#                      help='width')
##########exec_sim
  parser.add_argument('-ex', type=str, default="1,0.005,0.7,100,5,50,350", 
                      help='l_mode,gamma0,nentropy_thresh, n-it, n-display,rot_x,rot_y')
#
#  parser.add_argument('-lm', type=str, default="1,0.005,0.7", 
#                      help='l_mode,gamma0,nentropy_thresh')
#  parser.add_argument('-it', type=int, default=100, 
#                      help='l_times')
#  parser.add_argument('-nd', type=str, default="5,50,350", 
#                      help='num-of-display,rot_x,rot_y')
##########
  parser.add_argument('--seed', '-s', default=0, type=int,
                      help='seed of random number')
  parser.add_argument('--pinv', '-pI', default=0, type=int,
                      help='1 for use pseudo-inverse')
  parser.add_argument('-Tpinv', default=1000, type=int,
                      help='Tpinv for use pseudo-inverse from t=Tpinv learning iterations.')
  parser.add_argument('-T', default='100,50', type=str,   help='T and Tpinv')
 
#  parser.add_argument('--pinv', '-pI', default=0, type=int,
#                      help='1 for use pseudo-inverse')
  parser.add_argument('-DISP', default=0, type=int,
                      help='0 for display no-figures.')
  parser.add_argument('--resume', type=str, default='log/model.npz,0',
                      help='Resume the model')
#  parser.add_argument('-msd', type=str, default='0',
#                      help='nx,ny for making sample data, 0,0 for none')
  parser.add_argument('--alpha','-a', type=float, default=0.001) #learning rate?
#############
  args=parser.parse_args()
  argv=sys.argv
  cmd=''
  for i,a in enumerate(argv):
    cmd+=a+' '
  print('#start:python {}'.format(cmd))
  
##########################
####### Check whether using GPU or not
##########################
  if args.gpu>=0:
    import chainer
    from chainer import cuda, Variable, optimizers, serializers
    import chainer.functions  as F
    import chainer.links as L
    myshell('cat /dev/null>use_gpu.py;sleep 0')
  else:
    myshell('rm -f use_gpu.*;sleep 0');

  if (args.gpu>=0 and 'ElementwiseKernel'==dir(xp)[0]) or (args.gpu<0 and 'ElementwiseKernel'!=dir(xp)[0]):
    pass
  else:
    print('### Needs more than one trial when you change the value of --gpu <val>. Try again!')
#    import pdb;pdb.set_trace(); #for debug 
    quit()
##########################
####### Initialize
##########################
  set_random_seed(args.seed)
  givendata={}
  test={}
  net={}
###
  net['seed']=args.seed
#  net['pinv']=args.pinv
#  TTpinv=map(int,(args.TP0).split(':'))                                         
#  if len(TTpinv)==2:                                                            
#    T,Tpinv=TTpinv                                                              
#  else:                                                                         
#    T=TTpinv[0]                                                                 
#    Tpinv=T+1    
  net['Tpinv']=args.Tpinv
  net['pinvflag']=0
  net['NDS']=-0.5
  net['modify_M_batch']=my_plinn.modify_M_batch_RLS if args.pinv==0 else my_plinn.modify_M_batch_pinv
  net['nop']=args.nop
  net['print']=my_misc.print1 if args.nop==0 else my_misc.noprint
  _k=map(int,args.k.split(','))
  if len(_k)>=2:
    k1,k2=_k
  else:
    k1=_k[0]
    k2=0
  k=k1+k2
  net['k']=k
  fn=args.fn.split(',')
  if len(fn)>=2:
    fntrain,fntest=fn
  else:
    fntrain=fn[0]
    fntest='/dev/null'
  if args.t=='':
    net['data_class']='reg' #regression or function approximation
  else:
    net['data_class']='ts' #time_series
  net['fntrain']=fntrain
  net['fntest']=fntest
  net['BIAS']=args.BIAS
  net['r1'],net['r3'],net['r3']=map(int,args.r.split(','))
  net['DISP']=args.DISP
  net['ytrans']=args.ytrans
  net['xtrans']=args.xtrans
  net['t']=args.t

##########################
####### Load data (traininig and test)
##########################
  givendata,test,net=my_function.load_data(givendata,test,net) # load_data() in my_function.c
  net['print']('Finish load_data.')
##########################
####### Initialize Net
##########################
  my_plinn.init_net(net,args) # init_net() in my_plinn.c
  net['print']('Finish init_net.')
##########################
####### Execute single step prediction for training and test dataset
##########################
#  import pdb;pdb.set_trace(); #for debug 
  sim.exec_sim(net, givendata, test, args) # sim.py, exec_sim(net, givendata, test) in sim.c
#
#  import pdb;pdb.set_trace(); #for debug 
  if args.DISP>0: #disp result
    if k <= 2:
      myshell('export fntest={} fnpred=predict.dat;../sh/show{}dpred.sh&'.format(fntest,min(givendata['k'],2)))
    else:
      fp=open('tmp/predict.plt','w')
      fp.write('set grid;set title "Regression: T={} N={} seed={} Tpinv={}"\n'.format(net['i_times'],net['n_cells'],net['seed'],net['Tpinv']))
      fp.write('set term postscript eps enhanced color;set output "tmp/predict.eps"\n')
      fp.write('plot "predict.dat" using 2:3 w l t "y","" using 2:1 w l t "yp", "" using 2:($1-$3) w l t "yp-y"\n')
      fp.close()
      myshell('gnuplot tmp/predict.plt;gv tmp/predict.eps&')

  if net['data_class']=='ts':#time_series
#   test=copy.deepcopy(givendata)
   sim.exec_msp_test(net, givendata,test,args)
##
###########
##########################
####### Save results
##########################
  print(net['mes'])
  fnlst=[]
  fn='tmp/V.txt'
  fnlst.append(fn)
  fp=open(fn,'w')
  for i in xrange(net['n_cells']):
    fp.write('V{} {}'.format(i,net['V']['ij2t'][i]))
  fp.close()
##
  if args.gpu>0:
    fn='tmp/w.csv'
    df=pd.DataFrame(cuda.to_cpu(net['w']))
    df.to_csv(fn,index=False,sep=' ',header=None, float_format='%.7e')
    fnlst.append(fn)

    df=pd.DataFrame(cuda.to_cpu(net['am']['M'].reshape((net['n_cells'],net['n_channels']+1))))
    df.to_csv(fn,index=False,sep=' ',header=None, float_format='%.7e')
    fnlst.append(fn)
    df=pd.DataFrame(net['v'])
    df.to_csv(cuda.to_cpu("tmp/v.csv",index=False,sep=' ',header=None))
    fnlst.append(fn)
  else:
    fn='tmp/w.csv'
    df=pd.DataFrame(net['w'])
    df.to_csv(fn,index=False,sep=' ',header=None, float_format='%.7e')
    fnlst.append(fn)

    fn='tmp/M.csv'
    df=pd.DataFrame(net['am']['M'].reshape((net['n_cells'],net['n_channels']+1)))
    df.to_csv(fn,index=False,sep=' ',header=None, float_format='%.7e')
    fnlst.append(fn)

    df=pd.DataFrame(net['v'])
    df.to_csv("tmp/v.csv",index=False,sep=' ',header=None)
    fnlst.append(fn)

##
  fn='tmp/mse.csv'
  df=pd.DataFrame(test['MSE'])
  df.to_csv(fn,index=False,sep=' ',header=None, float_format='%.7e')
  fnlst.append(fn)
  fp=open('tmp/mse.plt','w')
  fp.write('set grid;set title "T={} N={} seed={} Tpinv={}"\n'.format(net['i_times'],net['n_cells'],net['seed'],net['Tpinv']))
  fp.write('set term postscript eps enhanced color;set output "tmp/mse.eps"\n')
  fp.write('set logscale y;set format y "%.1e"\n')
 ## fp.write('set format y"10^{%L}\n') ##
  fp.write('set ytics format "%.1t{/Symbol=12 \264}10^{%T}"\n')
  fp.write('plot "tmp/mse.csv" using 0:1 w lp t "MSEtr","" using 0:2 w lp t "MSE"\n')
  fp.write('set term postscript eps enhanced color;set output "tmp/nmse.eps"\n')
  fp.write('plot "tmp/mse.csv" using 0:3 w lp t "NMSEtr","" using 0:4 w lp t "NMSE";quit\n')
  fp.close()
  myshell('gnuplot tmp/mse.plt')
##
  if net['data_class']=='ts':#time_series
    fp=open('tmp/msp.plt','w')
    fp.write('set grid;set title "Recursive MultiStep Pred: T={} N={} seed={} Tpinv={} H={}(Ey{})"\n'.format(net['i_times'],net['n_cells'],net['seed'],net['Tpinv'],net['hpred'],net['tpEy']))
    fp.write('set term postscript eps enhanced color;set output "tmp/msp.eps"\n')
#    fp.write('set logscale y;set format y "%.1e"\n')
#    fp.write('set ytics format "%.1t{/Symbol=12 \264}10^{%T}"\n')
    fp.write('plot "msp.dat" using 2:3 w l t "y","" using 2:1 w l t "yp", "" using 2:($1-$3) w l t "yp-y"\n')
    fp.close()
    myshell('gnuplot tmp/msp.plt')
   
  if args.DISP>0:
#    myshell('gnuplot tmp/mse.plt; gv tmp/mse.eps&')
    myshell('gv tmp/mse.eps&')
    if net['data_class']=='ts':#time_series
      myshell('gv tmp/msp.eps&')

  net['print']('#saved in {}'.format(fnlst))

#  for i in xrange(net['n_cells']):net['print']('V{} {}'.format(i,net['V']['ij2t'][i]))
#  for i in xrange(net['n_cells']):net['print']('{} w{} M{}'.format(i,net['w'][i],net['cell'][i]['am']['M']))
#  for t in xrange(givendata['n_train'],givendata['n_total']): net['print']('{} x{} yr,y,Y{}'.format(t,test['x'][t],my_plinn.calc_output(net,test['x'][t])))
#  for t in xrange(len(test['MSE'])): net['print'] test['MSE'][t]
########
#  import pdb;pdb.set_trace(); #for debug 
