#!/usr/bin/env python
# -*- coding: utf-8 -*-
###
#chainer 3.1.0
##

#Bagging CAN2 Written by Hiromu Kitayama

import switchCupy
xp = switchCupy.xp_factory()
xpfloat=switchCupy.xpfloat(xp)
#import numpy as xp
#import cupy as xp
######

import argparse
from sklearn.datasets import load_diabetes
import chainer
import random
from chainer import cuda, Variable, optimizers, serializers
#from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import chainer.links as L
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
import re
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
#import can2_kitayama
####
ZERO=1e-20


def myshell(cmd): #no stop even when error occured

  try:
    retcode=subprocess.Popen(cmd, shell=True)

    if retcode < 0:
      print('{} {}'.format("my Child was terminated by signal", retcode))

#    else:
#      pass
#      print "my Child returned", retcode
  except OSError as e:
    print("Execution failed:{} {}".format(cmd, e))
  return retcode.wait()



##############メイン関数#############################
if __name__ == "__main__":
#  import pdb;pdb.set_trace(); #for debug 

#引数設定
  parser = argparse.ArgumentParser(description='Chainer example: MNIST')
  parser.add_argument('--gpu', '-g', default=-1, type=int,
                      help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--epochs', '-e', default=100, type=int,
                      help='number of epochs to learn')

  ### main() in main.c
  parser.add_argument('-BIAS', '-B', default=1, type=int,
                      help='Bias')
  parser.add_argument('-Lstd', '-Ls', default="0,2", type=str,
                      help='Lstd,Lsdtm')
  parser.add_argument('-ib', '-ib', default="0,0,0,0", type=str,
                      help='ib')

  ### load_data() in my_function.c
  parser.add_argument('--data_class', '-dc', default='1', type=str,
                      help='data_class 0:timeseries,1:regression')
  parser.add_argument('-k', type=str, default="2,0",
                      help='k1 and k2: number of input channels k=k1,k2=0')

  ##function_approximation
  parser.add_argument('-fn', type=str, default="",
                      help='fntrain,fntest')

  ### normalize_data(DATA *givendata, NET *net)
  parser.add_argument('--ytrans', '-y', type=str, default="0,0,0,0",
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

  ##########exec_sim
  parser.add_argument('-ex', type=str, default="1,0.005,0.7,100,5,50,350", 
                      help='l_mode,gamma0,nentropy_thresh, n-it, n-display,rot_x,rot_y')


  ##########
  parser.add_argument('--seed', '-s', default=0, type=int,
                      help='seed of random number')
  parser.add_argument('--pinv', '-pI', default=0, type=int,
                      help='1 for use pseudo-inverse')
  parser.add_argument('-Tpinv', default=1000, type=int,
                      help='Tpinv for use pseudo-inverse from t=Tpinv learning iterations.')


  parser.add_argument('-DISP', default=0, type=int,
                      help='0 for display no-figures.')
  parser.add_argument('--resume', type=str, default='log/model.npz,0',
                      help='Resume the model')


  parser.add_argument('--alpha','-a', type=float, default=0.001) #learning rate?
#  parser.add_argument('--LINESIZE','-L', type=float, default=0)
  parser.add_argument('--n_bags','-nb', type=float, default=1)
  parser.add_argument('--n','-n', type=str, default='1,10,10,1',
                      help='<0>,<N1>,<nens>,<NStep> or <1>,<N1>,<N2>,<NStep> ')
  parser.add_argument('--chkomit','-ch', type=int, default=0)
  parser.add_argument('--bagging','-bg', type=str, default='tmp/train.csv')
  parser.add_argument('--Tsk', '-Tsk', default="0,0,0,0", type=str,
                      help='<Task>[,<t1>,<t2>,<t0>] <Task>==1 for regression, ==0 for time-series [t0:t1+t0-1] for training;[t1+t0:t2+t0] for test.')
  parser.add_argument('--lossall','-lossall', type=str, default=0)
  parser.add_argument('--lcom','-lcom', type=str, default=0)
  parser.add_argument('--r0','-r0', type=int, default=0,
                      help='0 or 1')
  parser.add_argument('--ib','-ibm', type=str, default='0,0',help='-1 is NULL')#ibmode
  parser.add_argument('--LDm','-Ldm', type=str, default=2)#LDmode
  parser.add_argument('--bst','-bst', type=str, default='0,0')#boost
  parser.add_argument('--t','-t', type=str, default='result')#fn_target
  parser.add_argument('--i','-i', type=int, default='0')#intmode
  parser.add_argument('--rdm','-rdm', type=int, default='0')#rangedatamode
  parser.add_argument('--ssp','-ssp', type=int, default='0')#ssp
  parser.add_argument('-tau','-tau' , type=str, default="0,8.0,8.0,2.0",
                      help='0,tau_c,&tau_h,&eta1 or 1,tau_c,&tau_h')
  parser.add_argument('--bayes','-bayes' , type=str, default="0,0,0,0,1",
                      help='Bayes,BayesLambdaL,BayesLambdaS,BayesUseAllData,Bayesseed'),
  parser.add_argument('--nobt','-nobt', type=int, default='0')#nob_thresh
  parser.add_argument('--fupdate','-fupdate', type=str, default='1')#fupdate
  parser.add_argument('--pupdate','-pupdate', type=str, default='1')#pupdate
  parser.add_argument('--e4t','-e4t', type=float, default=0)#err4terminate
  parser.add_argument('--e4p','-e4p', type=float, default=0)#err4propagate


#############
  args=parser.parse_args()
  argv=sys.argv
  cmd=''
  for i,a in enumerate(argv):
    cmd+=a+' '
  print('Start:{}'.format(cmd))

  #初期化  
  MULTIFOLD = 1
  NoBAG = 0
  BAGwithVal = 1
  BAGwithoutVal = 2
  NoBoost = 0
  EmBoost = 1
  GbBoost = 2

  BAGGING = NoBAG;  #L177
  Boost = NoBoost;
  meannTestData = 0;
  nValData = 0;
  t_boost = 0;  #apply boosting for t_boost>=1, Gradient-based boosting for t_boost==-2
#  chkomit = 0;  #chkomit=1;
  fnsize = 256
  #char **fntrain=NULL;//fntrain[nFoldsmax][256];
  #char **fntest=NULL;// char fntest[nFoldsmax][256];
  err4propagate = 0;
  err4terminate = 0;
  nop=0;

  method=args.data_class
  nFolds=args.n_bags
  alpha=args.alpha

  if method == -1:
    nFolds=1
  elif nFolds<1 or (method==0 and alpha<1e-10):
#   usage(argc,argv);
    sys.exit()

  if method == MULTIFOLD:
    if alpha < 1: 
      alpha = 1
    nFolds1 = nFolds;
    nFolds = nFolds1*alpha;
  

#############
  N1 = 0
  N2 = 0
  ptn = 0
  _n = map(int,args.n.split(','))
  if _n[0] == 0:
    N1 = _n[1]
    nens = _n[2]
    N2 = N1 + nens*NStep-1
    NStep = _n[3]
    if NStep < 1:
      NStep = 1
  else:
    N1 = _n[1]
    N2 = _n[2]
    NStep = _n[3]

  if N1 < 1:
    print("nCells must be bigger than {}.".format(0))	#L579
    N1 = 1
    N2 = 1
    NStep = 1;

  nens = 0;
  for nCells in range(N1,N2,NStep):
    nens+=1
    NN[nens] = nCells;

##########L618
  ibg = 0
  rethresh_boost = 12.
  angedatamode = 0
  #double *cbst,*ypt,*wbst,*wbst1,
  tau_c = 4.
  tau_h = 4
  #double *cbst,*ypt,*wbst,*wbst1,
  tau_c = 8.
  tau_h = 8.
  eta1 = 2.0
#  Lstd=0
#  Lstdm=2  #double Lstd=0;int Lstdm=0;
#  lossall = NULL
  Bayes = 0
  BayesSeed = 0
  iBayes = 0
  BayesLambdaL = 0
  BayesLambdaS = 0
  BayesUseAllData = 0
  Bayesseed = 1


  _K = map(int,args.k.split(','))
  K = _K[0]
  xmin = xp.zeros(K,dtype=xpfloat)
  xmax = xp.zeros(K,dtype=xpfloat)
  xmin1 = xp.zeros(K,dtype=xpfloat)
  xmax1 = xp.zeros(K,dtype=xpfloat)
  x0m = xp.zeros(K,dtype=xpfloat)
  x1m = xp.zeros(K,dtype=xpfloat)
  x0M = xp.zeros(K,dtype=xpfloat)
  x1M = xp.zeros(K,dtype=xpfloat)
  y0m = 0.
  y1m = 0.
  y0M = 0.
  y1M = 0.

  alpha = args.alpha
  chkomit = args.chkomit

  fn_bagging = args.bagging
#  train=xp.array(pd.read_csv(fn_bagging,delim_whitespace=True,dtype=xpfloat,header=None))
  fp = open(fn_bagging,'r')
  if fn_bagging == '/dev/null':
    BAGGING = BAGwithoutVal	#bagging without validfile


  elif fp != None:
    BAGGING = BAGwithVal  #bagging with validfile
    nValData = 0
    bagging=xp.array(pd.read_csv(fp,delim_whitespace=True,dtype=xpfloat,header=None))
    nValdata = len(bagging)
    
  else:
    print("There is no bagging file {}.".format(fn_bagging))	#L669
    sys.exit()


##########
  xt=map(int,args.xtrans.split(','))   #L693
  if len(xt) == 4:
    _xm=xt[0]
    _xM=xt[1]
    _xm1=xt[2]
    _xM1=xt[3]
    xmin+=_xm	#brodcast
    xmax+=_xM
    xmax1+=_xm1
    xmin1+=_xM1
  elif len(xt) == 5:
    j = xt[0]
    _xm=xt[1]
    _xM=xt[2]
    _xm1=xt[3]
    _xM1=xt[4]
    xmin[j] = _xm
    xmax[j] = _xM
    xmin1[j] = _xm1
    xmax1[j] = _xM1
  else:
    j = xt[0]
    xmin[j] = xt[1]
    xmax[j] = xt[2]
    xmin1[j] = xt[3]
    xmax1[j] = xt[4]
    x0m[j] = xt[5]
    x1m[j] = xt[6]
    x0M[j] = xt[7]
    x1M[j] = xt[8]

  yt=map(int,args.ytrans.split(','))  #L708
  if len(yt) == 4:
    ymin = yt[0]
    ymax = yt[1]
    ymin1 = yt[2]
    ymax1 = yt[3]
  elif len(yt) == 8:
    ymin = yt[0]
    ymax = yt[1]
    ymin1 = yt[2]
    ymax1 = yt[3]
    y0m = yt[4]
    y1m = yt[5]
    y0M = yt[6]
    y1M = yt[7]
  else:
    ymin = yt[0]
    ymax = yt[1]
    ymin1 = 0
    ymax1 = 1
 

  _Tsk = map(int,re.split('[,-]',args.Tsk))  #L711
  if len(_Tsk) == 5:
    Task = _Tsk[0]
    tr0 = _Tsk[1]
    tr1 = _Tsk[2]
    tp0 = _Tsk[3]
    tp1 = _Tsk[4]
  elif len(_Tsk) == 4:
    Task = _Tsk[0]
    t1 = _Tsk[1]
    t2 = _Tsk[2]
    t0 = _Tsk[3]
  elif len(_Tsk) == 3:
    Task = _Tsk[0]
    t1 = _Tsk[1]
    t2 = _Tsk[2]
  else:
    Task = _Tsk[0]
    t1 = 0
    t2 = 0
    t0 = 0

  tr0=t0;
  tr1=t0+t1;
  tp0=t0+t1;
  tp1=t0+t2;

##############L730
  Ls = map(int,args.Lstd.split(','))
  Lstd = Ls[0]
  Lstdm = Ls[1]

  lcom = args.lcom  #lcom=0ならc言語でいうNULL
  lossall = args.lossall  #lossall=0ならc言語でいうNULL
  r0 = args.r0

  r = map(int,args.r.split(','))  
  r1 = r[0]
  r2 = r[1]
  r3 = r[2]

  inet = args.inet.split(',') 
  NC = int(inet[1])
  vt = float(inet[2])
  vm = int(inet[3])
  vr = float(inet[5])
  w = float(inet[6]) #L769

  ibm = map(int,args.ib.split(',')) 

  if ibm[0] == -1: ##ib[0]=-1ならc言語でいうNULL
    ibmode = str(-1)
  else: 
    ibmode = args.ib
  
  LDmode = args.LDm
  seed = args.seed
  
  bst = map(int,args.bst.split(',')) 
  if len(bst) >= 1:
    t_boost = bst[0]
  if len(bst) >= 2:
    Boost = bst[1]

###################L765
  fn_taret = args.t

  ex = args.ex.split(',')  
  gamma0 = float(ex[1])
  entropy_thresh = float(ex[2])

  T = args.epochs
  intmode = args.i
  #L773のNとばす
  rangedatamode = args.rdm
  ssp = args.ssp

  tau = args.tau.split(',') 
  if tau[0] ==0:
    tau_c = float(tau[1])
    tau_h = float(tau[2])
    eta1 = float(tau[3])

  else:
    tau_c = float(tau[1])
    tau_h = float(tau[2])
    eta1 = 2.0

  Bay = args.bayes.split(',') 
  BayesLambdaL = float(Bay[0])
  BayesLambdaS = float(Bay[1])
  BayesUseAllData = int(Bay[2])
  Bayesseed = int(Bay[3])

################L808
  nob_thresh = args.nobt
  BIAS = args.BIAS
  seed = args.seed
  Tpinv = args.Tpinv
  nop = args.nop
  fupdate = args.fupdate
  pupdate = args.pupdate
  err4terminate = args.e4t
  err4propagate = args.e4p

###############L821


