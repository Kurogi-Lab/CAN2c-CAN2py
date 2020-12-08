#!/usr/bin/env python
# -*- coding: utf-8 -*-
###
#chainer 3.1.0
##
import argparse
import subprocess
import numpy as np
import random
import math
import os
import pandas as pd #for pandas see http://keisanbutsuriya.hateblo.jp/entry/2015/12/16/161410

def myshell(cmd): #no stop even when error occured
  try:
    retcode=subprocess.Popen(cmd, shell=True)
    if retcode < 0:
      print "my Child was terminated by signal", -retcode
    else:
      pass
#      print "my Child returned", retcode
  except OSError as e:
    print "Execution failed:", cmd, e
  return retcode.wait()
 #   pass
def geo1d(x):
  if x<0.2:
    y=1.0
  elif x<0.4:
    y=1.0-(x-0.2)/0.2
  elif x<0.6:
    y=0
  elif x<0.8:
    y=math.cos((x-0.7)*math.pi*5.)/2.
  else:
    y=0
  return y

def make_sample_data(msd):
#  nx=n
#  ny=n

  if msd[0]=='ax+b' or msd[0]=='1':
    k=1
    nx=int(msd[1])
    a=float(msd[2])
    b=float(msd[3])
    xtrain=[]
    ytrain=[]
    xtest=[]
    ytest=[]
    for x in xrange(nx):
      y=a*x+b
      if x%2 ==0:
        xtrain.append([x])
        ytrain.append(y)
      else:
        xtest.append([x])
        ytest.append(y)
    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain).reshape(-1,1)
    xtest=np.array(xtest)
    ytest=np.array(ytest).reshape(-1,1)
    
  elif msd[0]=='axx+b' or msd[0]=='2':
    k=1
    nx=int(msd[1]) 
    a=float(msd[2])
    b=float(msd[3])
    xtrain=[]
    ytrain=[]
    xtest=[]
    ytest=[]
    for x in xrange(nx):
      y=a*x*x+b
      if x%2 ==0:
        xtrain.append([x])
        ytrain.append(y)
      else:
        xtest.append([x])
        ytest.append(y)
    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain).reshape(-1,1)
    xtest=np.array(xtest)
    ytest=np.array(ytest).reshape(-1,1)

  elif msd[0]=='Geo1d' or msd[0]=='3':
    k=1
    ntrain=int(msd[1])
    restest=int(msd[2])
    extest=int(msd[3])
    xtrain=[]
    ytrain=[]
    xtest=[]
    ytest=[]

    for n in xrange(ntrain):
      x=random.uniform(0,1)
      y=geo1d(x)
      xtrain.append([x])
      ytrain.append(y)
    xmin=-float(extest)/restest
    xmax=1-xmin
    nxmin=-extest
    nxmax=restest+extest+1
#    import pdb;pdb.set_trace(); #for debug 
    for nx in xrange(nxmin,nxmax):
      x=float(nx)/restest
      y=geo1d(x)
      xtest.append([x])
      ytest.append(y)

    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain).reshape(-1,1)
    xtest=np.array(xtest)
    ytest=np.array(ytest).reshape(-1,1)

  elif msd[0]=='geo1d' or msd[0]=='3':
    k=1
    nx=int(msd[1])
    xtrain=[]
    ytrain=[]
    xtest=[]
    ytest=[]

    for X in xrange(nx):
      x=X/float(nx)
      if x<0.2:
        y=1.0
      elif x<0.4:
        y=1.0-(x-0.2)/0.2
      elif x<0.6:
        y=0
      elif x<0.8:
        y=math.cos((x-0.7)*math.pi*5.)/2.
      else:
        y=0
      if X%2 ==0:
        xtrain.append([x])
        ytrain.append(y)
      else:
        xtest.append([x])
        ytest.append(y)


    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain).reshape(-1,1)
    xtest=np.array(xtest)
    ytest=np.array(ytest).reshape(-1,1)

  elif msd[0]=='uxa' or msd[0]=='4':
    k=1
    nx=int(msd[1])
    ax=float(msd[2])
    xtrain=[]
    ytrain=[]
    xtest=[]
    ytest=[]
    for X in xrange(nx):
      x=X/float(nx)
      if x<ax:
        y=0.0
      else:
        y=1.0
      if X%2 ==0:
        xtrain.append([x])
        ytrain.append(y)
      else:
        xtest.append([x])
        ytest.append(y)
    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain).reshape(-1,1)
    xtest=np.array(xtest)
    ytest=np.array(ytest).reshape(-1,1)

  elif msd[0]=='geo2d' or msd[0]=='13':
    k=2
    nx1=int(msd[1])
    nx2=nx1 #nx2=int(msd[2])
    xtrain=[]
    ytrain=[]
    xtest=[]
    ytest=[]
    for X2 in xrange(nx2):
      x2=float(X2)/nx2
      for X1 in xrange(nx1):
        x1=float(X1)/nx1
        xi=2.1*x1-0.1
        r=math.sqrt((xi-3./2.)**2+(x2-0.5)**2)
        if x2-xi > 0.5:
          y=1.0
        elif x2-xi > 0 and x2-xi <= 0.5:
          y=2.0*(x2-xi)
        elif r <= 0.25:
          y=(math.cos(4.*math.pi*r)+1.)/2.
        else:
          y=0.0
        if X1%2 ==0:
          xtrain.append([x1,x2])
          ytrain.append(y)
        else:
          xtest.append([x1,x2])
          ytest.append(y)
    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain).reshape(-1,1)
    xtest=np.array(xtest)
    ytest=np.array(ytest).reshape(-1,1)

  elif msd[0]=='axx+b2d' or msd[0]=='10':
    k=2
    nx1=int(msd[1])
    nx2=nx1
    a=float(msd[2])
    b=float(msd[3])
    xtrain=[]
    ytrain=[]
    xtest=[]
    ytest=[]
    for x1 in xrange(nx1):
      for x2 in xrange(nx2):
        y=a*x1*x1+b
        if x1%2 ==0:
          xtrain.append([x1,x2])
          ytrain.append(y)
        else:
          xtest.append([x1,x2])
          ytest.append(y)
    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain).reshape(-1,1)
    xtest=np.array(xtest)
    ytest=np.array(ytest).reshape(-1,1)
  #  import pdb;pdb.set_trace(); #for debug 
  #  n_train=(nx/2)*(ny/2)
  #  n_test=(nx/2)*(ny/2)
  #  xtrain=np.zeros((n_train,k),dtype=xpfloat)
  #  ytrain=np.zeros((n_train,1),dtype=xpfloat)
  #  xtest=np.zeros((n_test,k),dtype=xpfloat)
  #  ytest=np.zeros((n_test,1),dtype=xpfloat)
  #  n_train=0
  #  n_test=0
  #  for y in xrange(ny):
  #    for x in xrange(nx):
  #      if x%2 ==0:
  #        xtrain[n_train,0]=x #*0.01
  #        xtrain[n_train,1]=y #*0.01
  #        ytrain[n_train]=x*x
  #        n_train+=1
  #      else:
  #        xtest[n_test,0]=x #*0.01
  #        xtest[n_test,1]=y #*0.01
  #        ytest[n_test]=x*x
  #        n_test+=1
  d='./tmp'
  if not os.path.exists(d): os.mkdir(d)
  xy=np.concatenate((xtrain,ytrain),axis=1)
  df=pd.DataFrame(xy)
  df.to_csv("tmp/train.csv",index=False,sep=' ',header=None, float_format='%.7e')
  xy=np.concatenate((xtest,ytest),axis=1)
  df=pd.DataFrame(xy)
  df.to_csv("tmp/test.csv",index=False,sep=' ',header=None, float_format='%.7e')
  print '######################################################'
  print '#make_sample_data() finish! Files:tmp/train.csv tmp/test.csv are saved.'
  print 'ytrain in [{},{}], ytest in [{},{}]'.format(np.min(ytrain),np.max(ytrain),np.min(ytest),np.max(ytest))
  print 'n_train={} n_test={}'.format(len(ytrain),len(ytest))
  print '######################################################'

  fp=open('tmp/traintest.plt','w')
  if k==1 or k==2:
    if k==1:
      myshell('sort -g tmp/train.csv >tmp/trainsort.csv')
      fp.write('plot "tmp/trainsort.csv" using 1:2 w lp pt 6, "tmp/test.csv" using 1:2 w p pt 7 ps 0.4\n')
#      fp.write('plot "tmp/trainsort.csv" using 1:2 w p pt 6, "tmp/test.csv" using 1:2 w p pt 7 ps 0.4\n')
    if k==2:
      fp.write('splot "tmp/train.csv" using 1:2:3, "tmp/test.csv" using 1:2:3\n')
    fp.write('pause -1 "Hit a key to quit.\n')
    fp.close()
    myshell('xterm -geometry 40x5-0-100 -T traintest -e gnuplot -geometry 300x240 tmp/traintest.plt&')
#  myshell('export fn1=tmp/train.csv fn2=tmp/test.dat;show{}d.sh'.format(givendata['k']))

  #  fp.write('set term postscript eps enhanced color;set output "tmp/traintest.eps"\n')
  #  fp.write('splot "tmp/train.csv" using 1:2:3, "tmp/test.csv" using 1:2:3;quit\n')
  #  fp.close()
  #  myshell('gnuplot tmp/traintest.plt;gv tmp/traintest.eps&')
  #  import pdb;pdb.set_trace(); #for debug 
  return #finish def make_sample_data():

def set_random_seed(seed): 
  # set Python random seed
  random.seed(seed)
  # set NumPy random seed
  np.random.seed(seed)
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Chainer example: MNIST')
  parser.add_argument('--gpu', '-g', default=-1, type=int,
                      help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--seed', '-s', default=0, type=int,
                      help='seed of random number')
  parser.add_argument('-DISP', default=1, type=int,
                      help='0 for display no-figures.')
  parser.add_argument('-msd', type=str, default='0',
                      help='nx,ny for making sample data, 0,0 for none')
#############
  args=parser.parse_args()

  set_random_seed(args.seed)
  msd=args.msd.split(',')
  if msd[0]!='0':
    make_sample_data(msd)
    quit()
