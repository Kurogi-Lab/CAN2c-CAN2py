#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Chainer example: MNIST')
  parser.add_argument('-fn', default='tmp/w.csv', type=str,
                      help='w of can2')
  parser.add_argument('-n', default='0-10', type=str,
                      help='range of n')
  parser.add_argument('-type', default='w', type=str,
                      choices=('w','y'),
                      help='range of n')
  parser.add_argument('-DISP', default=1, type=int,
                      help='0 for display no-figures.')
  parser.add_argument('-msd', type=str, default='0',
                      help='nx,ny for making sample data, 0,0 for none')
#############
  args=parser.parse_args()
  fp=open('tmp/atr.csv','w')
  n1,n2=map(int,(args.n).split('-'))
  if args.type == 'w':
    w=np.array(pd.read_csv(args.fn,delim_whitespace=True,dtype=str,header=1))
#    w=np.array(pd.read_csv(args.fn,delim_whitespace=True,dtype=np.float32,header=None))
    N,k =w.shape
    if n2>N: n2=N
    if k<=2:
      ka=k
    else:
      ka=3
    for n in range(n1,n2):
      for i in range(k-ka+1):
        for ik in range(ka):
          fp.write('{} '.format(w[n,i+ik]))
        fp.write('\n')
      fp.write('\n\n')
  else:
#    import pdb;pdb.set_trace() #for debug 
    y=np.array(pd.read_csv(args.fn,delim_whitespace=True,dtype=str,header=1))[:,0]
    N=y.shape[0]
    if n2>N: n2=N
    ka=3;
    
    for t in range(n1,n2-ka):
      for i in range(ka):
        fp.write('{} '.format(y[t+i],y[t+i+1]))
      fp.write('\n')
   
  fp.close()
  fp=open('tmp/atr.plt','w')
  if ka==3:
    fp.write('splot "tmp/atr.csv" using 1:2:3 w l t "wt,wt+1,wt+2"\n')
  elif ka==2:
    fp.write('plot "tmp/atr.csv" using 1:2 w l t "wt,wt+1"\n')
  else:
    fp.write('plot "{}" using 0:1 t "wt"\n'.format(args.fn))
#    fp.write('plot "tmp/atr.csv" using 0:1 t "wt"\n')
  fp.write('pause -1 "Hit a key to quit.\n')
  fp.close()
 
  myshell('xterm -geometry 20x5-0-100 -T atractor -e gnuplot -geometry 300x240 tmp/atr.plt&')
#  import pdb;pdb.set_trace() #for debug 
  quit()

