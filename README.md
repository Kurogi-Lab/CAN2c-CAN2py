# CAN2
## What is CAN2
CAN2 is neural network learning nonlinear functions and approximate them as piecewise linear function , using  Competitive net and associative net. 

You can apply CAN2 to "Nonlinear function learning problem " ,  " voice recognition problem" , "presuming amount of rainfall"  and so on.

CAN2py is a  program whose processing speed is superior to deep learning for artificial neural network developed by our laboratory ,Kurogi Lab.

You can use C or Python for CAN2.



![image](https://user-images.githubusercontent.com/72387018/107901137-3c209080-6f86-11eb-987f-58a77fd95b7d.png)




Function approximation using CAN2(Left) and MLP|relu(Right)

![image](https://user-images.githubusercontent.com/72387018/108139650-dd812100-7103-11eb-9ef7-74fa689fe48b.png)

## Requirement
```
pip.py  
numpy
pandas
xterm 
gnuplot 
scikit-learn 
matplotlib 
python-tk
gv
gls
```

## How to use
- CAN2 C language data download  
   
```
$ git clone https://github.com/Kurogi-Lab/CAN2c-CAN2py/tree/main/can2c
$ cd can2c
```
  
  - CAN2 python data download  
   
```
$ git clone https://github.com/Kurogi-Lab/CAN2c-CAN2py/tree/main/can2py
$ cd can2py 
```
And run the list of "introduce".

## Examples
### function approximation

```
export fn=Geo1d  ntrain=1000 restest=50 extest=10 k=1;python makesamples.py -msd $fn,$ntrain,$restest,$extest
```

![kunren](https://user-images.githubusercontent.com/49471144/117872656-e71cbe00-b2d9-11eb-93fd-26f737b90826.png)

<img src="https://latex.codecogs.com/gif.latex?f(x)=\left\{\begin{matrix}&space;1.0~~~~~~~~~~~~~~~~~~~~~~~(0.0\leq&space;x\leq&space;0.2)\\&space;1.0-(x-0.2)/0.2~~(0.2\leq&space;x\leq&space;0.4)\\&space;0~~~~~~~~~~~~~~~~~~~~~~~~~(0.4\leq&space;x\leq&space;0.6)\\&space;\cos&space;5\pi&space;(x-0.7)/2.0~~(0.6\leq&space;x\leq&space;0.8)\\&space;0~~~~~~~~~~~~~~~~~~~~~~~~~(0.8\leq&space;x\leq&space;1.0)\\&space;\end{matrix}\right."/>

### learning curve

```
export fn=Geo1d  ntrain=1000 restest=50 extest=10 k=1;python makesamples.py -msd $fn,$ntrain,$restest,$extest 
export T=100 N=90 k=1 Tpinv=-1 seed=5 nop=1;python can2.py -fn tmp/train.csv,tmp/test.csv -k $k,0 -in $N,6,0.2,3,0,0.5,0.2 -ex 1,0.05,0.7,$T,5,50,350 --gpu -1 -DISP 1 -Tpinv $Tpinv -s $seed -nop $nop
```

![mse](https://user-images.githubusercontent.com/49471144/117884259-5d73ed00-b2e7-11eb-959f-e4ddae7ba969.jpg)

