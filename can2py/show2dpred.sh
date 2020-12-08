#!/bin/bash
#if [ $# -le 1 ]; then
#echo "Usage: $0 <fnpred> <fntest>"
#echo " For a function y=(x1,x2), <fntest> has lines of 'x1 x2 y', and <fnpred> has 'y^'."
#exit
#fi
#
#export fnpred=$1
#export fntest=$2
if [ "$fnpred" = "" -o "fntest" = "" ]; then
  echo "Use this prog ($0) after execute 'export fnpred=<fnpred> fntest=<fntest>', where"
  echo " <fntest> has lines of 'x1 x2 y', and <fnpred> has 'y^' for a function y=(x1,x2)."
  exit
fi
#export fntest=log/test.csv
#export fnpred=log/pred.txt
mkdir -p tmp
export fnpred1=tmp/pred1.txt
export fnplt=tmp/pred.plt
#if [ "$bl" = "" ]; then export bl=`wc $fnpred |awk '{print int(sqrt($1))}'`; fi
#if [ "$bl" = "" ];
#then
#echo "Before call this prog $0, set bl, e.g. 'export bl=25'."
#else
export bl=`cat $fntest 2>/dev/null|awk '{if(NR>2 && x2!=$2){printf("%d",NR-1);exit;}else{x2=$2;}}'`
#export bl=25
#export bl=50
#echo "bl=$bl"
paste $fntest $fnpred | awk 'BEGIN{l=0;bl=ENVIRON["bl"];}{printf("%s %s %s %s\n",$1,$2,$3,$4);l++;if(l%bl==0) printf("\n");}'> $fnpred1
cat > $fnplt <<EOF
#set hidden3d;
splot "$fnpred1" using 1:2:3 w l t "y", "" using 1:2:4 w l t "yp"
pause -1 "Press return to continue."
splot "$fnpred1" using 1:2:(\$4-\$3) w l t "yp-y"
pause -1 "Press return to continue."
splot "$fnpred1" using 1:2:4 w l t "yp"
pause -1 "Press return to continue."
EOF
xterm -e gnuplot $fnplt
#fi
