#
# Makefile
#
# For GSL do as
# make clean;make "GSL=0"
# make clean;make "GSL=1"
# make clean;make "GSL=2"
#after put source /opt/intel/bin/compilervars.sh ia32 in ~/.bashrc
#ifdef INTEL_LICENSE_FILE
ifeq ($(CC),icc)
CC	= icc  #for gcc
CXX	= icpc #for g++ after source /opt/intel/bin/compilervars.sh ia32
LM	=
else
CC	= gcc
CXX	= g++
LM	= -lm
endif
#CC	= gcc-2.95.3
#CFLAGS	= -O2 -Wall
ifdef DEBUG
DEBUG   = -g
else
DEBUG   =
endif
WA    = -Wall
WA    =
CFLAGS	= $(WA)  $(DEBUG)
#CFLAGS	= $(WA) -I./wrapplot -I./wrapunix $(DEBUG)
##LDFLAGS	= -lm -L./wrapplot -lwrapplot -L./wrapunix -lwrapunix 
LDFLAGS	= $(LM) -lgsl -lgslcblas
LDFLAGS	= 
LDLIBS	= $(LM)  -lgsl -lgslcblas
LDLIBS	= $(LM) 
DBFLAGS = $(DEBUG)
#OBJS	= can2.o my_plinn.o my_function.o my_misc.o sim.o simdata.o am.o randoms.o 
OBJS	= can2.o my_plinn.o my_function.o my_misc.o sim.o am.o randoms.o 
OPENCVLIB = -L/usr/local/lib
OPENCVINC = -I/usr/local/include/opencv
###################

TARGETS=can2 ensrs meanpred pred2y_ts
#TARGETSUbuntu1404=can2 ensrs meanpred pred2y_ts
#TARGETS=$(TARGETSUbuntu1404)
#ifeq ($(ONSEI),2012)
#echo "ONSEI=2012"
#TARGETS=$(TARGETS2012)
#else
#ifeq ($(OS),14.04.5)
#TARGETS=$(TARGETSUbuntu1404)
#else 
#TARGETS=$(TARGETSUbuntu1204)
#endif
#endif
####################

# for GSL
ifndef GSL
#20191031 for pinv in my_plinn.c
GSL = 1
endif
ifdef GSL
CFLAGSA	= $(WA) $(DEBUG) -DGSL=${GSL} -I/usr/local/gsl-1.1/include
LDFLAGS	= $(LM) $(DEBUG) -L/usr/local/gsl-1.1/lib -lgsl -lgslcblas
LDFLAGS	= 
LDLIBS	= $(LM) $(DEBUG) -L/usr/local/gsl-1.1/lib -lgsl -lgslcblas
endif
ifdef RAND
CFLAGS	= -DRAND=${RAND} $(CFLAGSA) 
else
RAND=2
CFLAGS	= -DRAND=${RAND} $(CFLAGSA) 
endif

# to display graphics for Gnuplot (ueno, Tue Feb 17 19:22:15 2004)
ifdef GPLT
DBFLAGS = $(DEBUG) -DGPLT=${GPLT}
endif

# all
all:		$(TARGETS)
#all:		lib $(TARGETS)
#all:		lib can2 rs2ge rs can2ens meanpred pred2y_ts ensrs ensrs2ge meanpredbayes ensrsbayes ens2ge concat_io1 inscr mklattice polar2rect rect2polar dataconv1 ijcnn06v2 ijcnn06mkhist ijcnn06v3 ijcnn06mkhistv3 vowelrecog dytimeseries norm_poles mabiki viewfacets ijcnn06v4 ijcnn06mkhistv4 

#time series
tsp:	lib can2 meanpred ensrs ens2ge tspredv3 mktspfigs pred2y_ts

all-clean:	data-clean clean 
		rm --force -rf result/ result-ensrs2ge/ 
#		rm --force -rf tmp/ result/ result-ensrs2ge/ 
# datas
data-clean:
		rm --force -rf *.dat *.gpl *.obj *.png *.net result/ result-ensrs2ge/ result-nn3/
#		rm --force -rf *.dat *.gpl *.obj *.png *.net tmp/ result/ result-ensrs2ge/ result-nn3/
#		rm *.dat *.gpl *.obj *.png *.net

# this program
clean:
		rm --force *.o $(TARGETS) *~ 
#		rm *~ *.o can2 rs2ge rs can2ens meanpred pred2y_ts ensrs ensrs2ge meanpredbayes ensrsbayes ens2ge concat_io1 inscr mklattice polar2rect rect2polar dataconv1 ijcnn06v2 ijcnn06mkhist ijcnn06v3 ijcnn06mkhistv3 vowelrecog dytimeseries norm_poles mabiki viewfacets ijcnn06v4 ijcnn06mkhistv4 
#                         can2 rs rs2ge can2ens meanpred pred2y_ts ensrs ensrs2ge inscr mklattice polar2rect rect2polar dataconv1 ijcnn06v2 ijcnn06mkhist ens2ge concat_io1
my_plinn.o:	my_plinn.c
		$(CC) $(CFLAGS) $(DBFLAGS) -c my_plinn.c

my_function.o:	my_function.c
		$(CC) $(CFLAGS) $(DBFLAGS) -c my_function.c

my_misc.o:	my_misc.c
		$(CC) $(CFLAGS) $(DBFLAGS) -c my_misc.c

sim.o:		sim.c
		$(CC) $(CFLAGS) $(DBFLAGS) -c sim.c

can2.o:		can2.c
		$(CC) $(CFLAGS) $(DBFLAGS) -c can2.c

am.o:		am.c
		$(CC) $(CFLAGS) $(DBFLAGS) -c am.c

randoms.o:	randoms.c
		$(CC) $(CFLAGS) $(DBFLAGS) -c randoms.c

can2:		$(OBJS)
		$(CC) $(CFLAGS) $(DBFLAGS) -o can2 $(OBJS) $(LDLIBS) 

ensrs: ensrs.c
	$(CC) $(CFLAGS) $(DBFLAGS) ensrs.c  -o ensrs $(LDLIBS)

meanpred: meanpred.c
	$(CC) $(CFLAGS) $(DBFLAGS) meanpred.c -o meanpred $(LDLIBS) -lgsl -lgslcblas

pred2y_ts: pred2y_ts.c
	$(CC) $(CFLAGS) $(DBFLAGS) pred2y_ts.c  -o pred2y_ts $(LDLIBS)
