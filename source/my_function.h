#ifndef __MY_FUNCTION_H__
#define __MY_FUNCTION_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "my_plinn.h"

// Data Type
#define SSP	"ssp"	
#define MSP	"msp"

//#define TRAIN	"train"	
//#define APRX	"aprx"
//#define TEST	"test"
//#define PRED	"pred"

// Function ID
#define MACKEY_GLASS	1000
#define ROSSLER		1001
#define LORENZ		1002
/*
 * TDATA - "Time Series Data"
 *
 * x[0], x[1], �� , x[t-1]
 */
//typedef struct {
//  FLOAT *x;
//  int n_total;
//  int fid;	    // function id
//} TDATA;

/*
 * DATA - "Data Set for Network (true, train, test, predict data)"
 *
 * total steps (time channels) : t (= training steps + predict steps)
 * input channels              : k
 * training steps              : m
 * predict steps               : n
 * 
 * input data         : x[t][k+1] = { x_0,  x_1,    x_2,    ��, x_(k) }
 *                                = { 1.0   x[t-1], x[t-2], ��, x[t-k] }
 * output value       : y[t]
 * output error value : e[t]
 *
 * train data         : x[t] = { x[k][], x[k+1][], ��, x[m-1][] }, y[t] and e[t]
 * predict data       : x[t] = { x[m][], x[m+1][], ��, x[t-1][] }, y[t] and e[t]
 */
typedef struct {
  FLOAT ds[5001];//data from smooth.dat
  FLOAT dr[5001];//data from smooth_.dat, or the rest of smooth.dat
  FLOAT dp[5001];//data of prediction write out to predict.dat
  FLOAT dt[5001];//data of data.txt
  FLOAT dl[5001];//data of linear approximation data.dat which is from dataconv0 or dataconv0_ with dlast
  FLOAT dd[5001];//data of data of linear approximation with data of n_train-1 and n_total+1
  int i_testblock;
  int t_t1,t_t2;
  char fname_s[256];
  char fname_r[256];
  char fname_p[256];
  char fname_g[256];
  char fname_t[256];
  char fname_d[256];
  char data_path[256];
  FLOAT MSEdp;//ueno(Mon Feb 16 23:45:09 2004)
  FLOAT MSEdl;
  FLOAT MSEds;
  FLOAT MSEmean;
} IJCNN04DATA;
typedef struct {
  char type[32];    // data type ("true", "train", "test", or "pred")
  char path[512];
  FLOAT **x;
  FLOAT *y;
  FLOAT *y0;
  FLOAT *e;
  FLOAT max;	    // max value of all data
  FLOAT min;	    // min value of all data
  FLOAT width;	    // width=max-min of all data
  FLOAT mean;	    // mean (average) of all data
  FLOAT MSE;	    // mean square error
  FLOAT MSE1;	    // mean square error
  FLOAT MSE2;	    // mean square error
  FLOAT NMSE;	    // nomarized MSE
  FLOAT MSEtr;      //training MSE 20191012
  //  FLOAT VAR;        // variance
  int block, block_begin, block_end;//ueno(Tue Feb 10 19:37:05 2004)
  int n_total;
  int k;
  int k1;
  int n_train;
  int n_test;
  int t0; //�ºݤν������
  int tr0,tr1,tp0,tp1,tpD,tpG;//time steps for training tr0-tr1, for prediction tp0-tp1, tpD delay, tpG use givendata
  int data_class; // data class TIME_SERIES FUNCTION_APPROX
  int application; // application
  FLOAT VARtest; // variance
  FLOAT VARtrain;
  FLOAT MEANtest;
  FLOAT MEANtrain;
  FLOAT *xmin,*xmax;
  FLOAT *xmin0,*xmax0;
  FLOAT ymin,ymax;

  FLOAT **X;
  FLOAT *Y;
  FLOAT *Y0;
  int i_testblock;
  IJCNN04DATA *ijcnn04data;
  int rot_x,rot_z;
  int (*printf)(char *);
} DATA;

/*
 * BDATA - "Data Set to Input Network on Batch Learning Mode"
 *
 * number of cell (unit)                     : n
 * number of input vectors belong Voronoi[i] : m[i]
 * input channels (= dim of weight vector)   : k
 * training steps                            : t
 *
 * input vector                    : x[i][j][k+1]
 * output value                    : y[i][j]
 * weight vector of net            : w[i][k+1];
 */
//typedef struct {
//  FLOAT ***x;
//  FLOAT **y;
//  FLOAT **w;
//  //FLOAT **x_init;
//  //FLOAT *y_init;
//  //int *t_init;    // NET.cell[i].w�����������x[t]�Υ���ǥå���t
//  int n;
//  int *m;
//  int *m0;
//  int k;
//  int n_total;
//} BDATA;


/**********************************************************************
 *
 * �ؿ��ץ�ȥ��������
 *
 **********************************************************************/

// ����оݤδؿ�������
//void get_function_id(int *fid, char *fname);

// �ե����뤫�������ǡ������ɤ߹���
//TDATA* init_time_data(int fid, char *fname);

// ������ǡ������˴�
//void remove_time_data(TDATA *tdata);

// �ǡ������åȤ�ɽ������
void show_data_parms(DATA *data);

// �ǡ������åȤΥѥ�᡼�����������
void get_data_parms(int *n_channels, int *n_train, int *n_pred, int n_total);

// �ǡ������åȤ�����
//DATA* init_data(FLOAT *tdata, char *type,
//		int fid, int n_channels, int n_train, int n_total);

// �ǡ������åȤ��Ѵ�����
void remove_data(DATA *data);

// �Хå����ؽ��ѤΥǡ�����ɽ��
//void show_batch_data_parms(BDATA *bdata);

// �Хå����ؽ��Ѥβٽť٥��ȥ�ν���ͤ�����
//FLOAT** init_batch_wvector(FLOAT **x, FLOAT *y,
//			  int n_cells, int n_channels, int n_train);
//void init_batch_wvector(FLOAT **w, FLOAT **x, int n_cells, int n_channels, int n_train);
// �Хå����ؽ��Ѥβٽť٥��ȥ�ν���ͤ��˴�
//void remove_batch_wvector(FLOAT **w, int n_cells);

// �Хå����ؽ��ѤΥǡ���������
//BDATA* init_batch_data(FLOAT **x, FLOAT *y, FLOAT **w,
//		       int n_cells, int n_channels, int n_train,NET *net);

// �Хå����ؽ��ѤΥǡ������˴�
//void remove_batch_data(BDATA *bdata);
//void prepare_data();
//void prepare_data(int *n_channels, 
//		    int *train_steps, 
//		    int *pred_steps, 
//		    int *total_steps,
//		    int *fid,
//		    DATA *train, 
//		    DATA *test, 
//		    DATA *pred);
void load_data(DATA *givendata, DATA *test, NET *net);
//void load_data(DATA *train, DATA *test, DATA *pred, NET *net,DATA *train0);
//void load_data(DATA *train, DATA *test, DATA *pred, NET *net);
FLOAT moverange(FLOAT x,FLOAT x0, FLOAT y0,FLOAT x1, FLOAT y1);
double BIAS;//1 default in main.c
int LINESIZE; //10240 default in main.c
#endif /* __MY_FUNCTION_H__ */
