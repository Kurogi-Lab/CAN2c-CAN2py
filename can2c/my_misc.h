#ifndef __MY_MISC_H__
#define __MY_MISC_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
//#include "random.h" //kuro


/*====================================================================*
 *
 * Macro Declarations
 *
 *====================================================================*/
//#define FLOAT float
#define FLOAT double
// kuro
#define scanf1(f,x) {char _b[256];fgets(_b,256,stdin);sscanf(_b,f,x);}
#define scanf2(f,x,y) {char _b[256];fgets(_b,256,stdin);sscanf(_b,f,x,y);}
#define scanf3(f,x,y,z) {char _b[256];fgets(_b,256,stdin);sscanf(_b,f,(x),(y),(z));}
#define scanf4(f,x1,x2,x3,x4) {char _b[256];fgets(_b,256,stdin);sscanf(_b,f,x1,x2,x3,x4);}
#define strx1(f,x) ({char _b[256];sprintf(_b,f,x);_b;})
#define strx2(f,x1,x2) ({char _b[256];sprintf(_b,f,x1,x2);_b;})
#define strx3(f,x1,x2,x3) ({char _b[256];sprintf(_b,f,x1,x2,x3);_b;})
#define strx4(f,x1,x2,x3,x4) ({char _b[256];sprintf(_b,f,x1,x2,x3,x4);_b;})
#define strx5(f,x1,x2,x3,x4,x5) ({char _b[512];sprintf(_b,f,x1,x2,x3,x4,x5);_b;})
#define strx6(f,x1,x2,x3,x4,x5,x6) ({char _b[512];sprintf(_b,f,x1,x2,x3,x4,x5,x6);_b;})
#define strx7(f,x1,x2,x3,x4,x5,x6,x7) ({char _b[512];sprintf(_b,f,x1,x2,x3,x4,x5,x6,x7);_b;})
#define strx8(f,x1,x2,x3,x4,x5,x6,x7,x8) ({char _b[512];sprintf(_b,f,x1,x2,x3,x4,x5,x6,x7,x8);_b;})
#define strx9(f,x1,x2,x3,x4,x5,x6,x7,x8,x9) ({char _b[512];sprintf(_b,f,x1,x2,x3,x4,x5,x6,x7,x8,x9);_b;})
#define strx10(f,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) ({char _b[512];sprintf(_b,f,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10);_b;})
#define strx11(f,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11) ({char _b[512];sprintf(_b,f,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11);_b;})

//int time_series;
#define TIME_SERIES     0
#define FUNCTION_APPROX 1
int data_class;
#define DIRECT_APPLI 0
#define IJCNN04_APPLI 1
#define RANGE_APPLI 2
int application;

/*====================================================================*
 *
 * Function Prototype Declarations
 *
 *====================================================================*/

// ファイルを開きファイルポインタを返す
FILE* open_file(char *fname, char *mode);

// ファイルを閉じファイルポインタを空にする
void close_file(FILE *fp);

// パイプを開きファイルポインタを返す
FILE* open_pipe(char *cmd, char *mode);

// パイプを閉じファイルポインタを空にする
void close_pipe(FILE *fp);

// ファイルの行数を数える
int count_file(FILE *fp);

// x[dim]とy[dim]との距離を計算する
FLOAT calc_length(FLOAT *x, FLOAT *y, int dim);
FLOAT distance2(FLOAT *x, FLOAT *y, int dim);
int GlobalTime;
int GlobalTimeMax;
int ReinitTime;
int SuccessiveLearn;
int my_malloc_total;
void *my_malloc(int size,char *mes,int s);
void my_free(void *ptr);
#include <sys/time.h>
#include <unistd.h>
//struct timeval *mytv,*mytv0,*mytv1,*mytv2,*mytv3;
struct timeval mytv[3],*mytv0,*mytv1,*mytv2,*mytv3;
void mytimer_start();
double mytimer_lap();
double mytimer_total();
char *fnbody();
//double square();double square(double x){return(x*x);}
//double square(double x){return(x*x);}
#define square(x) ((x)*(x))
//double square();

#define tv2double(tv) (tv.tv_sec+(double)tv.tv_usec/1000000.)
//#define isspace(c) (c==' ')
//#define isnump(c) ((c >= '0' && c <= '9')||c=='.'||c=='-'||c=='+'||c==' ')
#define isnump(c) ((c >= '0' && c <= '9')||c=='.'||c=='-'||c=='+'||c==' '||c=='e')
int printf1(char *);
int noprintf(char *);
#endif /* __MY_MISC__H */
