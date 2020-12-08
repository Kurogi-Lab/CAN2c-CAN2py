#ifndef __AM_H__
#define __AM_H__

#include <stdlib.h>
#include <math.h>
#include "my_misc.h"

/*
 * Available AM Version: 0-7
 *
 * Ver. 0:	LMS
 * Ver. 1:	RLS
 * Ver. 2:	�༡Ū���ʲ󵢥�ǥ�
 * Ver. 3:	UD�ե��륿
 * Ver. 4:	Levinson-Durbin
 * Ver. 5:	LSL
 * Ver. 6:	������LSL
 * Ver. 7:	FTF
 *
 */
#define AM_VER 1
//#define AM_VER 2
//#define AM_VER 3

//#define FLOAT double
//#define FLOAT float

/***********************************************************************
 * LMS Algorithm
 ***********************************************************************/
#if AM_VER == 0

typedef struct {
  FLOAT **M;
  FLOAT r;
  FLOAT *x;
  FLOAT *y;
  int nx;
  int ny;
} AM;

void init_AM(AM *q, int nx, int ny);
void free_AM(AM *q);
void calc_AM(AM q);


/***********************************************************************
 * RLS Algorithm
 ***********************************************************************/
#elif AM_VER == 1

typedef struct {
  FLOAT **M;
  FLOAT **P;
  FLOAT *x;
  FLOAT *y;
  int nx;
  int ny;
} AM;

void init_AMdata(AM *q);
void init_AM(AM *q, int nx, int ny);
void free_AM(AM *q);
void calc_AM(AM q);


/***********************************************************************
 * �༡Ū���ʲ󵢥�ǥ�
 * Cf. Kohonen(1977)Associative Memory,Sect.3.3.5
 * ��ë�����������ۥͥ�(1977)Ϣ�۵�����3.3.5��(pp.164-165)
 ***********************************************************************/
#elif AM_VER == 2

typedef struct {
  FLOAT **M;	// associative memory
  FLOAT **P;	//
  FLOAT **Q;	//
  FLOAT *x;	// input vector
  FLOAT *y;	// output vector
  int nx;	// dimension of x
  int ny;	// dimension of y
} AM;

void init_AMdata(AM *q);
void init_AM(AM *q, int nx, int ny);
void free_AM(AM *q);
int calc_AM(AM q);


/***********************************************************************
 * UD �ե��륿
 * ��Ԣ��Ŭ������������르�ꥺ��(2000),������,p125,or p205
 ***********************************************************************/
#elif AM_VER == 3

typedef struct {
  FLOAT **M;
  FLOAT **U;
  FLOAT *D;
  FLOAT *x;
  FLOAT *y;
  int nx;
  int ny;
} AM;

void init_AMdata(AM *q);
void init_AM(AM *q, int nx, int ny);
void free_AM(AM *q);
int calc_AM(AM q);


/***********************************************************************
 * Levinson-Durbin
 ***********************************************************************/
#elif AM_VER == 4

typedef struct {
  FLOAT **aa;	// a
  FLOAT *rho;	// sigma^2
  FLOAT *x;
  FLOAT *r;
  int p;
  int N;
} AM;

void init_AM(AM *q, int p, int N);
void free_AM(AM *q);
void calc_AM(AM q);


/***********************************************************************
 * LSL
 ***********************************************************************/
#elif AM_VER == 5

typedef struct {
  FLOAT *x;
  FLOAT *R,*Delta,*f,*r,*F;
  FLOAT *theta,*R1,*r1,*alpha,*beta;
  int p;
  int N;
  FLOAT *n;	// �����������
} AM;

void init_AM(AM *q, int p, int N);
void free_AM(AM *q);
void calc_AM(AM q);
void calc_AM0(AM q);


/***********************************************************************
 * ������LSL
 ***********************************************************************/
#elif AM_VER == 6

typedef struct {
  FLOAT *x;
  FLOAT *rho1,*rho,*f,*r,*r1;
  int p;
  int N;
} AM;

void init_AM(AM *q, int p, int N);
void free_AM(AM *q);
void calc_AM(AM q);


/***********************************************************************
 * FTF
 ***********************************************************************/
#elif AM_VER == 7

typedef struct {
  FLOAT *x,*y,*w;
  FLOAT *gp,*g,*newgp,*alpha,*beta;
  FLOAT *rr,*ff,*thetap;
  int nx;
  int ny;
  FLOAT *n;	// firing number
} AM;

void init_AM(AM *q, int nx, int ny);
void free_AM(AM *q);
void calc_AM(AM q);


/**********************************************************************/
#endif /* AM_VER */
#endif /* __AM_H__ */
