#ifndef __MY_PLINN_H__
#define __MY_PLINN_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <dirent.h>

#include <math.h>
#include "my_misc.h"

/*
 * �ؽ��⡼��
 */
#define ONLINE_MODE 0
#define BATCH_MODE 1

/*
 * include functions of "AM"
 * �����ˡ������Ĥ��Ƥ� am.h �ˤ���ޥ��� AM_VER ���ѹ�����
 */
#include "am.h"


/*
 * CELL��UNIT��
 */
typedef struct {
  FLOAT *w;     // �ٽť٥��ȥ롤�ܥ�Υ��ΰ���濴�٥��ȥ�
  FLOAT *dw;    // �ٽť٥��ȥ���Ѳ���
  FLOAT v;      // ���͡�ȯ�в���������������١�
  FLOAT S;      // �����
  FLOAT alpha;  // ��_i�ʳƥ���β��͡�
  int enough;   // Number of Input Vectors in Voronoi Region are Enough? (for Batch Learning)
  AM am;        // Ϣ�۹���M��ޤ๽¤�Ρ�see am.c��
  int v0; //�ƽ����ľ�塣
  FLOAT vv;      // ���͡�ȯ�в���������������١�for NIPS
  FLOAT SS;      // ����� for NIPS
  FLOAT E;      // ����
} CELL;

typedef struct {
  int      *t2i; //s[t]:x[t]�˺Ƕ��٤Υ����ֹ�i
  int    **ij2t; //it[i][j] i���ܤΥ����Vi°����j���ܤ�x[t]��t
  FLOAT **it2d2; //d2[i][t]: sum_l (w[i][l]-x[t][l])^2
  int   *i2v; //i2v[i]:i���ܤΥ����Vi��°��x�θĿ�
  int   *i2v0; //i2v[i]:i���ܤΥ����Vi��°��x�θĿ�
  int vmax;
} VORONOI;

/*
 * NET
 */
typedef struct {
  CELL *cell;		// ����ʥ�˥åȡ˹�¤��
  FLOAT *S;		// �ƻ���Υ����ʿ�ѣ������
  int l_mode;		// �ؽ��⡼�ɡ�ONLINE or BATCH MODE��
  int i_times;		// �ؽ��ʷ����֤��˲��
  int d_times;		// ɽ������ʳؽ����Ƥ����ͽ¬��̤�ɽ����
  int k;		// ���ϼ���
  int k1;		// AR��ǥ����ϼ���
  int k2;		// MA��ǥ����ϼ���
  int c;		// ����˾��ä�����Υ���ǥå���
  int l;		// ���ͦ�_i�κǾ��Ȥʤ륤��ǥå���
  int n_cells;		// �ͥå����ΤΥ����
  int n_compare;	// ���Ϥ��褿�Ȥ���Ӥ�����ϤΥ����
  int n_fires;		// ȯ�Ф��������
  int t_ri;		// �ƽ�����Τ���γؽ����
  FLOAT alpha_min;	// ���ͦ�_i�κǾ���
  FLOAT alpha_bar;	// ���ͦ�_i��ʿ����
  FLOAT alpha_hat;
  FLOAT sigma2_hat;
  int alpha_NG;		// ���ͦ�_i�Τ����줫���ĤǤ�����ˤʤä��Ȥ���
  FLOAT v_thresh;	// Ϣ�۵������Ǿ��µ������٤��٥��ȥ�ο�
  FLOAT v_ratio;	// ��_���ʺƽ�������Τ������͡�
  FLOAT tau_E;		// ���ʺƽ�������Ѥ��������
  FLOAT r_E;		// ˺�ѷ���
  FLOAT width;		// Voronoi�ΰ�ζ�������
  FLOAT *xmin,*xmax,xwidth;
  FLOAT ymin,ymax,ywidth;
  FLOAT *xmin0,*xmax0,xwidth0;
  FLOAT ymin0,ymax0,ywidth0;
  //  FLOAT xmin1,xmax1,xwidth1;
  //  FLOAT ymin1,ymax1,ywidth1;
  FLOAT **w,**dw;
  VORONOI *V;
  int r1,r2,nr; //p1/p2���ǡ�����ʬ��ǽ(p1=0���¿�, p1=p2=1��������)
  FLOAT r3; //���Ϥ��������� y:=pow(y,r3) r3 in [0,infty]
  FLOAT *r,r12;
  FLOAT *R,R12;
  int vmin;
  int vmin2;
  int n_cells2;
  int init;
  int Vinit;
  int winit;
  FLOAT nentropy;//Normalized Entropy
  FLOAT nentropy_thresh;//Normalized Entropy Thresh for reinit
  int N_reinit_max;//maximum number of reinit units 
  FLOAT gamma0;//
  FLOAT Tgamma;//
  int ret1;
  int vmax;
  int dc;//discontinuity?
  FLOAT cost;//for rangedata
  int nEns;//number of ensembles
  unsigned long seed;
  int GlobalTime;
  int Tpinv;
  int (*printf)(char *);
  FLOAT tpEy;
  FLOAT tpH;
  char *msg;
  int DISP;
  int nop;
} NET;

/**********************************************************************
 *
 * �ؿ��ץ�ȥ��������
 *
 **********************************************************************/

// �ͥåȤ���������
//NET* init_net(int *mode, int n_channels);
//NET* init_net(int *mode, int n_channels,NET *net);
NET* init_net(NET *net);
void init_net_batch(NET *net, FLOAT **x, int n_train);
// �ͥåȤ��Ѵ�����
//void remove_net(NET *net);

// �ͥåȤ�ƽ��������
//void reinit_net(NET **net);

// �ͥåȤγƥѥ�᡼����ɽ������
void show_net_parms(NET *net);

void show_weights(NET *net,int nn);

// �ͥåȤǽ��Ϥ�׻�����
FLOAT calc_output(NET *net, FLOAT *x, 
		  FLOAT *yr,//normalized && !digitized for recursive call
		  FLOAT *y, //normalized && digitized  
		  FLOAT *Y  //!normalized && digitized for error evaluate
		  );
     //FLOAT calc_output(NET *net, FLOAT *x, FLOAT *yr);

// ����ʥ�˥åȡˤ˵���������
int store_vector(NET *net, FLOAT *x, FLOAT y);

// �ٽť٥��ȥ����Ф�
void get_w_batch(NET *net, FLOAT **w);

// �ٽť٥��ȥ�����ꤹ��
void set_w_batch(NET *net, FLOAT **w);

// �Хå��ؽ������ǥ���ʥ�˥åȡˤ˵���������
//int store_vector_batch(NET *net, FLOAT ***x, FLOAT **y,
//			 int *n_ivectors, int i_times,int *n0_ivectors,
//			 FLOAT **train_x, FLOAT *train_y,int n_train);
void store_vector_batch(NET *net, FLOAT **x_train, FLOAT *y_train,int n_train,int phase);
int net_save(NET *net,char *fname);
int net_load(NET *net,char *fname);
FLOAT moverange(FLOAT x,FLOAT x0, FLOAT y0,FLOAT x1, FLOAT y1);
FLOAT calc_output_c(NET *net, FLOAT *x, 
		  FLOAT *yr,//normalized && !digitized for recursive call
		  FLOAT *y, //normalized && digitized  
		  FLOAT *Y  //!normalized && digitized for error evaluate
		  );
NET *net_loads(NET *net);
void calc_poles_of_M(NET *net);
void calc_poles_of_Mmean(NET *net);
void poles_of_M_shrink(NET *net);
//void search_planes(NET *net);
void save_wm(NET *net);
void save_wM(NET *net);
void save_M(NET *net);
void calc_Ainvb(FLOAT *M, FLOAT *a_data, FLOAT *b_data, int nx, int ndata);
void modify_M_batch_RLS(NET *net, FLOAT **x, FLOAT *y);
void modify_M_batch_pinv(NET *net, FLOAT **x, FLOAT *y);
#endif /* __MY_PLINN_H__ */
