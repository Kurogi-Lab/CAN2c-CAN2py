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
 * 学習モード
 */
#define ONLINE_MODE 0
#define BATCH_MODE 1

/*
 * include functions of "AM"
 * 近似手法に選択ついては am.h にあるマクロ AM_VER を変更する
 */
#include "am.h"


/*
 * CELL（UNIT）
 */
typedef struct {
  FLOAT *w;     // 荷重ベクトル，ボロノイ領域の中心ベクトル
  FLOAT *dw;    // 荷重ベクトルの変化量
  FLOAT v;      // 価値（発火回数，アクセス頻度）
  FLOAT S;      // ２乗誤差
  FLOAT alpha;  // α_i（各セルの価値）
  int enough;   // Number of Input Vectors in Voronoi Region are Enough? (for Batch Learning)
  AM am;        // 連想行列Mを含む構造体（see am.c）
  int v0; //再初期化直後。
  FLOAT vv;      // 価値（発火回数，アクセス頻度）for NIPS
  FLOAT SS;      // ２乗誤差 for NIPS
  FLOAT E;      // 誤差和
} CELL;

typedef struct {
  int      *t2i; //s[t]:x[t]に最近隣のセル番号i
  int    **ij2t; //it[i][j] i番目のセルのVi属するj番目のx[t]のt
  FLOAT **it2d2; //d2[i][t]: sum_l (w[i][l]-x[t][l])^2
  int   *i2v; //i2v[i]:i番目のセルのViに属すxの個数
  int   *i2v0; //i2v[i]:i番目のセルのViに属すxの個数
  int vmax;
} VORONOI;

/*
 * NET
 */
typedef struct {
  CELL *cell;		// セル（ユニット）構造体
  FLOAT *S;		// 各時刻のセルの平均２乗誤差和
  int l_mode;		// 学習モード（ONLINE or BATCH MODE）
  int i_times;		// 学習（繰り返し）回数
  int d_times;		// 表示回数（学習してる時に予測結果を表示）
  int k;		// 入力次数
  int k1;		// ARモデル入力次数
  int k2;		// MAモデル入力次数
  int c;		// 競合に勝ったセルのインデックス
  int l;		// 価値α_iの最小となるインデックス
  int n_cells;		// ネット全体のセル数
  int n_compare;	// 入力が来たとき比較する周囲のセル数
  int n_fires;		// 発火したセル数
  int t_ri;		// 再初期化のための学習回数
  FLOAT alpha_min;	// 価値α_iの最小値
  FLOAT alpha_bar;	// 価値α_iの平均値
  FLOAT alpha_hat;
  FLOAT sigma2_hat;
  int alpha_NG;		// 価値α_iのいずれか１つでも負数になったとき真
  FLOAT v_thresh;	// 連想記憶が最小限記憶すべきベクトルの数
  FLOAT v_ratio;	// θ_α（再初期化条件のしきい値）
  FLOAT tau_E;		// ？（再初期化で用いる定数）
  FLOAT r_E;		// 忘却係数
  FLOAT width;		// Voronoi領域の境界の幅
  FLOAT *xmin,*xmax,xwidth;
  FLOAT ymin,ymax,ywidth;
  FLOAT *xmin0,*xmax0,xwidth0;
  FLOAT ymin0,ymax0,ywidth0;
  //  FLOAT xmin1,xmax1,xwidth1;
  //  FLOAT ymin1,ymax1,ywidth1;
  FLOAT **w,**dw;
  VORONOI *V;
  int r1,r2,nr; //p1/p2がデータの分解能(p1=0→実数, p1=p2=1→整数型)
  FLOAT r3; //出力を非線形化 y:=pow(y,r3) r3 in [0,infty]
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
 * 関数プロトタイプ宣言
 *
 **********************************************************************/

// ネットを初期化する
//NET* init_net(int *mode, int n_channels);
//NET* init_net(int *mode, int n_channels,NET *net);
NET* init_net(NET *net);
void init_net_batch(NET *net, FLOAT **x, int n_train);
// ネットを廃棄する
//void remove_net(NET *net);

// ネットを再初期化する
//void reinit_net(NET **net);

// ネットの各パラメータを表示する
void show_net_parms(NET *net);

void show_weights(NET *net,int nn);

// ネットで出力を計算する
FLOAT calc_output(NET *net, FLOAT *x, 
		  FLOAT *yr,//normalized && !digitized for recursive call
		  FLOAT *y, //normalized && digitized  
		  FLOAT *Y  //!normalized && digitized for error evaluate
		  );
     //FLOAT calc_output(NET *net, FLOAT *x, FLOAT *yr);

// セル（ユニット）に記憶させる
int store_vector(NET *net, FLOAT *x, FLOAT y);

// 荷重ベクトルを取り出す
void get_w_batch(NET *net, FLOAT **w);

// 荷重ベクトルを設定する
void set_w_batch(NET *net, FLOAT **w);

// バッチ学習方式でセル（ユニット）に記憶させる
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
