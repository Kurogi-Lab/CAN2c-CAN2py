#ifndef __SIM_H__
#define __SIM_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "./my_function.h"
typedef struct {
  FLOAT *MSEssp;
  FLOAT *MSEmsp;
  FLOAT *MSEtrain;
  FLOAT *MSE1;
  FLOAT *MSE2;
  FLOAT *MSEdp;//for IJCNN04 (Mon Feb 16 23:49:27 2004)
  FLOAT *MSEdl;
  FLOAT *MSEds;
  FLOAT *MSEmean;
  FLOAT *Entro;
  FLOAT *VARtrain;
  FLOAT *VARssp;
  FLOAT *VARmsp;
  int   *MSEtime;
  int   *c_times;
  int   *i_times;
  int   max_i_times;
} MSEbank;

/**********************************************************************
 *
 * 関数プロトタイプ宣言
 *
 **********************************************************************/

// シミュレーションの実行
void exec_sim(NET *net, DATA *givendata, DATA *test);
//void exec_sim(NET *net, DATA *train, DATA *aprx, DATA *test, DATA *pred);
void pred_out(NET *net, DATA *train, DATA *pred);
void exec_msp_test(NET *net, DATA *train, DATA *pred,double err4terminate,double err4propagate);
void exec_plot(DATA *train, DATA *result, char *title,char *type,NET *net);
void exec_msp_train(NET *net, DATA *train, DATA *pred);//訓練データの多段予測(multistep prediction)
void exec_msp_traintest(NET *net, DATA *train, DATA *pred);//訓練データの多段予測(multistep prediction)
void exec_ssp_train(NET *net, DATA *train, DATA *pred);//訓練データの一段予測(single-step prediction)
void exec_msp_test1(NET *net, DATA *train, DATA *pred);
void exec_ssp_test(NET *net, DATA *givendata, DATA *test);
void exec_msp_test_IJCNN04_out(NET *net, DATA *givendata, DATA *test);
void exec_ssp_test_NIPS04(NET *net, DATA *givendata, DATA *test);
void exec_ssp_test_r(NET *net, DATA *givendata, DATA *test);
void exec_ssp_test_rt(NET *net, DATA *givendata, DATA *test);
int PREDTRAIN;
void exec_msp_test_ensemble(NET *net, DATA *givendata, DATA *test);
void exec_msp_test_Ensemble(NET *net, DATA *givendata, DATA *test);
void exec_ssp_test_ensemble(NET *net, DATA *givendata, DATA *test);
void search_planes_2d(NET *net, DATA *givendata, DATA *test,FLOAT t1,int maxnp,int maxoptit,int DISP);
//void exec_ssp_train_E(NET *net, DATA *givendata, DATA *test);
#endif /* __SIM_H__ */
