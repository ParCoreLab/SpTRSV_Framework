

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <thrust/device_vector.h>
#define WARP 32

struct FeatureVect
{
	int nnzs;
	int m;
	int lvls;
	int max_nnz_pl_rw;
	int max_nnz_pl_cw;
	int max_rl;
	int max_cl;
	int max_rpl;
	int max_rl_cnt;
	int max_cl_cnt;
	int min_cl_cnt;
	int min_rl_cnt;
	float median_rpl;
	float median_rl;
	float median_cl;
	float mean_rl;
	float mean_rpl;
	float mean_nnz_pl_rw;
	float std_rpl;
	float std_nnz_pl_rw;
	float std_cl;
	float std_rl;
	float mean_max_cl_pl;
	float mean_mean_cl_pl;
	float mean_std_cl_pl;
	float mean_max_rl_pl;
	float mean_std_rl_pl;
	float mean_mean_rl_pl;
	float mean_min_rl_pl;
	float mean_median_rl_pl;
};

int Device_Allocs(int **d_ia_, int **d_ja_, double **d_vala_, int **d_ib_, int **d_jb_, double **d_valb_,
                  int **d_dpL_, int **d_dpU_, int **d_rpl_, int **d_nnz_rw_, int **d_nnz_cw_, int **d_dpL_col_, int **d_dpU_col_, 
                  int **d_rowl_L_, int **d_rowl_U_, int **d_jlev_, int **d_last_, int **ia_, int **ja_, double **vala_, 
                  int **d_rl_pl_, int **d_cl_pl_, int **d_cntr_, int **d_cntr1_, int rows, int nnz);
int Host_Allocs(int **dpL_, int **dpU_, int **dpL_col_, int **dpU_col_, int **ilev_, int rows);
void Calculate_Depn(int *d_ia, int *d_ja, int *d_dpL, int *d_ib, 
			   int *d_jb, int *d_dpL_col, int rows, double *time);
int Calc_Levels_L(int *d_dp, int *d_dpr, int *d_dpc, int *d_nnz_rw, 
						   int *d_nnz_cw, int *d_rpl, int *d_rl_pl, int *d_cl_pl, int *d_jlev, 
						   int *ilev, int *d_ib, int *d_jb, int *d_last, int *d_cntr, int *d_cntr1, int rows, double *time);
int Calc_Levels_U(int *d_dp, int *d_dpr, int *d_dpc, int *d_nnz_rw, 
               int *d_nnz_cw, int *d_rpl, int *d_rl_pl, int *d_cl_pl, int *d_jlev, 
               int *ilev, int *d_ib, int *d_jb, int *d_last, int *d_cntr, int *d_cntr1, int rows, double *time);
void Calc_Depn_LT_row(int *d_ia, int *d_ja, int *d_dpL, int rows);
void Calc_Depn_LT_col(int *d_ib, int *d_jb, int *d_dpL, int rows);
void Calc_Depn_UT_row(int *d_ia, int *d_ja, int *d_dpU, int rows);
void Calc_Depn_UT_col(int *d_ib, int *d_jb, int *d_dpU, int rows);
int Find_Levels_L(int *d_dp, int *d_dpr, int *d_dpc, int *d_nnz_rw, 
						   int *d_nnz_cw, int *d_rpl, int *d_rl_pl, int *d_cl_pl, int *d_jlev, 
						   int *ilev, int *d_ib, int *d_jb, int *d_last, int *d_cntr, int *d_cntr1, int rows);
int Find_Levels_U(int *d_dp, int *d_dpr, int *d_dpc, int *d_nnz_rw, 
						   int *d_nnz_cw, int *d_rpl, int *d_rl_pl, int *d_cl_pl, int *d_jlev, 
						   int *ilev, int *d_ib, int *d_jb, int *d_last, int *d_cntr, int *d_cntr1, int rows);
void CalcStats(FeatureVect *features, int *d_nnz_rw, int *d_nnz_cw, int *d_rowl_L, int *d_dpL_col, int *d_rpl,
               int *d_rl_pl, int *d_cl_pl, int *ilev, int *d_jlev, int lvls, int rows, double *time);
void IncrementVect(int *d_vect_src, int *d_vect_dest,  int length);
void PrintStats(FeatureVect *predFeatures);

