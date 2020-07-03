
#include "anamodgpu.h"

int Device_Allocs(int **d_ia_, int **d_ja_, double **d_vala_, int **d_ib_, int **d_jb_, double **d_valb_,
                  int **d_dpL_, int **d_dpU_, int **d_rpl_, int **d_nnz_rw_, int **d_nnz_cw_, int **d_dpL_col_, int **d_dpU_col_, 
                  int **d_rowl_L_, int **d_rowl_U_, int **d_jlev_, int **d_last_, int **ia_, int **ja_, double **vala_, 
                  int **d_rl_pl_, int **d_cl_pl_, int **d_cntr_, int **d_cntr1_, int rows, int nnz)
{
  int *d_ia, *d_ja, *d_ib, *d_jb;
  int *d_dpL, *d_dpU;
  int *d_dpL_col, *d_dpU_col;
  int *d_rowl_L, *d_rowl_U;
  int *d_jlev, *d_last;
  int *d_nnz_rw, *d_nnz_cw;
  int *d_rpl;
  int *d_rl_pl, *d_cl_pl;
  int *d_cntr, *d_cntr1;
  int *ia, *ja;
  double *vala;
  double *d_vala, *d_valb; 

  cudaError_t cudaStatus;

  ia = *ia_;
  ja = *ja_;
  vala = *vala_;

  cudaStatus = cudaMalloc((int**)&d_ia, sizeof(int)*(rows + 1));
  checkCudaErrors(cudaStatus);
 
  cudaStatus = cudaMalloc((int**)&d_ja, sizeof(int)*nnz);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((double**)&d_vala, sizeof(double)*nnz);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_ib, sizeof(int)*(nnz));
  checkCudaErrors(cudaStatus);
 
  cudaStatus = cudaMalloc((int**)&d_jb, sizeof(int)*(rows + 1));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((double**)&d_valb, sizeof(double)*nnz);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_dpL, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_dpU, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_rpl, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_dpL_col, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_dpU_col, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_nnz_rw, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_nnz_cw, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_rl_pl, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_cl_pl, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMemset(d_nnz_rw, 0, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMemset(d_nnz_cw, 0, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_rowl_L, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_rowl_U, sizeof(int)*(rows));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMemcpy(d_ia, ia, sizeof(int)*(rows+1), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaStatus);
  
  cudaStatus = cudaMemcpy(d_ja, ja, sizeof(int)*nnz, cudaMemcpyHostToDevice);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMemcpy(d_vala, vala, sizeof(double)*nnz, cudaMemcpyHostToDevice);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_jlev, sizeof(int)*rows);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_last, sizeof(int));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_cntr, sizeof(int));
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((int**)&d_cntr1, sizeof(int));
  checkCudaErrors(cudaStatus);

  cudaMemset(d_last, 0, sizeof(int));
  cudaMemset(d_cntr, 0, sizeof(int));
  cudaMemset(d_cntr1, 0, sizeof(int));

  *d_ia_ = d_ia;
  *d_ja_ = d_ja;
  *d_vala_ = d_vala;
  *d_ib_ = d_ib;
  *d_jb_ = d_jb;
  *d_valb_ = d_valb;
  *d_dpU_ = d_dpU;
  *d_dpL_ = d_dpL;
  *d_dpU_col_ = d_dpU_col;
  *d_dpL_col_ = d_dpL_col;
  *d_rowl_L_ = d_rowl_L;
  *d_rowl_U_ = d_rowl_U;
  *d_nnz_rw_ = d_nnz_rw;
  *d_nnz_cw_ = d_nnz_cw;
  *d_cl_pl_ = d_cl_pl;
  *d_rl_pl_ = d_rl_pl;
  *d_rpl_ = d_rpl;
  *d_last_ = d_last;
  *d_jlev_ = d_jlev;
  *d_cntr_ = d_cntr;
  *d_cntr1_ = d_cntr1;
}

int Host_Allocs(int **dpL_, int **dpU_, int **dpL_col_, int **dpU_col_, int **ilev_, int rows)
{
  int *dpL, *dpU;
  int *dpL_col, *dpU_col;
  int *ilev;

  cudaError_t cudaStatus;
  
  cudaStatus = cudaMallocHost((void**)&dpL, sizeof(int)*rows);  
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMallocHost((void**)&dpU, sizeof(int)*rows);  
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMallocHost((void**)&dpL_col, sizeof(int)*rows);  
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMallocHost((void**)&dpU_col, sizeof(int)*rows);  
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMallocHost((void**)&ilev, sizeof(int)*rows);  
  checkCudaErrors(cudaStatus);

  *dpL_ = dpL;
  *dpU_ = dpU;
  *dpL_col_ = dpL_col;
  *dpU_col_ = dpU_col;
  *ilev_ = ilev; 
}