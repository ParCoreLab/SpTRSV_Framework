/*
  Title: Matrix Analysis module tester 
  Implementation: CUDA-GPU implementation
  Author: Najeeb Ahmad
  Date: 17-09-2019
  First version: 17-09-2019
  Modification History:
                    
              
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <libufget.h>
#include <time.h>
#include <fstream>
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization
#include <time.h>
#include <string>
#include <fstream>

#define WARP_SIZE 32
#define BLOCKDIM 128
#define ERR_TOL  1e-9
#define END_TIME 10.0

#include "profiling.h"
#include "anamodgpu.h"

const char *sSDKname     = "AnaModGPU";

int ReadMatrixFromFile(double **val_, int **I_, int **J_, int *M_, int *N_, int *nz_)
{
  ifstream dFile;
  int *I, *J;
  double *val;

  double nnzLength, rowLength;
  long nnzL, rL;
  
  dFile.open("../../datasets/datasetGenerator/data/lengths.txt", ios::in);
  if(!dFile.is_open())
    {
      printf("Failed to open lengths file for reading \n");
      return -1;
    }
  else
    {
      dFile >> nnzLength;
      dFile >> rowLength;
      nnzL = nnzLength;
      rL = rowLength;
      dFile.close();
    }
  val = (double *)malloc(sizeof(double) * nnzL);
  J = (int *)malloc(sizeof(int) * nnzL);
  I = (int *)malloc(sizeof(int) * (rL + 2));
  *M_ = rL;
  *N_ = rL;
  *nz_ = nnzL;

  dFile.open("../../datasets/datasetGenerator/data/val.txt", ios::in);
  if(!dFile.is_open())
    {
      printf("Failed to open val file for reading\n");
      return -1;
    }
  else
    {     
      long count = 0;
      while(count < nnzL)
  {
    dFile >> val[count++];
  }
      dFile.close();
    }

  dFile.open("../../datasets/datasetGenerator/data/J.txt", ios::in);
  if(!dFile.is_open())
    {
      printf("Failed to open val file for reading\n");
      return -1;
    }
  else
    {     
      long count = 0;
      double rVal;
      while(count < nnzL)
  {
    
    dFile >> rVal;
    J[count++] = rVal;
  }
      dFile.close();
    }

  long count = 0;
  dFile.open("../../datasets/datasetGenerator/data/I.txt", ios::in);
  if(!dFile.is_open())
    {
      printf("Failed to open val file for reading\n");
      return -1;
    }
  else
    {     
      double rVal;
      while(count < (rL + 1))
  {
    dFile >> rVal;
    I[count++] = rVal;
  }    
    }
  *val_ = val;
  *I_ = I;
  *J_ = J;

}


int main(int argc, char **argv)
{
  if (argc < 3)
    {
      printf("Usage: %s matrix_ID mode\n", argv[0]);
      return -1;
    }
  
  timespec time1, time2;
  float milliseconds = 0.0;
  double elapsed_time = 0.0, elapsed_time_GPU = 0.0;
  double timeDepn = 0.0, timeLevelsL = 0.0, timeStatsL = 0.0, timeLevelsU=0.0, timeStatsU = 0.0;
  float timeDepnCPU = 0.0, timeLevelLCPU = 0.0, timeStatsLCPU = 0.0, timeStatsUCPU = 0.0, timeILU = 0.0;
  float timeConv = 0.0;
  cudaEvent_t start = 0, stop = 0;
  //   
  uf_collection_t *col;
  uf_field_t field;
  uf_matrix_t *mat=NULL;
  int matID;
  int rows, cols, nnz;
  FeatureVect predFeatures;

  // Host Memory
  int *ia, *ja, *ib, *jb;
  double *vala, *valb;
  int *ilev;

  // Device Memory
  int *d_ia, *d_ja, *d_ib, *d_jb;
  double *d_vala, *d_valb;
  int *d_dpL, *d_dpU, *d_rowl_L, *d_rowl_U;
  int *d_dpL_col, *d_dpU_col;
  int *dpL, *dpU;
  int *dpL_col, *dpU_col;
  int *d_last, *d_jlev;
  int *d_cntr, *d_cntr1;
  int *d_rl_pl, *d_cl_pl;
  int *d_nnz_rw, *d_nnz_cw;
  int *d_rpl;
  int h_last;
  string str1 = "uf";

  cusparseMatDescr_t descr_A = 0;
  cusparseMatDescr_t descr_L = 0;
  cusparseMatDescr_t descr_U = 0;
  csrilu02Info_t info_A  = 0;
  int pBufferSize_A;
  size_t pBufferSize_CSC;
  void *pBuffer = 0, *pBuffer1 = 0;
  const cusparseSolvePolicy_t policy_A = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

  cudaError_t cudaStatus;
  cusparseHandle_t cusparseHandle = 0;
  cusparseStatus_t cusparseStatus;
  cusparseStatus = cusparseCreate(&cusparseHandle);
  checkCudaErrors(cusparseStatus);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Welcome message
  printf("Matrix feature extractor \n");
  printf("Starting [%s]...\n", sSDKname);
   
  cudaDeviceProp deviceProp;
  int devID = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

  printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
  deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  matID = atoi(argv[1]);

  if(str1.compare(argv[2])==0)
  {
    col = uf_collection_init();
    //printf("UF Sparse Matrix Collection Database contains %d matrices.\n", uf_collection_num_matrices(col));
    mat = uf_collection_get_by_id(col, matID);
    printf("Matrix[%4d] %s/%s\n", mat->id, mat->group_name, mat->name);
    printf("Downloading matrix %s/%s, ID: %4d\n", mat->group_name, mat->name, mat->id);
    uf_matrix_coord_int32(&field, &rows, &cols, &nnz, &ia, &ja, (void **)&vala, mat);
    printf("Converting to CSR format ... \n");
    uf_matrix_coord_to_csr_int32(field, rows, cols, nnz, &ia, &ja, (void **)&vala);
  }
  else
  {
    printf("Matrix ID: %d\n", matID);
    printf("Reading matrix from file\n");
    ReadMatrixFromFile(&vala, &ia, &ja, &rows, &cols, &nnz);
    printf("Matrix already in CSR format");
  }  

  Device_Allocs(&d_ia, &d_ja, &d_vala, &d_ib, &d_jb, &d_valb, &d_dpL, &d_dpU, &d_rpl, 
                &d_nnz_rw, &d_nnz_cw, &d_dpL_col, &d_dpU_col, &d_rowl_L, &d_rowl_U, 
                &d_jlev, &d_last, &ia, &ja, &vala, &d_rl_pl, &d_cl_pl, &d_cntr, &d_cntr1, rows, nnz);
  Host_Allocs(&dpL, &dpU, &dpL_col, &dpU_col, &ilev, rows);

  cusparseCreateMatDescr(&descr_A);
  cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
  
  // ILU 0 Factorization
  cudaEventRecord(start);
  cusparseStatus = 
  cusparseCreateCsrilu02Info(&info_A);
  checkCudaErrors(cusparseStatus);
  cusparseStatus =
  cusparseDcsrilu02_bufferSize(cusparseHandle, rows, nnz,
    descr_A, d_vala, d_ia, d_ja, info_A, &pBufferSize_A);
  checkCudaErrors(cusparseStatus);

  cudaStatus = cudaMalloc((void**)&pBuffer, pBufferSize_A);
  checkCudaErrors(cudaStatus);

  cusparseStatus =
  cusparseDcsrilu02_analysis(cusparseHandle, rows, nnz, descr_A,
    d_vala, d_ia, d_ja, info_A, policy_A, pBuffer);
  checkCudaErrors(cusparseStatus);

  cusparseStatus =
  cusparseDcsrilu02(cusparseHandle, rows, nnz, descr_A,
    d_vala, d_ia, d_ja, info_A, policy_A, pBuffer);
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeILU, start, stop);

  // CSR to CSC conversion
  cudaEventRecord(start);
  cusparseStatus = 
  cusparseCsr2cscEx2_bufferSize(cusparseHandle, rows, cols, nnz, d_vala, d_ia, d_ja, 
                                d_valb, d_jb, d_ia, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, &pBufferSize_CSC);
  checkCudaErrors(cusparseStatus);
  
  cudaStatus = cudaMalloc((void**)&pBuffer1, pBufferSize_CSC);
  checkCudaErrors(cudaStatus);

  printf("Converting to CSC format ...\n");
  cusparseStatus =
  cusparseCsr2cscEx2(cusparseHandle, rows, cols, nnz, d_vala, d_ia, d_ja,
                      d_valb, d_jb, d_ib, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                      CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, pBuffer1);
  checkCudaErrors(cudaStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeConv, start, stop);

  // cudaStatus = cudaMemcpy(ia, d_jb, sizeof(int)*(rows+1), cudaMemcpyDeviceToHost);
  // checkCudaErrors(cudaStatus);
  // cudaStatus = cudaMemcpy(ja, d_ib, sizeof(int)*(rows), cudaMemcpyDeviceToHost);
  // checkCudaErrors(cudaStatus);

  // for(int i = 0; i < rows; i++)
  // {
  //   printf("%d ", ia[i]);
  // }
  // printf("\n");
  // for(int i = 0; i < nnz; i++)
  // {
  //   printf("%d ", ja[i]);
  // }
  // printf("\n");

  // return 1;
  
  printf("Calculating row and column lengths, L & U\n");
  profile_start(&time1);
  Calculate_Depn(d_ia, d_ja, d_dpL, d_ib, d_jb, d_dpL_col, rows, &timeDepn);
  //Calc_Depn(d_ia, d_ja, d_dpL, d_dpU, rows);
  //Calc_Depn(d_jb, d_ib, d_dpU_col, d_dpL_col, rows);
  //Calc_Depn_LT_row(d_ia, d_ja, d_dpL, rows);
  //Calc_Depn_LT_col(d_ib, d_jb, d_dpL_col, rows);
  //cudaDeviceSynchronize();
  timeDepnCPU = profile_end(&time1, &time2);
  elapsed_time += milliseconds;
  
  //cudaEventRecord(stop);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&milliseconds, start, stop);
  //elapsed_time += milliseconds;

  //cudaStatus = cudaMemcpy(d_rowl_L, d_dpL, sizeof(int)*(rows), cudaMemcpyDeviceToDevice);
  //checkCudaErrors(cudaStatus);
  IncrementVect(d_dpL, d_rowl_L, rows);
  IncrementVect(d_dpL_col, d_dpL_col, rows);

  IncrementVect(d_dpU, d_rowl_U, rows);
  IncrementVect(d_dpU_col, d_dpU_col, rows);

  //cudaStatus = cudaMemcpy(d_rowl_U, d_dpU, sizeof(int)*(rows), cudaMemcpyDeviceToDevice);
  //checkCudaErrors(cudaStatus);
  

  // cudaStatus = cudaMemcpy(ia, d_dpL_col, sizeof(int)*(rows), cudaMemcpyDeviceToHost);
  // checkCudaErrors(cudaStatus);
  // for(int i = 0; i < rows; i++)
  // {
  //   printf("%d ", ia[i]);
  // }

  // cudaStatus = cudaMemcpy(dpL_col, d_dpL_col, sizeof(int)*(rows), cudaMemcpyDeviceToHost);
  // checkCudaErrors(cudaStatus);
  // cudaStatus = cudaMemcpy(dpU_col, d_dpU_col, sizeof(int)*(rows), cudaMemcpyDeviceToHost);
  // checkCudaErrors(cudaStatus);
  int lvls;
  cudaMemset(d_last, 0, sizeof(int));
  printf("Calculating LT levels ... \n");
  profile_start(&time1);

  lvls = Calc_Levels_L(d_dpL, d_rowl_L, d_dpL_col, d_nnz_rw, d_nnz_cw, d_rpl, 
                       d_rl_pl, d_cl_pl, d_jlev, ilev, d_ib, d_jb, d_last, 
                       d_cntr, d_cntr1, rows, &timeLevelsL);
  timeLevelLCPU = profile_end(&time1, &time2);
  elapsed_time += timeLevelLCPU;

  // cudaStatus = cudaMemcpy(ja, d_cl_pl, sizeof(int)*(rows), cudaMemcpyDeviceToHost);
  // checkCudaErrors(cudaStatus);
  // cudaStatus = cudaMemcpy(ia, ilev, sizeof(int)*(rows), cudaMemcpyDeviceToHost);
  // checkCudaErrors(cudaStatus);

  

  //
  profile_start(&time1); 
  CalcStats(&predFeatures, d_nnz_rw, d_nnz_cw, d_rowl_L, d_dpL_col, d_rpl, 
            d_rl_pl, d_cl_pl, ilev, d_jlev, lvls, rows, &timeStatsL);
  timeStatsLCPU = profile_end(&time1, &time2);
  elapsed_time += timeStatsLCPU;

  cudaMemset(d_last, 0, sizeof(int));
  cudaMemset(d_cntr, 0, sizeof(int));
  cudaMemset(d_cntr1, 0, sizeof(int));
  //cudaMemset(d_jlev, 0, sizeof(int));
  printf("Calculating UT levels ... \n");
  profile_start(&time1);
  lvls = Calc_Levels_U(d_dpU, d_rowl_U, d_dpU_col, d_nnz_rw, d_nnz_cw, d_rpl, 
                      d_rl_pl, d_cl_pl, d_jlev, ilev, d_ib, d_jb, d_last,
                      d_cntr, d_cntr1, rows, &timeLevelsU);
  timeStatsUCPU = profile_end(&time1, &time2);
  //elapsed_time += timeStatsUCPU;
  
  //profile_start(&time1); 
  //CalcStats(&predFeatures, d_nnz_rw, d_nnz_cw, d_rowl_U, d_dpU_col, d_rpl, 
  //          d_rl_pl, d_cl_pl, ilev, d_jlev, lvls, rows, &timeStatsU);
  //timeStatsUCPU = profile_end(&time1, &time2);
  //elapsed_time += timeStatsUCPU;

  //printf("Upper levels = %d\n", lvls);

  PrintStats(&predFeatures);
  printf("%f\n", elapsed_time);
  //printf("%f\n", elapsed_time_GPU);
  //printf("Timings:\n");
  printf("%f\n", timeILU);
  printf("%f\n", timeConv);
  printf("%f\n", timeDepn);
  printf("%f\n", timeLevelsL);
  printf("%f", timeStatsL);
  //printf("%f\n", timeLevelsU);
  //printf("%f\n", timeStatsU);

  return 1;


  // for(int i = 0; i < lvls; i++)
  // {
  //   printf("Level %d\n", i+1);

  //   for(int j = ia[i]; j < ia[i+1]; j++)
  //     printf("%d, ", ja[j]);
  //   printf("\n");
  // }
  
   cudaMemset(d_last, 0, sizeof(int));
   cudaMemset(d_cntr, 0, sizeof(int));
   cudaMemset(d_cntr1, 0, sizeof(int));
   printf("Calculating UT levels ... \n");
   lvls = Find_Levels_U(d_dpU, d_rowl_U, d_dpU_col, d_nnz_rw, d_nnz_cw, d_rpl, 
                       d_rl_pl, d_cl_pl, d_jlev, ilev, d_ib, d_jb, d_last, d_cntr, d_cntr1, rows);
   printf("Upper levels = %d\n", lvls);


  // cudaStatus = cudaMemcpy (&h_last , d_last , sizeof(int), cudaMemcpyDeviceToHost);
  // checkCudaErrors(cudaStatus);
  // cudaStatus = cudaMemcpy (ia , d_jlev , sizeof(int)*h_last, cudaMemcpyDeviceToHost);
  // checkCudaErrors(cudaStatus);

  // for(int i = 0; i < h_last; i++)
  //  {
  //    printf("%d ", ia[i]);
  //  }

  //  return 1;
  cudaFree(pBuffer);
  cusparseDestroyMatDescr(descr_A);
  cusparseDestroyCsrilu02Info(info_A);
  cusparseDestroy(cusparseHandle);


}
