/*
  Title: ILU solve on GPU 
  Implementation: CUDA GPU implementation with pinned memory
  Author: Najeeb Ahmad
  Date: 12-03-2019
  First version: 14-03-2019
  Modification History:
                       10-04-2019: cusparse version 2 ilu0
		       and triangular solve functions added
		       22-10-2019: Added no-lvl cusparse v2, analysis,
		       ilu time
              
*/
/* This program collects total execution time and iterations for
   each of the following operations which are executed for END_TIME
   seconds
   - SpMV 
   - Device to Host vector transfer (vector size = number of columns in matrix)
   - Lower triangular solve
   - Upper triangular solve
   - Lower and Upper triangular solve combined
   - Host to Device vector transfer (vector size = number of columns in matrix)
   The data is stored in output file in the following format:
  totalSpMV time, SpMV iterations, DtoHtime, DtoH iterations, Lsolve time, 
  Lsolve iterations, Usolve time, Usolve iterations, HtoDtime, HtoD iterations

  File is stored with the filename matID_LU.txt, where matID is the UoF matrix
  ID
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization
#include <libufget.h>
#include <time.h>
#include <fstream>
#include <string>
#define END_TIME 10.0
#define ITERATIONS 100.0

#include "profiling.h"

int Device_Allocs(int **d_I_, int **d_J_, double **d_val_, double **d_vect_, double **d_bilu0_, double **d_Av_, 
		  double **d_y_, double **d_res_,
		  int **I_, int **J_, double **val_, double **vect_,
		  int M, int N, int nz);
int Host_Allocs(int **I, int **J, double **val, double **vect,
                int **Il, int **Jl, double **vall, double **Av, double **res,
		int M, int N, int nz);

const char *sSDKname     = "ILU_GPU";


int ReadMatrixFromFile(double **val_, int **I_, int **J_, int *M_, int *N_, int *nz_)
{
  ifstream dFile;
  int *I, *J;
  double *val;

  double nnzLength, rowLength;
  long nnzL, rL;
  
  dFile.open("lengths.txt", ios::in);
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

  dFile.open("val.txt", ios::in);
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

  dFile.open("J.txt", ios::in);
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
  dFile.open("I.txt", ios::in);
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
      printf("Usage: %s matrix_ID matrix_read_mode(uf/file)\n", argv[0]);
      return -1;
    }

  // Log file
  ofstream logFile;
  char nameBuffer[100];
  
  // CUDA events
  cudaEvent_t start, stop;
  cudaEvent_t start1, stop1;
  cudaEvent_t start2, stop2;
  float milliseconds = 0.0;
  double SpMV_pi = 0.0, LT_pi = 0.0, UT_pi = 0.0, LUT_pi = 0.0, DtoH_pi = 0.0, HtoD_pi;
  double LT_pi_NoLvl = 0.0, UT_pi_NoLvl = 0.0, LUT_pi_NoLvl = 0.0;
  double totalSpMVtime = 0.0, totalLtime = 0.0, totalUtime = 0.0, totalLUtime = 0.0;
  double memAlloctime = 0.0; 
  float iluTimeAnalysis = 0.0, iluTime = 0.0, analysisTimeL = 0.0, analysisTimeU = 0.0, dataConvTime = 0.0;
  float analysisTimeL_NoLvl = 0.0, analysisTimeU_NoLvl = 0.0, iluTimeAnalysis_NoLvl = 0.0; 
  double totalHtoDtime = 0.0, totalDtoHtime = 0.0;

  // Misc vars
  double  doubleone=1.0, doublezero=0.0;
  timespec time1, time2;  
  double elapsed_time = 0.0; 
  unsigned long iterationsSpMV = 0, iterationsL = 0, iterationsU = 0, iterationsLU = 0;
  unsigned long iterationsHtoD = 0, iterationsDtoH = 0;

  // LIBUFGET vars
  uf_collection_t *col;
  uf_field_t field;
  uf_matrix_t *mat=NULL;
  int matID;
  int M, N, nz;
  int *Il, *Jl;
  double *vall;
  
  // Pinned memory allocations
  int *I, *J;
  double *val;
  double *vect;     // Vector
  double *Av, *res;
  
  // Device vars
  int *d_I, *d_J;
  double *d_val;
  double *d_Av, *d_bilu0, *d_vect, *d_y, *d_res;

  // Welcome message
  printf("Program to solve Ax = b using incomplete LU factorization \n");
  printf("Starting [%s]...\n", sSDKname);

  cudaDeviceProp deviceProp;
  int devID = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

  printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
	 deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
#ifdef CUS1
  printf("cuSPARSE version 1\n");
#else
  printf("cuSPARSE version 2\n");
#endif

  cudaError_t cudaStatus;
  // Create cuSPARSE, cublas contexts and status vars
  cusparseHandle_t cusparseHandle = 0;
  cusparseStatus_t cusparseStatus;
  cusparseStatus = cusparseCreate(&cusparseHandle);
  checkCudaErrors(cusparseStatus);
  cublasHandle_t cublasHandle = 0;
  cublasStatus_t cublasStatus;
  cublasStatus = cublasCreate(&cublasHandle);
  checkCudaErrors(cublasStatus);

  // Create descriptor for input sparse matrix 
  cusparseMatDescr_t descr = 0;
  cusparseStatus = cusparseCreateMatDescr(&descr);
  checkCudaErrors(cusparseStatus);  

  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

   // Descriptor for lower triangular matrix
  cusparseMatDescr_t descrL = 0;
  cusparseStatus = cusparseCreateMatDescr(&descrL);
  cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);
  checkCudaErrors(cusparseStatus);

  // Descriptor for upper triangular matrix
  cusparseMatDescr_t descrU = 0;
  cusparseStatus = cusparseCreateMatDescr(&descrU);
  cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
  cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
  checkCudaErrors(cusparseStatus);

  // Analysis object for input matrix
#ifdef CUS1
  cusparseSolveAnalysisInfo_t infoA = 0;
  cusparseSolveAnalysisInfo_t info_u = 0;  
  cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA);
  checkCudaErrors(cusparseStatus);
  cusparseCreateSolveAnalysisInfo(&info_u);
  checkCudaErrors(cusparseStatus);  
#else
  csrsv2Info_t infoA = 0;
  csrsv2Info_t info_u = 0;
  csrilu02Info_t info_M = 0;
  cusparseStatus = cusparseCreateCsrsv2Info(&infoA);
  checkCudaErrors(cusparseStatus);
  cusparseStatus = cusparseCreateCsrsv2Info(&info_u);
  checkCudaErrors(cusparseStatus);
  cusparseStatus = cusparseCreateCsrilu02Info(&info_M);
#endif
  
  // Create CUDA events for profiling
  cudaEventCreate(&start); cudaEventCreate(&start1); cudaEventCreate(&start2);
  cudaEventCreate(&stop); cudaEventCreate(&stop1); cudaEventCreate(&stop2);

  matID = atoi(argv[1]);
#ifdef TIME_BASED
  // Open log file
  sprintf(nameBuffer, "%d_%s", matID, "LU_GPU.txt");
  logFile.open(nameBuffer, ios::out);
  if(!logFile.is_open())
    {
      printf("Failed to open file for logging\n");
      return -1;
    }
#endif
  string str1 = "uf";
  if(str1.compare(argv[2])==0)
    {
      col = uf_collection_init();
      printf("UF Sparse Matrix Collection Database contains %d matrices.\n", uf_collection_num_matrices(col));
      mat = uf_collection_get_by_id(col, matID);
      printf("Matrix[%4d] %s/%s\n", mat->id, mat->group_name, mat->name);
      printf("Downloading matrix %s/%s, ID: %4d\n", mat->group_name, mat->name, mat->id);
      uf_matrix_coord_int32(&field, &M, &N, &nz, &Il, &Jl, (void **)&vall, mat);
      printf("Converting to CSR format ... \n");
      uf_matrix_coord_to_csr_int32(field, M, N, nz, &Il, &Jl, (void **)&vall);
    }
  else
    {
      ReadMatrixFromFile(&vall, &Il, &Jl, &M, &N, &nz);	
    }
      // Host and Device Allocations
  profile_start(&time1);
  Host_Allocs(&I, &J, &val, &vect,
		  &Il, &Jl, &vall, &Av, &res,
		  M, N, nz);
      
  Device_Allocs(&d_I, &d_J, &d_val, &d_vect, &d_bilu0, &d_Av, 
                &d_y, &d_res,
		&I, &J, &val, &vect, 
		M, N, nz);
  memAlloctime = profile_end(&time1, &time2);
  

 // Input matrix analysis
  printf("Starting matrix analysis ... \n");
  float tempr = 0.0;
#ifdef CUS1
  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					   N, nz, descr, d_val, d_I, d_J, infoA);
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&analysisTimeL, start, stop);

  // Upper triangular matrix analysis

  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrU, d_val, d_I, d_J, info_u);
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&analysisTimeU, start, stop);

#else
  void *pBuffer = NULL;
  int pBufferSizeInBytesL;
  int pBufferSizeInBytesU;
  int pBufferSizeInBytesA;
  int pBufferSize;

  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrsv2_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descr, d_val, d_I, d_J, infoA, &pBufferSizeInBytesL);
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tempr, start, stop);
  analysisTimeL += tempr;
  analysisTimeL_NoLvl = analysisTimeL;

  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrsv2_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrU, d_val, d_I, d_J, info_u, &pBufferSizeInBytesU);
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tempr, start, stop);
  analysisTimeU += tempr;
  analysisTimeU_NoLvl = analysisTimeU;

  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrilu02_bufferSize(cusparseHandle, N, nz, descr, d_val, d_I, d_J, info_M, &pBufferSizeInBytesA);
  checkCudaErrors(cusparseStatus);
  pBufferSize = max(pBufferSizeInBytesL, max(pBufferSizeInBytesU, pBufferSizeInBytesA));
  cudaMalloc((void **)&pBuffer, pBufferSize);
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tempr, start, stop);
  iluTimeAnalysis += tempr;

  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrsv2_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descr, d_val, d_I, d_J, infoA, 
					    CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);  checkCudaErrors(cusparseStatus);
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tempr, start, stop);
  analysisTimeL_NoLvl += tempr;

  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrsv2_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descr, d_val, d_I, d_J, infoA, 
					    CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);  checkCudaErrors(cusparseStatus);
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tempr, start, stop);
  analysisTimeL += tempr;

  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrsv2_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrU, d_val, d_I, d_J, info_u, 
					    CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);  
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tempr, start, stop);
  analysisTimeU_NoLvl += tempr;

  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrsv2_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrU, d_val, d_I, d_J, info_u, 
					    CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);  
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tempr, start, stop);
  analysisTimeU += tempr;

  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrilu02_analysis(cusparseHandle, N, nz, descr, d_val, d_I, d_J, info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tempr, start, stop);
  iluTimeAnalysis_NoLvl += tempr;

  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrilu02_analysis(cusparseHandle, N, nz, descr, d_val, d_I, d_J, info_M, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tempr, start, stop);
  iluTimeAnalysis += tempr;
  printf("Done successfully!\n");
  //return 1;
#endif
  printf("Performing ILU decomposition (level 0) ... \n");
  cublasStatus = cublasDcopy(cublasHandle, nz, d_val, 1, d_bilu0, 1);
  checkCudaErrors(cublasStatus);

#ifdef CUS1  
  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, descr, d_bilu0, d_I, d_J, infoA);  
  checkCudaErrors(cusparseStatus);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tempr, start, stop);
  iluTime += tempr;
#else
  cudaEventRecord(start);
  cusparseStatus = cusparseDcsrilu02(cusparseHandle, N, nz, descr, d_bilu0, d_I, d_J, info_M, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer); 
  checkCudaErrors(cusparseStatus);  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tempr, start, stop);
  iluTime += tempr;
#endif

#ifdef TIME_BASED
  printf("Running SpMV in a loop for %lg seconds\n", END_TIME);  
  profile_start(&time1);
  do
    {
      cudaEventRecord(start);
      cusparseStatus = cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nz, 
				      &doubleone, descr, d_val, d_I, d_J, d_vect, &doublezero, d_Av);
      cudaEventRecord(stop);
      checkCudaErrors(cusparseStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalSpMVtime += milliseconds;
      elapsed_time = profile_end(&time1, &time2);       
      iterationsSpMV++;
    }
  while(elapsed_time < END_TIME);
  printf("Total milliseconds elapsed in SpMV = %lg, Iterations=%ld\n", totalSpMVtime, iterationsSpMV);
#else
  printf("Running SpMV in a loop for %lg iterations\n", ITERATIONS);  
  do
    {
      cudaEventRecord(start);
      cusparseStatus = cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nz, 
				      &doubleone, descr, d_val, d_I, d_J, d_vect, &doublezero, d_Av);
      cudaEventRecord(stop);
      checkCudaErrors(cusparseStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalSpMVtime += milliseconds;
    }
  while(++iterationsSpMV != ITERATIONS);
  SpMV_pi = totalSpMVtime/iterationsSpMV;
#endif

  cudaDeviceSynchronize();
#ifdef TIME_BASED
  printf("Performing device to host transfer of d_Av in a loop for %lg seconds\n", END_TIME);
  profile_start(&time1);
  do
    {
      cudaEventRecord(start);
      cudaStatus = cudaMemcpy(Av, d_Av, sizeof(int)*N, cudaMemcpyDeviceToHost);
      cudaEventRecord(stop);
      checkCudaErrors(cudaStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalDtoHtime += milliseconds;
      elapsed_time = profile_end(&time1, &time2);       
      iterationsDtoH++;
    }
  while(elapsed_time < END_TIME);
  printf("Total milliseconds elapsed in Device to Host transfer= %lg, Iterations=%ld\n", totalDtoHtime, iterationsDtoH);
#else
  printf("Performing device to host transfer of d_Av in a loop for %lg times\n", ITERATIONS);
  do
    {
      cudaEventRecord(start);
      cudaStatus = cudaMemcpy(Av, d_Av, sizeof(int)*N, cudaMemcpyDeviceToHost);
      cudaEventRecord(stop);
      checkCudaErrors(cudaStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalDtoHtime += milliseconds;
    }
  while(++iterationsDtoH != ITERATIONS);
  DtoH_pi = totalDtoHtime / iterationsDtoH;
#endif
  cudaDeviceSynchronize();  


#ifdef TIME_BASED
  printf("Running Lower triangular solve in a loop for %lg seconds\n", END_TIME);  
  iterationsL = 0;
  profile_start(&time1);
  do
    {      
      cudaEventRecord(start);
#ifdef CUS1
      //printf("Using CUSPARSE 1 solve\n");

      cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &doubleone, descrL,
					d_bilu0, d_I, d_J, infoA, d_Av, d_y);
#else
      //printf("Using CUSPARSE 2 solve\n");
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrL,
					    d_bilu0, d_I, d_J, infoA, d_Av, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);     
#endif
      cudaEventRecord(stop);      
      checkCudaErrors(cusparseStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalLtime += milliseconds;
      elapsed_time = profile_end(&time1, &time2);       
      iterationsL++;
    }
  while(elapsed_time < END_TIME);
  printf("Total milliseconds elapsed in L solve = %lg, Iterations=%ld\n", totalLtime, iterationsL);
#else
  printf("Running Lower triangular solve in a loop for %lg times\n", ITERATIONS);  
  iterationsL = 0;
  do
    {      
      cudaEventRecord(start);
#ifdef CUS1
      cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &doubleone, descrL,
					d_bilu0, d_I, d_J, infoA, d_Av, d_y);
#else
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrL,
					    d_bilu0, d_I, d_J, infoA, d_Av, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);     
#endif
      cudaEventRecord(stop);      
      checkCudaErrors(cusparseStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalLtime += milliseconds;      
    }
  while(++iterationsL != ITERATIONS);
  LT_pi = totalLtime / iterationsL;

#endif
  cudaDeviceSynchronize();

#ifdef TIME_BASED

#else
#ifdef CUS2
  printf("Running Lower triangular solve (No Level) in a loop for %lg times\n", ITERATIONS);  
  iterationsL = 0;
  totalLtime = 0.0;
  do
    {      
      cudaEventRecord(start);
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrL,
					     d_bilu0, d_I, d_J, infoA, d_Av, d_y, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);     
      cudaEventRecord(stop);      
      checkCudaErrors(cusparseStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalLtime += milliseconds;      
    }
  while(++iterationsL != ITERATIONS);
  LT_pi_NoLvl = totalLtime / iterationsL;

#endif

#endif

  cudaDeviceSynchronize();

#ifdef TIME_BASED  
  printf("Running Upper triangular solve in a loop for %lg seconds\n", END_TIME);  
  iterationsU = 0;
  profile_start(&time1);
  do
    {      
      cudaEventRecord(start);
#ifdef CUS1
      cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &doubleone, descrU,
					    d_bilu0, d_I, d_J, info_u, d_y, d_res);
#else
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrU,
					     d_bilu0, d_I, d_J, info_u, d_y, d_res, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
					     pBuffer);
#endif
      cudaEventRecord(stop);      
      checkCudaErrors(cusparseStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalUtime += milliseconds;
      elapsed_time = profile_end(&time1, &time2);       
      iterationsU++;
    }
  while(elapsed_time < END_TIME);
  printf("Total milliseconds elapsed in U solve = %lg, Iterations=%ld\n", totalUtime, iterationsU);

#else

  printf("Running Upper triangular solve in a loop for %lg times\n", ITERATIONS);  
  iterationsU = 0;
  do
    {      
      cudaEventRecord(start);
#ifdef CUS1
      cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &doubleone, descrU,
					    d_bilu0, d_I, d_J, info_u, d_y, d_res);
#else
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrU,
					     d_bilu0, d_I, d_J, info_u, d_y, d_res, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
					     pBuffer);
#endif
      cudaEventRecord(stop);      
      checkCudaErrors(cusparseStatus);
      cudaEventSynchronize(stop);
      totalUtime += milliseconds;
    }
  while(++iterationsU < END_TIME);
  UT_pi = totalUtime/iterationsU;
#endif

  cudaDeviceSynchronize();

#ifdef TIME_BASED  

#else
#ifdef CUS2
  printf("Running Upper triangular solve (No Lvl) in a loop for %lg times\n", ITERATIONS);  
  iterationsU = 0;
  totalUtime = 0.0;
  do
    {      
      cudaEventRecord(start);
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrU,
					     d_bilu0, d_I, d_J, info_u, d_y, d_res, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);
      cudaEventRecord(stop);      
      checkCudaErrors(cusparseStatus);
      cudaEventSynchronize(stop);
      totalUtime += milliseconds;
    }
  while(++iterationsU < END_TIME);
  UT_pi_NoLvl = totalUtime/iterationsU;
#endif
#endif
  cudaDeviceSynchronize();



#ifdef TIME_BASED
  printf("Running LU solve in a loop for %lg seconds\n", END_TIME);  
  profile_start(&time1);
  do
    {      
      cudaEventRecord(start);
#ifdef CUS1
      cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &doubleone, descrL,
					d_bilu0, d_I, d_J, infoA, d_Av, d_y);
      
      cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &doubleone, descrU,
      					    d_bilu0, d_I, d_J, info_u, d_y, d_res);      
#else
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrL,
					    d_bilu0, d_I, d_J, infoA, d_Av, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrU,
					     d_bilu0, d_I, d_J, info_u, d_y, d_res, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
					     pBuffer);
#endif
      cudaEventRecord(stop);
      checkCudaErrors(cusparseStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalLUtime += milliseconds;
      elapsed_time = profile_end(&time1, &time2);       
      iterationsLU++;
    }
  while(elapsed_time < END_TIME);
 
  printf("Total milliseconds elapsed in LU solve = %lg, Iterations=%ld\n", totalLUtime, iterationsLU);
#else
  printf("Running LU solve in a loop for %lg times\n", ITERATIONS);  
  do
    {      
      cudaEventRecord(start);
#ifdef CUS1
      cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &doubleone, descrL,
					d_bilu0, d_I, d_J, infoA, d_Av, d_y);
      
      cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &doubleone, descrU,
      					    d_bilu0, d_I, d_J, info_u, d_y, d_res);      
#else
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrL,
					    d_bilu0, d_I, d_J, infoA, d_Av, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrU,
					     d_bilu0, d_I, d_J, info_u, d_y, d_res, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
					     pBuffer);
#endif
      cudaEventRecord(stop);
      checkCudaErrors(cusparseStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalLUtime += milliseconds;      
    }
  while(++iterationsLU != ITERATIONS);
  LUT_pi = totalLUtime/iterationsLU;
#endif

  cudaDeviceSynchronize();



#ifdef TIME_BASED

#else
#ifdef CUS2
  printf("Running LU solve (No lvl) in a loop for %lg times\n", ITERATIONS);  
  totalLUtime = 0.0;
  iterationsLU = 0;
  do
    {      
      cudaEventRecord(start);
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrL,
					    d_bilu0, d_I, d_J, infoA, d_Av, d_y, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);
      cusparseStatus = cusparseDcsrsv2_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &doubleone, descrU,
					     d_bilu0, d_I, d_J, info_u, d_y, d_res, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
					     pBuffer);
      cudaEventRecord(stop);
      checkCudaErrors(cusparseStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalLUtime += milliseconds;      
    }
  while(++iterationsLU != ITERATIONS);
  LUT_pi_NoLvl = totalLUtime/iterationsLU;
#endif
#endif

  cudaDeviceSynchronize();

#ifdef TIME_BASED
  printf("Performing host to device transfer of res in a loop for %lg seconds\n", END_TIME);  
  profile_start(&time1);
  do
    {
      cudaEventRecord(start);
      cudaStatus = cudaMemcpy(d_res, res, sizeof(int)*N, cudaMemcpyHostToDevice);
      cudaEventRecord(stop);
      checkCudaErrors(cudaStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalHtoDtime += milliseconds;
      elapsed_time = profile_end(&time1, &time2);       
      iterationsHtoD++;
    }
  while(elapsed_time < END_TIME);
  printf("Total milliseconds elapsed in Host to Device transfer= %lg, Iterations=%ld\n", totalHtoDtime, iterationsHtoD);
#else
  printf("Performing host to device transfer of res in a loop for %lg times\n", ITERATIONS);  
  do
    {
      cudaEventRecord(start);
      cudaStatus = cudaMemcpy(d_res, res, sizeof(int)*N, cudaMemcpyHostToDevice);
      cudaEventRecord(stop);
      checkCudaErrors(cudaStatus);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      totalHtoDtime += milliseconds;      
    }
  while(++iterationsHtoD != ITERATIONS);
  HtoD_pi = totalHtoDtime / iterationsHtoD;
#endif
  cudaDeviceSynchronize();  
  
  printf("Finished test case ... \n");

#ifdef TIME_BASED
  // Writing data to output file
  logFile << totalSpMVtime << "," << iterationsSpMV << ","
	  << totalDtoHtime << "," << iterationsDtoH << ","
	  << totalLtime << "," << iterationsL << ","
	  << totalUtime << "," << iterationsU << "," 
	  << totalLUtime << ","<< iterationsLU << ","
	  << totalHtoDtime << "," << iterationsHtoD << endl;
  logFile.close();
#else
  cout << SpMV_pi << " " << LT_pi << " " << UT_pi << " " << LUT_pi << " " << LT_pi_NoLvl << " " 
       << UT_pi_NoLvl << " " << LUT_pi_NoLvl << " " << HtoD_pi << " " << DtoH_pi << 
    " " << iluTimeAnalysis << " " << iluTime << " " << analysisTimeL << " " << analysisTimeU << " " <<
    memAlloctime << " " << analysisTimeL_NoLvl << " " << analysisTimeU_NoLvl << " " << iluTimeAnalysis_NoLvl <<  endl; 
#endif
  return 1;
}

int Host_Allocs(int **I_, int **J_, double **val_, double **vect_,
                int **Il_, int **Jl_, double **vall_, double **Av_, double **res_,
		int M, int N, int nz)
{
  
  int *I, *J;
  int *Il, *Jl;
  double *val, *vect, *vall;
  double *Av, *res;
  
  Il = *Il_;
  Jl = *Jl_;
  vall = *vall_;
  

  cudaError_t cudaStatus;
  cudaStatus = cudaMallocHost((void**)&I, sizeof(int)*(M+1));  
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMallocHost((void**)&J, sizeof(int)*nz);  
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMallocHost((void**)&val, sizeof(double)*nz);  
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMallocHost((void**)&vect, sizeof(double)*N);  
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMallocHost((void**)&Av, sizeof(double)*N);  
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMallocHost((void**)&res, sizeof(double)*N);  
  checkCudaErrors(cudaStatus);
  
  for(int i = 0; i < (M+1); i++)
    {
      I[i] = Il[i];
    }
  
  for(int i = 0; i < nz; i++)
    {
      J[i] = Jl[i];
      val[i] = vall[i];
    }
  // Initialize vect to 1
  for(int i = 0; i < N; i++)
    {
      vect[i] = 1.0;
    }
  
  *I_= I;
  *J_= J;
  *vall_ = vall;
  *vect_ = vect;
  *val_ = val;
  *Av_ = Av;
  *res_ = res;
  
  return 1;
}

int Device_Allocs(int **d_I_, int **d_J_, double **d_val_, double **d_vect_,  
		  double **d_bilu0_, double **d_Av_, double **d_y_, double **d_res_,
		  int **I_, int **J_, double **val_, double **vect_, 
		  int M, int N, int nz)
{
  int *d_I, *d_J;
  int *I, *J;
  double *d_val, *d_vect, *d_bilu0, *d_Av, *d_y, *d_res;
  double *val, *vect;
  cudaError_t cudaStatus;

  I = *I_;
  J = *J_;
  val = *val_;
  vect = *vect_;
    
  cudaStatus = cudaMalloc((int**)&d_I, sizeof(int)*(M+1));
  checkCudaErrors(cudaStatus);
  
  cudaStatus = cudaMalloc((int**)&d_J, sizeof(double)*nz);
  checkCudaErrors(cudaStatus);     
  
  cudaStatus = cudaMalloc((double**)&d_val, sizeof(double)*nz);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((double**)&d_Av, sizeof(double)*N);
  checkCudaErrors(cudaStatus);
  
  cudaStatus = cudaMalloc((double**)&d_y, sizeof(double)*N);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((double**)&d_vect, sizeof(double)*N);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((double**)&d_res, sizeof(double)*N);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMalloc((double**)&d_bilu0, sizeof(double)*nz);
  checkCudaErrors(cudaStatus);    
  
  cudaStatus = cudaMemcpy(d_I, I, sizeof(int)*(M+1), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaStatus);
  
  cudaStatus = cudaMemcpy(d_J, J, sizeof(int)*nz, cudaMemcpyHostToDevice);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMemcpy(d_val, val, sizeof(double)*nz, cudaMemcpyHostToDevice);
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaMemcpy(d_vect, vect, sizeof(double)*N, cudaMemcpyHostToDevice);
  checkCudaErrors(cudaStatus);

  *d_I_ = d_I;
  *d_J_ = d_J;
  *d_val_=d_val;
  *d_vect_ = d_vect;
  *d_bilu0_ = d_bilu0;
  *d_Av_ = d_Av;
  *d_y_ = d_y;
  *d_res_ = d_res;  
  
  return 1;
}
