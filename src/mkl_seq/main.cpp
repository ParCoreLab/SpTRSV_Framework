/*
  Title: ILU solve on CPU 
  Implementation: MKL CPU implementation
  Author: Najeeb Ahmad
  Date: 14-03-2019
  First version: 14-03-2019 
  Modification History:
              
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
  totalSpMV time, SpMV iterations, Lsolve time, 
  Lsolve iterations, Usolve time, Usolve iterations, 

  File is stored with the filename matID_LU.txt, where matID is the UoF matrix
  ID
*/

#include <stdio.h>
#include <mkl.h>
#include <mkl_spblas.h>
#include <libufget.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>
#define END_TIME 10000.0
#define ITERATIONS 100
#include "profiling.h"

int Mem_Allocs(double **vect_, double **Av_, double **res_, double **bilu0_, double **y_,
		 int M, int N, int nz);

void MKLErrorHandle(sparse_status_t sp_status, char *msg);

const char *sSDKname     = "ILU_CPU";


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
      printf("Usage: %s matrix_ID [uf or file]\n", argv[0]);
      return -1;
    }

  // Log file
  ofstream logFile;
  char nameBuffer[100];

  // Timing vars
  timespec time3, time4;
  float milliseconds = 0.0; 
  double totalSpMVtime = 0.0, totalLtime = 0.0, totalUtime = 0.0, totalLUtime = 0.0;
  double SpMV_pi = 0.0, LT_pi = 0.0, UT_pi = 0.0, LUT_pi = 0.0;
  double totalHtoDtime = 0.0, totalDtoHtime = 0.0;

  // Misc vars
  double  doubleone=1.0, doublezero=0.0;
  timespec time1, time2;  
  double elapsed_time = 0.0;
  double iluTime=0.0, iluTimeAnalysisL=0.0, iluTimeAnalysisU=0.0, Mem_alloc_time=0.0;  
  unsigned long iterationsSpMV = 0, iterationsL = 0, iterationsU = 0, iterationsLU = 0;
  unsigned long iterationsHtoD = 0, iterationsDtoH = 0;
  int n_iter;

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
  double *Av, *res, *bilu0, *y;
  
 
  // Create cuSPARSE, cublas contexts and status vars
  sparse_matrix_t csrA, csrLU, csrLU1;
  struct matrix_descr descrA;
  struct matrix_descr descrL, descrU;
  sparse_status_t sp_status;
  MKL_INT ipar[128];
  double dpar[128];
  MKL_INT ivar, ierr;
  
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  descrA.mode = SPARSE_FILL_MODE_UPPER; 
  descrA.diag = SPARSE_DIAG_NON_UNIT;
  descrL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descrL.mode = SPARSE_FILL_MODE_LOWER;
  descrL.diag = SPARSE_DIAG_UNIT;
  descrU.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descrU.mode = SPARSE_FILL_MODE_UPPER;
  descrU.diag = SPARSE_DIAG_NON_UNIT;
  
  ipar[30] = 1;
  dpar[30] = 1.E2;
  dpar[31] = 1.E2;

  // Create descriptor for input sparse matrix 
 
   // Descriptor for lower triangular matrix
 
  // Descriptor for upper triangular matrix
 
  // Analysis object for input matrix
  
  // Create CUDA events for profiling
 
  matID = atoi(argv[1]);
  // Open log file
#ifdef TIME_BASED
  sprintf(nameBuffer, "%d_%s", matID, "MKL_seq.txt");
  logFile.open(nameBuffer, ios::out);
  if(!logFile.is_open())
    {
      printf("Failed to open file for logging\n");
      return -1;
    }
#endif

  // Welcome message
  printf("Program to solve Ax = b using incomplete LU factorization \n");
  printf("Starting [%s]...\n", sSDKname);
  string str1 = "uf";

    if(str1.compare(argv[2])==0)
    {
      col = uf_collection_init();
      printf("UF Sparse Matrix Collection Database contains %d matrices.\n", uf_collection_num_matrices(col));
      mat = uf_collection_get_by_id(col, matID);
      printf("Matrix[%4d] %s/%s\n", mat->id, mat->group_name, mat->name);
      printf("Downloading matrix %s/%s, ID: %4d\n", mat->group_name, mat->name, mat->id);
      uf_matrix_coord_int32(&field, &M, &N, &nz, &I, &J, (void **)&val, mat);
      printf("Converting to CSR format ... \n");
      uf_matrix_coord_to_csr_int32(field, M, N, nz, &I, &J, (void **)&val);
    }
    else
      {
	ReadMatrixFromFile(&val, &I, &J, &M, &N, &nz);
      }
      ivar = N;

  // Starting I and J indices from 1 as per requirement of some of the MKL routines
  for(int i=0; i <=M; i++)
    {
      I[i] = I[i] + 1;
    }

  for(int i=0; i < nz; i++)
    {
      J[i] = J[i] + 1;
    }
  int cntr = 0;
  for(int i = 0; i < M; i++)
    {
      for(int j = I[i]; j < I[i+1]; j++)
	{
	  if(i+1 == J[j-1])
	    cntr++;
	}
    }

  mkl_set_num_threads(40);
  profile_start(&time1);
  // Host Allocations
  Mem_Allocs(&vect, &Av, &res, &bilu0, &y, M, N, nz);
  Mem_alloc_time = profile_end(&time1, &time2);

  // Create input sparse matrix
  sp_status = mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, M, N, I, I + 1, J, val);
  MKLErrorHandle(sp_status, "Sparse matrix descriptor creation");
  printf("Performing ILU decomposition (level 0) ... \n");
  profile_start(&time1);
  dcsrilu0(&ivar, val, I, J, bilu0, ipar, dpar, &ierr);
  iluTime = profile_end(&time1, &time2);

  if(ierr != 0)
    {
      printf("Error performing ILU decomposition\n");
      printf("Error code: %d\n", ierr);
      return -1;
    }
  sp_status = mkl_sparse_d_create_csr(&csrLU, SPARSE_INDEX_BASE_ONE, M, N, I, I + 1, J, bilu0);
  MKLErrorHandle(sp_status, "LU descriptor creation");

// Input matrix analysis
  // Upper triangular matrix analysis

  // Inspector-executor model first stage, analysis

#ifdef TIME_BASED

  printf("Running SpMV in a loop for %lg seconds\n", END_TIME);  
  profile_start(&time1);
  do
    {
      profile_start(&time3);
      sp_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descrA, vect, 0.0, Av);
      milliseconds = profile_end(&time3, &time4);
      MKLErrorHandle(sp_status, "SpMV");
      totalSpMVtime += milliseconds;
      elapsed_time = profile_end(&time1, &time2);
      iterationsSpMV++;
    }
  while(elapsed_time < END_TIME);
  printf("Total milliseconds elapsed in SpMV = %lg, Iterations=%ld\n", totalSpMVtime, iterationsSpMV);
 
#else
  iterationsSpMV = 0;
  totalSpMVtime = 0.0;

  printf("Running SpMV in a loop for %d times\n", ITERATIONS);
  do
    {
      profile_start(&time3);
      sp_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descrA, vect, 0.0, Av);
      milliseconds = profile_end(&time3, &time4);
      MKLErrorHandle(sp_status, "SpMV");
      totalSpMVtime += milliseconds;
    }
  while(++iterationsSpMV != ITERATIONS);
  SpMV_pi = totalSpMVtime/iterationsSpMV;
  
#endif
  profile_start(&time1);
  mkl_sparse_set_sv_hint(csrLU, SPARSE_OPERATION_NON_TRANSPOSE, descrL, n_iter);
  mkl_sparse_optimize(csrLU);
  iluTimeAnalysisL = profile_end(&time1, &time2);
#ifdef TIME_BASED
  printf("Running Lower triangular solve in a loop for %lg seconds\n", END_TIME);  
  iterationsL = 0;
  profile_start(&time1);
  do
    {      
      profile_start(&time3);
      sp_status = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrLU, descrL, Av, y);
      milliseconds = profile_end(&time3, &time4);
      MKLErrorHandle(sp_status, "Lower triangular solve");
      totalLtime += milliseconds;
      elapsed_time = profile_end(&time1, &time2);       
      iterationsL++;
    }
  while(elapsed_time < END_TIME);
  printf("Total milliseconds elapsed in L solve = %lg, Iterations=%ld\n", totalLtime, iterationsL);

#else
  struct timeval t1, t2;

  printf("Running Lower triangular solve in a loop for %d times\n", ITERATIONS);

  iterationsL = 0;
  totalLtime = 0.0;
   do
     {
       profile_start(&time3);
       //gettimeofday(&t1, NULL);
       sp_status = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrLU, descrL, Av, y);
       //gettimeofday(&t2, NULL);
       //milliseconds = (t2.tv_sec - t1.tv_sec)*1000.+(t2.tv_usec-t1.tv_usec)/1000.0;
       milliseconds = profile_end(&time3, &time4);
       MKLErrorHandle(sp_status, "Lower triangular solve");
       totalLtime += milliseconds;
     }
   while(++iterationsL != ITERATIONS);
   //printf("%f:%d\n", totalLtime, iterationsL);
   LT_pi = totalLtime/iterationsL;

#endif
 

  // Inspector-executor model first stage, analysis
   profile_start(&time1);
   mkl_sparse_set_sv_hint(csrLU, SPARSE_OPERATION_NON_TRANSPOSE, descrU, n_iter);
   mkl_sparse_optimize(csrLU);
   iluTimeAnalysisU = profile_end(&time1, &time2);

#ifdef TIME_BASED

  printf("Running Upper triangular solve in a loop for %lg seconds\n", END_TIME);  
  iterationsU = 0;
  profile_start(&time1);
  do
    {      
      profile_start(&time3);
      sp_status = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrLU, descrU, y, res);
      milliseconds = profile_end(&time3, &time4);
      MKLErrorHandle(sp_status, "Upper triangular solve");
      totalUtime += milliseconds;
      elapsed_time = profile_end(&time1, &time2);       
      iterationsU++;
    }
  while(elapsed_time < END_TIME);
  printf("Total milliseconds elapsed in U solve = %lg, Iterations=%ld\n", totalUtime, iterationsU);
#else

  printf("Running Upper triangular solve in a loop for %d times\n", ITERATIONS);  
  iterationsU = 0;
  totalUtime = 0.0;
  do
    {      
      profile_start(&time3);
      sp_status = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrLU, descrU, y, res);
      milliseconds = profile_end(&time3, &time4);
      MKLErrorHandle(sp_status, "Upper triangular solve");
      totalUtime += milliseconds;
    }
  while(++iterationsU != ITERATIONS);
  UT_pi = totalUtime/iterationsU;

#endif

#ifdef TIME_BASED
  printf("Running LU solve in a loop for %lg seconds\n", END_TIME);  
  profile_start(&time1);
  do
    {
      profile_start(&time3);
      sp_status = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrLU, descrL, Av, y);
      sp_status = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrLU, descrU, y, res);
      milliseconds = profile_end(&time3, &time4);
      totalLUtime += milliseconds;
      elapsed_time = profile_end(&time1, &time2);       
      iterationsLU++;
    }
  while(elapsed_time < END_TIME);
  printf("Total milliseconds elapsed in LU solve = %lg, Iterations=%ld\n", totalLUtime, iterationsLU);

#else
  printf("Running triangular solve in a loop for %d times\n", ITERATIONS);
  iterationsLU = 0;
  totalLUtime = 0.0;
  do
    {
      profile_start(&time3);
      sp_status = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrLU, descrL, Av, y);
      sp_status = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrLU, descrU, y, res);
      milliseconds = profile_end(&time3, &time4);
      totalLUtime += milliseconds;      
    }
  while(++iterationsLU != ITERATIONS);
  LUT_pi = totalLUtime/iterationsLU;

#endif

  printf("Finished test case ... \n");

#ifdef TIME_BASED
  // Writing data to output file
  logFile << totalSpMVtime << "," << iterationsSpMV << ","
	  << totalLtime << "," << iterationsL << ","
	  << totalUtime << "," << iterationsU << "," 
	  << totalLUtime << ","<< iterationsLU;
  logFile.close();
#else
  cout << SpMV_pi << " " << LT_pi << " " << UT_pi << " " << LUT_pi << " " <<
    iluTime << " " << iluTimeAnalysisL << " " <<  iluTimeAnalysisU << " " <<
    Mem_alloc_time << endl;
#endif
  return 1;
}

int Mem_Allocs(double **vect_, double **Av_, double **res_, double **bilu0_, double **y_,
		int M, int N, int nz)
{
  
  int *I, *J;
  double *vect, *Av, *res, *bilu0, *y;
  
  vect = (double *)malloc(sizeof(double) * N);
  Av = (double *)malloc(sizeof(double) * N);
  res = (double *)malloc(sizeof(double) * N);
  bilu0 = (double *)malloc(sizeof(double) * nz);
  y = (double *)malloc(sizeof(double) * N);
  
  // Initialize vect to 1
  for(int i = 0; i < N; i++)
    {
      vect[i] = 1.0;
    }
  
  *vect_ = vect;
  *Av_ = Av;
  *res_ = res;
  *bilu0_=bilu0;
  *y_=y;
  
  return 1;
}

 void MKLErrorHandle(sparse_status_t sp_status, char * msg)
{
  if (sp_status != SPARSE_STATUS_SUCCESS)
    {
      printf("Error performing requested sparse operation: %d\n", msg);
      return;
    }
  else
    {
      //printf("Success:%s\n", msg); 
    }  
}
