
#include "anamodgpu.h"

__global__ void Fill_Level(int *d_dp, int *d_dpr, int *d_dpc, int *d_nnz_rw, 
						   int *d_nnz_cw, int *d_rpl, int *d_rl_pl, int *d_cl_pl,
						   int *d_jlev, int *d_last, int *d_cntr, int *d_cntr1, int lvl, int rows)
{
   int wid = (blockIdx.x * blockDim.x + threadIdx.x);   
   if(wid >=rows)
     return;
   
   if(d_dp[wid] == 0)
   {
   	  //printf("%d, ", last);
      
      d_rl_pl[atomicAdd(d_cntr,1)] = d_dpr[wid];
      d_cl_pl[atomicAdd(d_cntr1,1)] = d_dpc[wid];
      
      //printf("%d, ", d_last);
      d_jlev[atomicAdd(d_last,1)] = wid;      
      
      atomicAdd(&d_nnz_rw[lvl], d_dpr[wid]);
      atomicAdd(&d_nnz_cw[lvl], d_dpc[wid]);
      atomicAdd(&d_rpl[lvl], 1);
      
      //atomicAdd(d_last,1);
   }   
}

__global__ void Remove_Nodes_L(int *ib, int *jb, int *d_jlev, int *d_dp, int first, int *d_last)
{

   int wid = (blockIdx.x * blockDim.x + threadIdx.x)/WARP;
   int lane = threadIdx.x & (WARP - 1);
   //printf("[%d]", *d_last);
   if((first + wid) >=*d_last)
     return;
   //printf("[%d]", wid);
   int i = d_jlev[first + wid];
   int p1 = jb[i], q1 = jb[i + 1];
                      
   for(int j = p1 + lane; j < q1; j += WARP)
   {
   	 //printf("[%d, %d], ", wid, i);
   	 if(ib[j] >= i)
     	atomicSub(&d_dp[ib[j]], 1);       
   }   
}

__global__ void Remove_Nodes_U(int *ib, int *jb, int *d_jlev, int *d_dp, int first, int *d_last)
{

   int wid = (blockIdx.x * blockDim.x + threadIdx.x)/WARP;
   int lane = threadIdx.x & (WARP - 1);
   //printf("[%d]", *d_last);
   if((first + wid) >=*d_last)
     return;
   //printf("[%d]", wid);
   int i = d_jlev[first + wid];
   int p1 = jb[i], q1 = jb[i + 1];

   for(int j = p1 + lane; j < q1; j += WARP)
   {
   	 //printf("[%d, %d], ", wid, i);
   	 if(ib[j] <= i)
     	atomicSub(&d_dp[ib[j]], 1);       
   }   
}

__global__ void Find_Level_Ruipeng(int *ib , int *jb, int *dp , int *jlev , int first , int *last, int rows) 
 {

	int wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
	
	if((first + wid) >=*last)
     return;
	//printf("[%d, %d]", first, *last);
	int lane = threadIdx.x & (WARP - 1);
	int i = jlev[first+wid];
	int p1 = jb[i], q1 = jb[i+1];
	//printf("[%d, %d, %d], ", wid, p1, q1);

	for (int j=p1+lane; j<q1; j+=WARP)
	{
		if(ib[j] > wid)
		{
			if (atomicSub (&dp[ib[j]], 1) == 0)
			{
				jlev[atomicAdd(last , 1)] = ib[j];	
			}
		}				
	}
 } 

__global__ void Find_Level0(int rows, int *dp, int *jlev , int *last) 
{

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= rows) 
		return;

	if (dp[gid] == 0)
		jlev[atomicAdd(last , 1)] = gid;  
}

__global__ void Krnl_Depn(int *ia, int *ja, int *d_dpL, int *d_dpU, int rows)
{
	int tid = (blockIdx.x * blockDim.x + threadIdx.x);

	if(tid >= rows)
    	return;

    for(int i = ia[tid]; i < ia[tid+1]; i++)
    { 
      if(ja[i] < tid)
	  	d_dpL[tid]++;

      if(ja[i] > tid)
	  	d_dpU[tid]++;
    }
}

__global__ void Krnl_Depn_UT_row(int *ia, int *ja, int *d_dpU, int rows)
{
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);

  if(tid >= rows)
      return;

    for(int i = ia[tid]; i < ia[tid+1]; i++)
    { 
      if(ja[i] > tid)
      d_dpU[tid]++;
    }
}

__global__ void Krnl_Depn_UT_col(int *ib, int *jb, int *d_dpU, int rows)
{
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);

  if(tid >= rows)
      return;

    for(int i = jb[tid]; i < jb[tid+1]; i++)
    { 
      if(ib[i] < tid)
      d_dpU[tid]++;
    }
}

__global__ void Krnl_Depn_LT_row(int *ia, int *ja, int *d_dpL, int rows)
{
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);

  if(tid >= rows)
      return;

    for(int i = ia[tid]; i < ia[tid+1]; i++)
    { 
      if(ja[i] < tid)
      d_dpL[tid]++;
    }
}

__global__ void Krnl_Depn_LT_col(int *ib, int *jb, int *d_dpL, int rows)
{
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);

  if(tid >= rows)
      return;

    for(int i = jb[tid]; i < jb[tid+1]; i++)
    { 
      if(ib[i] > tid)
      d_dpL[tid]++;
    }
}

void Calc_Depn(int *d_ia, int *d_ja, int *d_dpL, int *d_dpU, int rows)
{
	int num_threads = 128;
	int num_blocks = ceil((double)rows/ (double)num_threads);

	Krnl_Depn<<<num_blocks, num_threads>>>(d_ia, d_ja, d_dpL, d_dpU, rows);
}

void Calculate_Depn(int *d_ia, int *d_ja, int *d_dpL, int *d_ib, 
         int *d_jb, int *d_dpL_col, int rows, double *time)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0.0;
  cudaEventRecord(start);
  Calc_Depn_LT_row(d_ia, d_ja, d_dpL, rows);
  Calc_Depn_LT_col(d_ib, d_jb, d_dpL_col, rows);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  *time += milliseconds;
}

void Calc_Depn_LT_row(int *d_ia, int *d_ja, int *d_dpL, int rows)
{
  int num_threads = 128;
  int num_blocks = ceil((double)rows/ (double)num_threads);

  Krnl_Depn_LT_row<<<num_blocks, num_threads>>>(d_ia, d_ja, d_dpL, rows);
}

void Calc_Depn_LT_col(int *d_ib, int *d_jb, int *d_dpL, int rows)
{
  int num_threads = 128;
  int num_blocks = ceil((double)rows/ (double)num_threads);

  Krnl_Depn_LT_col<<<num_blocks, num_threads>>>(d_ib, d_jb, d_dpL, rows);
}

void Calc_Depn_UT_row(int *d_ia, int *d_ja, int *d_dpU, int rows)
{
  int num_threads = 128;
  int num_blocks = ceil((double)rows/ (double)num_threads);

  Krnl_Depn_UT_row<<<num_blocks, num_threads>>>(d_ia, d_ja, d_dpU, rows);
}

void Calc_Depn_UT_col(int *d_ib, int *d_jb, int *d_dpU, int rows)
{
  int num_threads = 128;
  int num_blocks = ceil((double)rows/ (double)num_threads);

  Krnl_Depn_UT_row<<<num_blocks, num_threads>>>(d_ib, d_jb, d_dpU, rows);
}


int Find_Levels_L(int *d_dp, int *d_dpr, int *d_dpc, int *d_nnz_rw, 
						   int *d_nnz_cw, int *d_rpl, int *d_rl_pl, int *d_cl_pl, int *d_jlev, 
						   int *ilev, int *d_ib, int *d_jb, int *d_last, int *d_cntr, int *d_cntr1, int rows)
{
	int h_last = 0, first;
	cudaError_t cudaStatus;
	int nlev = 0;
	int num_threads = 128;
	int warps_per_block = num_threads / WARP;
	int num_blocks = ceil((double)rows/ (double)num_threads);
	int num_blocks_1 = ceil((double)rows/ (double)warps_per_block);

	for(first = 0, nlev = 0, ilev[0] = 0; h_last < rows;)
  	{
     	Fill_Level<<<num_blocks, num_threads>>>(d_dp, d_dpr, d_dpc, d_nnz_rw, d_nnz_cw, d_rpl, 
     		                                    d_rl_pl, d_cl_pl, d_jlev, d_last, d_cntr, d_cntr1, nlev, rows);
     	cudaStatus = cudaMemcpy(&h_last, d_last, sizeof(int), cudaMemcpyDeviceToHost);
     	checkCudaErrors(cudaStatus);
     	ilev[++nlev] = h_last;
     	Remove_Nodes_L<<<num_blocks_1, num_threads>>>(d_ib, d_jb, d_jlev, d_dp, first, d_last);
     	first = h_last;   
  	}
  	return nlev;
}

int Calc_Levels_L(int *d_dp, int *d_dpr, int *d_dpc, int *d_nnz_rw, 
               int *d_nnz_cw, int *d_rpl, int *d_rl_pl, int *d_cl_pl, int *d_jlev, 
               int *ilev, int *d_ib, int *d_jb, int *d_last, int *d_cntr, int *d_cntr1, int rows, double *time)
{
  cudaEvent_t start = 0, stop = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0.0;
  cudaEventRecord(start);
  int lvls = Find_Levels_L(d_dp, d_dpr, d_dpc, d_nnz_rw, d_nnz_cw, d_rpl, 
                       d_rl_pl, d_cl_pl, d_jlev, ilev, d_ib, d_jb, d_last, d_cntr, d_cntr1, rows);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  *time += milliseconds;
  return lvls;
}


int Find_Levels_U(int *d_dp, int *d_dpr, int *d_dpc, int *d_nnz_rw, 
						   int *d_nnz_cw, int *d_rpl, int *d_rl_pl, int *d_cl_pl, int *d_jlev, 
						   int *ilev, int *d_ib, int *d_jb, int *d_last, int *d_cntr, int *d_cntr1, int rows)
{
	int h_last = 0, first;
	cudaError_t cudaStatus;
	int nlev = 0;
	int num_threads = 128;
	int warps_per_block = num_threads / WARP;
	int num_blocks = ceil((double)rows/ (double)num_threads);
	int num_blocks_1 = ceil((double)rows/ (double)warps_per_block);

	for(first = 0, nlev = 0, ilev[0] = 0; h_last < rows;)
  	{
     	Fill_Level<<<num_blocks, num_threads>>>(d_dp, d_dpr, d_dpc, d_nnz_rw, d_nnz_cw, d_rpl, 
     		                                    d_rl_pl, d_cl_pl, d_jlev, d_last, d_cntr, d_cntr1, nlev, rows);
     	cudaStatus = cudaMemcpy(&h_last, d_last, sizeof(int), cudaMemcpyDeviceToHost);
     	checkCudaErrors(cudaStatus);
     	ilev[nlev++] = h_last;
     	Remove_Nodes_U<<<num_blocks_1, num_threads>>>(d_ib, d_jb, d_jlev, d_dp, first, d_last);
     	first = h_last;   
  	}
  	return nlev;
}

int Calc_Levels_U(int *d_dp, int *d_dpr, int *d_dpc, int *d_nnz_rw, 
               int *d_nnz_cw, int *d_rpl, int *d_rl_pl, int *d_cl_pl, int *d_jlev, 
               int *ilev, int *d_ib, int *d_jb, int *d_last, int *d_cntr, int *d_cntr1, int rows, double *time)
{
  cudaEvent_t start = 0, stop = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0.0;
  cudaEventRecord(start);
  int lvls = Find_Levels_U(d_dp, d_dpr, d_dpc, d_nnz_rw, d_nnz_cw, d_rpl, 
                       d_rl_pl, d_cl_pl, d_jlev, ilev, d_ib, d_jb, d_last, d_cntr, d_cntr1, rows);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  *time += milliseconds;
  return lvls;
}

struct stdev
{
  const float meu;
  stdev(float _meu) : meu(_meu) {}

  __host__ __device__
    float operator() (const float& x) const
    {
       return (x - meu) * (x - meu);
    }
};

struct incr
{
  const int val;
  incr(int _val) : val(_val) {}

  __host__ __device__
  int operator()(const int& x) const
  {
    return (x + val);
  }
};

void CalcStats(FeatureVect *predFeatures, int *d_nnz_rw, int *d_nnz_cw, int *d_rowl_L, int *d_dpL_col, int *d_rpl,
               int *d_rl_pl, int *d_cl_pl, int *ilev, int *d_jlev, int lvls, int rows, double *time)
{
	//FeatureVect predFeatures;  
  cudaEvent_t start = 0, stop = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0.0;
  cudaEventRecord(start);
	thrust::device_ptr<int> nnz_rw(d_nnz_rw);
	thrust::device_ptr<int> nnz_cw(d_nnz_cw);
	thrust::device_ptr<int> rowl_L(d_rowl_L);
	thrust::device_ptr<int> dpL_col(d_dpL_col);
  thrust::device_ptr<int> cl_pl(d_cl_pl);
  thrust::device_ptr<int> rl_pl(d_rl_pl);
  thrust::device_ptr<int> rpl(d_rpl);
  thrust::host_vector<int> max_col_len(lvls);
  thrust::host_vector<int> max_row_len(lvls);
  thrust::host_vector<float> mean_col_len(lvls);
  thrust::host_vector<float> std_col_len(lvls);
  thrust::host_vector<float> std_rl_pl(lvls);
  thrust::host_vector<float> mean_row_len(lvls);
  thrust::host_vector<int> min_row_len(lvls);
  thrust::host_vector<float> median_rl_pl(lvls);
  
  predFeatures->m = rows;
  predFeatures->lvls = lvls;
  predFeatures->mean_rpl = (double) rows/lvls;
  predFeatures->max_nnz_pl_rw = thrust::reduce(nnz_rw, nnz_rw + lvls, (int)0, thrust::maximum<int>());
  predFeatures->max_nnz_pl_cw = thrust::reduce(nnz_cw, nnz_cw + lvls, (int)0, thrust::maximum<int>());
  predFeatures->max_rl = thrust::reduce(rowl_L, rowl_L + rows, (int)0, thrust::maximum<int>());
  predFeatures->max_cl = thrust::reduce(dpL_col, dpL_col + rows, (int)0, thrust::maximum<int>());
  predFeatures->max_rpl = thrust::reduce(rpl, rpl + lvls, (int)0, thrust::maximum<int>());
  predFeatures->nnzs = thrust::reduce(nnz_rw, nnz_rw + lvls, (int)0, thrust::plus<int>());
  predFeatures->mean_nnz_pl_rw = (double)predFeatures->nnzs/lvls;
  predFeatures->mean_rl = (double)predFeatures->nnzs/rows;
  predFeatures->max_rl_cnt = thrust::count(rowl_L, rowl_L + rows, predFeatures->max_rl);
  predFeatures->max_cl_cnt = thrust::count(dpL_col, dpL_col + rows, predFeatures->max_cl);
  predFeatures->min_cl_cnt = thrust::count(dpL_col, dpL_col + rows, 1);
  predFeatures->min_rl_cnt = thrust::count(rowl_L, rowl_L + rows, 1);
  thrust::sort(rpl, rpl + lvls);
  thrust::sort(rowl_L, rowl_L + rows);
  thrust::sort(dpL_col, dpL_col + rows);
  

  predFeatures->median_rpl = (lvls % 2)?rpl[lvls/2]:(rpl[lvls/2-1] + rpl[lvls/2])/2.0;
  predFeatures->median_rl = (lvls % 2)?rowl_L[rows/2]:(rowl_L[rows/2-1] + rowl_L[rows/2])/2.0;
  predFeatures->median_cl = (lvls % 2)?dpL_col[rows/2]:(dpL_col[rows/2-1] + dpL_col[rows/2])/2.0;

  predFeatures->std_rpl = sqrt(thrust::transform_reduce(rpl, rpl + lvls, stdev(predFeatures->mean_rpl), 
                                                      (float)0, thrust::plus<float>())/(float)lvls);
  predFeatures->std_nnz_pl_rw = sqrt(thrust::transform_reduce(nnz_rw, nnz_rw + lvls, stdev(predFeatures->mean_nnz_pl_rw), 
                                                      (float)0, thrust::plus<float>())/(float)lvls);
  predFeatures->std_rl = sqrt(thrust::transform_reduce(rowl_L, rowl_L + rows, stdev(predFeatures->mean_rl), 
                                                      (float)0, thrust::plus<float>())/(float)rows);
  predFeatures->std_cl = sqrt(thrust::transform_reduce(dpL_col, dpL_col + rows, stdev(predFeatures->mean_rl), 
                                                      (float)0, thrust::plus<float>())/(float)rows);
  //return;
  // Level length stats
    
  int lvl_length;
  for(int i = 0; i < lvls; i++)
  {
     lvl_length = ilev[i+1] - ilev[i];
     max_col_len[i] = thrust::reduce(cl_pl+ilev[i], cl_pl + ilev[i+1], (int)0, thrust::maximum<int>());
     mean_col_len[i] = thrust::reduce(cl_pl+ilev[i], cl_pl + ilev[i+1], (int)0, thrust::plus<int>())/float(lvl_length);
     std_col_len[i] = sqrt(thrust::transform_reduce(cl_pl+ilev[i], cl_pl + ilev[i+1], stdev(mean_col_len[i]), 
                                                      (float)0, thrust::plus<float>())/float(lvl_length));
     max_row_len[i] = thrust::reduce(rl_pl+ilev[i], rl_pl + ilev[i+1], (int)0, thrust::maximum<int>());
     min_row_len[i] = thrust::reduce(rl_pl+ilev[i], rl_pl + ilev[i+1], std::numeric_limits<int>::max(), thrust::minimum<int>());
     mean_row_len[i] = thrust::reduce(rl_pl+ilev[i], rl_pl + ilev[i+1], (int)0, thrust::plus<int>())/float(lvl_length);
     std_rl_pl[i] = sqrt(thrust::transform_reduce(rl_pl+ilev[i], rl_pl + ilev[i+1], stdev(mean_row_len[i]), 
                                                      (float)0, thrust::plus<float>())/float(lvl_length));
     thrust::sort(rl_pl+ilev[i], rl_pl+ilev[i+1]);
     median_rl_pl[i] = (lvl_length%2)?rl_pl[ilev[i]+lvl_length/2]:(rl_pl[ilev[i]+lvl_length/2-1] + rl_pl[ilev[i]+lvl_length/2])/2.0;
     
  }
  
  
  predFeatures->mean_max_cl_pl = thrust::reduce(max_col_len.begin(), max_col_len.end(), (int)0, thrust::plus<int>())/(float)lvls;
  predFeatures->mean_mean_cl_pl = thrust::reduce(mean_col_len.begin(), mean_col_len.end(), (float)0, thrust::plus<float>())/(float)lvls;
  predFeatures->mean_std_cl_pl = thrust::reduce(std_col_len.begin(), std_col_len.end(), (float)0, thrust::plus<float>())/(float)lvls;
  predFeatures->mean_max_rl_pl = thrust::reduce(max_row_len.begin(), max_row_len.end(), (int)0, thrust::plus<int>())/(float)lvls;
  predFeatures->mean_std_rl_pl = thrust::reduce(std_rl_pl.begin(), std_rl_pl.end(), (float)0, thrust::plus<float>())/(float)lvls;
  predFeatures->mean_mean_rl_pl = thrust::reduce(mean_row_len.begin(), mean_row_len.end(), (float)0, thrust::plus<float>())/(float)lvls;
  predFeatures->mean_min_rl_pl = thrust::reduce(min_row_len.begin(), min_row_len.end(), (float)0, thrust::plus<float>())/(float)lvls;
  predFeatures->mean_min_rl_pl = thrust::reduce(min_row_len.begin(), min_row_len.end(), (float)0, thrust::plus<float>())/(float)lvls;
  predFeatures->mean_median_rl_pl = thrust::reduce(median_rl_pl.begin(), median_rl_pl.end(), (float)0, thrust::plus<float>())/(float)lvls;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  *time += milliseconds;

}

void PrintStats(FeatureVect *predFeatures)
{

  printf("%d\n", predFeatures->m);
  printf("%d\n", predFeatures->nnzs);
  printf("%d\n", predFeatures->lvls);
  printf("%d\n", predFeatures->max_rl_cnt);
  printf("%d\n", predFeatures->max_cl_cnt);
  printf("%d\n", predFeatures->min_cl_cnt);
  printf("%d\n", predFeatures->min_rl_cnt);
  printf("%d\n", predFeatures->max_nnz_pl_rw);
  printf("%d\n", predFeatures->max_nnz_pl_cw);  
  printf("%d\n", predFeatures->max_rl);
  printf("%d\n", predFeatures->max_cl);
  printf("%f\n", predFeatures->mean_rl);
  printf("%d\n", predFeatures->max_rpl);
  printf("%f\n", predFeatures->median_rpl);
  printf("%f\n", predFeatures->median_rl);  
  printf("%f\n", predFeatures->median_cl);    
  printf("%f\n", predFeatures->mean_nnz_pl_rw);
  printf("%f\n", predFeatures->std_rpl);
  printf("%f\n", predFeatures->std_nnz_pl_rw);
  printf("%f\n", predFeatures->std_rl);
  printf("%f\n", predFeatures->std_cl);
  printf("%f\n", predFeatures->mean_rpl);
  printf("%f\n", predFeatures->mean_max_cl_pl);
  printf("%f\n", predFeatures->mean_mean_cl_pl);
  printf("%f\n", predFeatures->mean_std_cl_pl);
  printf("%f\n", predFeatures->mean_max_rl_pl);
  printf("%f\n", predFeatures->mean_std_rl_pl);
  printf("%f\n", predFeatures->mean_mean_rl_pl);
  printf("%f\n", predFeatures->mean_min_rl_pl);
  printf("%f\n", predFeatures->mean_median_rl_pl);
}

void IncrementVect(int *d_vect_src, int *d_vect_dest,  int length)
{
  thrust::device_ptr<int> vect_src(d_vect_src);
  thrust::device_ptr<int> vect_dest(d_vect_dest);
  thrust::transform(vect_src, vect_src + length, vect_dest, incr(1));
}

