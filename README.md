### SpTRSV Framework: A Prediction Framework for Fast Sparse Triangular Solves

### Overview
The SpTRSV prediction framework is a tool for automated prediction of the fastest sparse triangular solve (SpTRSV) algorithm for a given input sparse matrix on a CPU-GPU platform. It comprises several modules such as sparse matrix feature extractor, SpTRSV algorithms repository, performance data collector and prediction model trainer and tester. The source codes for profiling the performance of SpTRSV algorithms and matrix feature extractor are written in C++ and NVIDIA CUDA. The script for automatically downloading the matrices from SuiteSparse collection, calling matrix feature extractor and performance data collector on each matrix and generating the feature and training datasets is written in Python. The evaluation script uses SSGet command line tool (available from https://github.com/ginkgo-project/ssget) to download the matrices from SuiteSparse collection. All the SpTRSV algorithm profiling codes, except SyncFree, have been developed by the authors. For profiling SyncFree algorithm, SyncFree CUDA code available online (available from https://github.com/bhSPARSE/Benchmark_SpTRSM_using_CSC) is used. Except for the Intel compiler C/C++ compiler and Intel MKL library used for building MKL(seq) and MKL(par) algorithms and the host code, the rest of the framework depends on freely available tools like Python and Python libraries, SSGet, libUFget library and NVIDIA CUDA toolkit. 

For the validation of the results presented in the paper, it is required to collect 30 selected features and execution time for the six SpTRSV algorithms presented for each of the 998 square sparse matrices from the SuiteSparse collection. As this may be a time consuming process with long running hours, we provide the dataset, we collected and used for the results in the accepted paper, with this artifact submission . If it is desired to re-generate the datasets, detailed instructions on installation of the required tools and generating the datasets are provided separately in data generation guide (DataGenGuide.pdf) available in the main directory.

#### Getting Started Guide

This section covers the details of how to install necessary tools to reproduce the results for the paper. Because our tool is machine learning-based and requires long running ours to produce the training and the feature data sets, we provide the dataset we used for our results with this artifact submission. Consequently, the steps mentioned in this section are required to reproduce results from the available dataset.

#### Updating System

Before beginning the tool installation process, it is recommended to have an updated Ubuntu system. Following commands are sufficient for updating Debian-based systems (including Ubuntu). Depending upon how often the system is updated, the time taken for the update may vary. Typically, the time taken is between 1 to 5 minutes.

```bash
sudo apt-get update
sudo apt-get upgrade
```

#### Installing Python

For consistency with the rest of the instructions in this guide, it is recommended to install Python version 3.7. For this purpose, the following list of command line instructions on Ubuntu should be sufficient:

```bash
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
```
Press Enter (↲) key when prompted. Once completed, continue with the following command:

```bash
sudo apt install python3.7
```

#### Installing Anaconda

Our framework depends on a number of Python libraries like Scikit-learn, Pandas, matplotlib, numpy, scipy etc. that can be installed individually on Linux. However, it is more convenient to install Anaconda data science platform that includes all of these platforms by default. The installation package for the 64-bit Individual version of Anaconda for Python3.7 can be downloaded with the following command line:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
```

Alternatively, the installation package for the latest Individual version of Anaconda can be downloaded from https://www.anaconda.com/products/individual.

Verify data integrity of the installation package and install Anaconda with the following commands:

```bash
sha256sum Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```

Follow the prompts during the installation to complete the installation. For more comprehensive details, refer to the link: https://docs.anaconda.com/anaconda/install/linux/

#### Testing Platform

All the datasets used in this evaluation are generated on an Intel CPU Gold machine with an NVIDIA V100 GPU with the following specifications:

CPU 
..................................................
CPU Model Name: Intel Xeon(R) Gold 6148 @ 2.40 GHz
Number of Sockets: 2
Number of Cores/Socket: 20
Model: 85
L1d/L1i cache: 32K each
L2 cache: 1024K
L3 cache: 28160K
RAM: 500 GB
..................................................

GPU
..................................................
GPU Model Name: NVIDIA Tesla V100 PCIe 32GB
PCI Express Interface: PCI Express 3.0 x 16
Memory: 32 GB HBM2
Peak memory bandwidth: 900 GB/s
Tensor cores: 640
CUDA cores: 5120
Double precision performance: 7 TFlops
Single precision performance: 14 TFlops
..................................................

#### Reproducing Figure 2 as on Page 4 of the accepted paper

Figure 2. shows comparison of the fastest SpTRSV on the CPU and on the GPU for a set of 37 matrices taken from SuiteSparse matrix collection. The dataset used for the figure is located in ‘./datasets/CPU_GPU_best_SpTRSV_37_matrices.csv’ file in the main directory. Generating the figure takes less than one minute. The command line to plot the figure

```bash
python evaluation.py figure2 
```
#### Reproducing Table 1 as on Page 3 of the accepted paper

Table 1 shows the breakdown of the winning SpTRSV algorithms for the 37 matrices shown in Figure 2. The dataset used for the figure is located in ‘'./datasets/CPU_GPU_SpTRSV_perf_data_37_matrices.csv'’ file in the main directory. Generating the table takes less than 30 seconds.The command line to display the table is:

```bash
python evaluation.py table1 
```
#### Reproducing Table 2 as on Page 7 of the accepted paper

Table 2 shows the selected feature set for the prediction framework and their score ranking. The dataset used for the calculating the features scores is located in './datasets/Features.csv' file in the main directory. It stores the selected 30 features for the 998 matrices from the SuiteSparse collection. To display the features scores and their ranking, the following command line should be used. Generating the table takes less than 30 seconds.

```bash
python evaluation.py table2
```
#### Reproducing Table 3 as on Page 9 of the accepted paper

Table 3 shows the breakdown of the winning SpTRSV algorithms for all the 998 matrices used in this evaluation. The dataset used for the calculating the features scores is located in './datasets/CPU_GPU_SpTRSV_comparison_full_dataset.csv' file in the main directory. To display the statistics in Table 3, the following command should be used. Generating the table takes less than 30 seconds.

```bash
python evaluation.py table3
```

#### Reproducing Table 4 as on Page 9 of the accepted paper

Table 4 shows the number of non-zero and number of rows statistics for our set of 998 matrices from the SuiteSparse collection. The dataset used for generating the table is located in './datasets/CPU_GPU_SpTRSV_comparison_full_dataset.csv' file in the main directory. Following command should be used to generate statistics in Table 4. Generating the table takes less than 30 seconds.

```bash
python evaluation.py table4
```

#### Reproducing Figure 5 as on Page 10 of the accepted paper

Figure 5 shows the cross validation scores for our classifier with all 30 features used as the feature set. It also displays the classifier performance statistics as claimed in the last paragraph on page 10. It should be noted that the figure and statistics may slightly vary from the accepted paper due to the dynamic nature of the cross-validation without harming the overall conclusions. The dataset used for generating the figure is located in './datasets/Training_data.csv' file in the main directory. Following command should be used to generate the figure. Generating the figure takes less than 30 seconds.

```bash
python evaluation.py figure5
```

#### Reproducing Figure 6 as on Page 10 of the accepted paper

Figure 6 shows the cross validation scores for our classifier with top scoring 10 features used as the feature set. It should be noted that the figure and statistics may slightly vary from the accepted paper due to the dynamic nature of the cross-validation without harming the overall conclusions. The dataset used for generating the figure is located in './datasets/Training_data.csv' file in the main directory. Following command should be used to generate the figure. Generating the figure takes less than 30 seconds.

```bash
python evaluation.py figure6
```

#### Reproducing Figure 7 as on Page 10 of the accepted paper

Figure 7 shows speedup gained by the predicted SpTRSV algorithm over the lazy algorithm choice. The dataset used for generating the figure is located in './datasets/Training_data.csv' file in the main directory. Following command should be used to generate the figure. It takes less than 30 seconds to generate the figure. It should be noted that the figure and statistics may slightly vary from the accepted paper due to the dynamic nature of the prediction.

```bash
python evaluation.py figure7
```

#### Reproducing Figure 8 as on Page 10 of the accepted paper

Figure 8 shows mean framework overhead versus mean empirical execution times for the aggressive and the lazy users for different matrix size ranges. It also displays the mean number of iterations for matrices > 1000K as claimed on line 3 on page 13. It should be noted that the statistics may slightly vary from the accepted paper due to the dynamic nature of the prediction without harming the overall conclusions. The dataset used for generating the figure is located in './datasets/Training_data.csv' file in the main directory. Following command should be used to generate the figure. It takes less than 30 seconds to generate the figure. 

```bash
python evaluation.py figure8
```

#### Copyright notice
(c) 2020 Najeeb Ahmad  
All rights reserved.   
Please read license file LICENSE on usage license. 