from downloadmatrix import *
import numpy as np
import sys
import csv
import scipy.io as mm
import pandas as pd

class FeatureExtraction():

	def __init__( self ):
		print("Sparse Matrix Feature Extraction Tool")

	def ExtractFeatures(self, outputfilename, startmat, endmat, append=False):
		header = ['m', 'nnzs', 'lvls', 'max_rl_cnt', 'max_cl_cnt', 'min_cl_cnt', \
		          'min_rl_cnt', 'max_nnz_pl_rw', 'max_nnz_pl_cw', 'max_rl', 'max_cl', \
		          'mean_rl', 'max_rpl', 'median_rpl', 'median_rl', 'median_cl', \
		          'mean_nnz_pl_rw', 'std_rpl', 'std_nnz_pl_rw', 'std_rl', 'std_cl', \
		          'mean_rpl', 'mean_max_cl_pl', 'mean_mean_cl_pl', 'mean_std_cl_pl', \
		          'mean_max_rl_pl', 'mean_std_rl_pl', 'mean_mean_rl_pl', \
		          'mean_min_rl_pl', 'mean_median_rl_pl', 'Elapsed Time total', \
		          'timeILU','timeConv','timeDepn',\
		          'timeLevelsL','timeStatsL','matID']
		#filename = '../data/features.csv'
		start_mat = startmat
		end_mat = endmat
		matsize = 1000
		squarematrices = 0
		file_downloaded = 0
		non_zero_diagonal = 0
		have_zero_diagonal_entries = 0
		counter = 0
		csvFile = open(outputfilename, 'w')
		writer = csv.writer(csvFile)
		writer.writerow(header)
		csvFile.close()

		with open("./data/UF Sparse data set - Matrix list.csv") as f:
			f.readline()
			data = f.readlines()
			DM = DownloadMatrix()
			if end_mat > 1500:
				end_mat = len(data)
			for d in reversed(data[start_mat:end_mat]):
				spl = d.split(",")
				spl = d.split(",")
				rows = spl[5]
				cols = spl[6]
				have_zero_diagonal_entries = 0
				read_mode = 'file'
				if rows == cols and int(rows) >= matsize:
					squarematrices = squarematrices + 1
					file_downloaded = 0
					while file_downloaded == 0:
						print('Downloading matrix {}'.format(spl[1]))
						[rc,path] = DM.downloadSSGET(spl[1])
						if rc == 7:
							file_downloaded = 0
						else:
							file_downloaded = 1
					downloaded_file = path
					print('Reading MM file for {}'.format(spl[1]))
					mat_csr = mm.mmread(path)
					mat_csr = mat_csr.tocsr()
					main_diagonal = np.array(mat_csr.diagonal())
					nnzs = mat_csr.nnz
					read_mode = 'file'
					if np.count_nonzero(main_diagonal) != len(main_diagonal):
						print('Matrix {} contains zero diagonal entries'.format(spl[1]))
						have_zero_diagonal_entries = 1            	
					else:
						non_zero_diagonal = non_zero_diagonal + 1
					print('Extracting features ...')
		
					if have_zero_diagonal_entries == 1:
						read_mode = 'file'
						mat_csr.setdiag(1.0)
						nnzs = mat_csr.nnz 
					#new_nnz = mat_csr.nnz
					if read_mode == 'file':
						lengths_array = np.array(nnzs, dtype='int64')
						lengths_array = np.append(lengths_array, float(rows))
						np.savetxt('./data/val.txt', mat_csr.data, delimiter='',newline='\n')
						np.savetxt('./data/I.txt', mat_csr.indptr, delimiter='',newline='\n')
						np.savetxt('./data/J.txt', mat_csr.indices, delimiter='',newline='\n')             
						np.savetxt('./data/lengths.txt', lengths_array, delimiter='', newline='\n')
		
					print('Running GPU Matrix Analysis Module')
					result = subprocess.run(['../../src/matrix_feature_extractor/bin/AnaModGPU', spl[1], read_mode], stdout=subprocess.PIPE)
					result = result.stdout.decode().split('\n')
					result.append(spl[1])
					print(result[12:])
	                                
					csvFile = open(outputfilename, 'a+')
					writer = csv.writer(csvFile)
					if read_mode == 'uf':
						writer.writerow(result[12:])
					else:
						writer.writerow(result[12:])
					csvFile.close()
					counter += 1

class PerformanceData():
	def __init__( self ):
		print("Sparse Matrix Performance Data Collection Tool")

	def CollectPerformanceData(self, outputfilename, startmat, endmat, append=False):
		header = ['matID','MKL_seq','MKL_par','cusparse_v1','cusparse_v2_lvl', \
		          'cusparse_v2_lvl_nolvl','syncfree','winner','CPU winner','GPU winner', '2nd', \
		          '3rd','4th','5th','6th']
		header_oh = ['matID']
		csvFile = open(outputfilename, 'w')
		writer = csv.writer(csvFile)
		writer.writerow(header)
		csvFile.close()
		start_mat = startmat
		end_mat = endmat
		matsize = 1000
		squarematrices = 0
		file_downloaded = 0
		non_zero_diagonal = 0
		have_zero_diagonal_entries = 0
		counter = 0
		with open("./data/UF Sparse data set - Matrix list.csv") as f:
			f.readline()
			data = f.readlines()
			DM = DownloadMatrix()
			if end_mat > 1500:
				end_mat = len(data)
			for d in reversed(data[start_mat:end_mat]):
				spl = d.split(",")
				rows = spl[5]
				cols = spl[6]
				if rows == cols and int(rows) >= matsize:
					file_downloaded = 0
					while file_downloaded == 0:
						print('Downloading matrix {}'.format(spl[1]))
						[rc,path] = DM.downloadSSGET(spl[1])
						if rc == 7:
							file_downloaded = 0
						else:
							file_downloaded = 1
					downloaded_file = path
					print('Reading MM file for {}'.format(spl[1]))
					#[mat_csr] = DM.ReadMM(path, 'csr')
					mat_csr = mm.mmread(path)
					mat_csr = mat_csr.tocsr()
					orig_nnz = mat_csr.nnz
					main_diagonal = np.array(mat_csr.diagonal())
					if len(main_diagonal) != int(rows):
						print('Matrix {} contains non-existent diagonal entries'.format(spl[1]))                    
					    #mat_csr.setdiag(1.0)                
					if np.count_nonzero(main_diagonal) != len(main_diagonal):
						print('Matrix {} contains {} zero diagonal entries'.format(spl[1], int(rows) - np.count_nonzero(main_diagonal)))
						have_zero_diagonal_entries = 1
						mat_csr.setdiag(1.0, 0)
					else:
						non_zero_diagonal = non_zero_diagonal + 1
					print('Executing ..');
					#if have_zero_diagonal_entries == 1:
					read_mode = 'file'
					#else:
					#    read_mode = 'uf'
					perf_result = []
					perf_result.append(spl[1])
					new_nnz = mat_csr.nnz
					mat_csr.setdiag(1.0, 0)
					print('Writing CSR files ...')
					lengths_array = np.array(mat_csr.nnz, dtype='int64')
					lengths_array = np.append(lengths_array, float(rows))                   
					np.savetxt('val.txt', mat_csr.data, delimiter='',newline='\n')
					np.savetxt('I.txt', mat_csr.indptr, delimiter='',newline='\n')
					np.savetxt('J.txt', mat_csr.indices, delimiter='',newline='\n')             
					np.savetxt('lengths.txt', lengths_array, delimiter='', newline='\n')
					
					mat_csr = mat_csr.tocsc()
					print('Writing CSC files ...')
					np.savetxt('valC.txt', mat_csr.data, delimiter='',newline='\n')
					np.savetxt('IC.txt', mat_csr.indptr, delimiter='',newline='\n')
					np.savetxt('JC.txt', mat_csr.indices, delimiter='',newline='\n')             
					np.savetxt('lengthsC.txt', lengths_array, delimiter='', newline='\n')

					print('Running MKL(seq)')
					result_seq = subprocess.run(['../../src/mkl_seq/bin/MKL_seq', spl[1], read_mode], stdout=subprocess.PIPE) 
					result = result_seq.stdout.decode().split('\n')
					result = result[-2].split(' ')
					perf_result.append(result[1])

					print('Running MKL(par)')
					result_par = subprocess.run(['../../src/mkl_par/bin/MKL_par', spl[1], read_mode], stdout=subprocess.PIPE)
					result = result_par.stdout.decode().split('\n')
					result = result[-2].split(' ')
					perf_result.append(result[1])

					print('Running cuSPARSE(v1)')
					result_cus1 = subprocess.run(['../../src/cusparse_v1/bin/CUS1_lvl', spl[1], read_mode], stdout=subprocess.PIPE)
					result = result_cus1.stdout.decode().split('\n')
					result = result[-2].split(' ')
					perf_result.append(result[1])

					print('Running cuSPARSE(v2)')
					result_cus1 = subprocess.run(['../../src/cusparse_v2_lvl_nolvl/bin/CUS2_lvl', spl[1], read_mode], stdout=subprocess.PIPE)
					result = result_cus1.stdout.decode().split('\n')
					result = result[-2].split(' ')
					perf_result.append(result[1])
					perf_result.append(result[4])
					
					print('Running SyncFree')
					result = subprocess.run(['../../src/Benchmark_SpTRSM_using_CSC/SpTRSV_cuda/sptrsv', '-d','0','-rhs','1','forward', '-mtx', downloaded_file], stdout=subprocess.PIPE)
					result = result.stdout
					result = result.decode().split(' ')
					perf_result.append(result[120])
					exec_times = [float(i) for i in perf_result]
					print(exec_times)
					perf_result.append(exec_times.index(np.min(exec_times[1:])))					
					perf_result.append(exec_times.index(np.min(exec_times[1:3])))
					perf_result.append(exec_times.index(np.min(exec_times[3:7])))					
					perf_result.append(exec_times.index(sorted(exec_times[1:])[1]))   # Second minimum
					perf_result.append(exec_times.index(sorted(exec_times[1:])[2]))   # Third minimum
					perf_result.append(exec_times.index(sorted(exec_times[1:])[3]))   # Fourth minimum
					perf_result.append(exec_times.index(sorted(exec_times[1:])[4]))   # Fifth minimum
					perf_result.append(exec_times.index(sorted(exec_times[1:])[5]))   # Sixth minimum

					csvFile = open(outputfilename, 'a+')
					writer = csv.writer(csvFile)
					writer.writerow(perf_result)
					csvFile.close()

class Training():

	def __init__( self ):
		print("SpTRSV Framework - Classifier Training Tool")

	def GenTrainingData(self, featurefile, perfdatafile, trainingfile):
		featuredata = pd.read_csv(featurefile)
		perfdata = pd.read_csv(perfdatafile)

		if len(featuredata) != len(perfdata):
			print("Number of records in feature and perfdata file do not match!!")
			sys.exit()

		#trainingdata = pd.concat([featuredata,perfdata], axis=1)
		trainingdata = pd.merge(featuredata, perfdata, on='matID', how='inner')
		trainingdata.to_csv(trainingfile,index=False)



###############################################################
### Below is the main code of the program
###############################################################
if __name__ == "__main__":
	print("Matrix Feature Extraction and SpTRSV Algo performance data collection tool")

	if len(sys.argv) > 1:
		option = sys.argv[1]
		if option == "featureextraction":
			fe = FeatureExtraction()
			fe.ExtractFeatures('./data/features.csv',0,2000,False)

		if option == "perfdata":
			perfd = PerformanceData()
			perfd.CollectPerformanceData('./data/perfdata.csv', 0, 2000, False)

		if option == "trainingfile":
			tr = Training()
			tr.GenTrainingData('./data/features.csv', './data/perfdata.csv', './data/Training_data.csv')

