import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import numpy as np
import scipy as sp

class CPUGPUComparison():
	def __init__( self ):
		print('CPU GPU SpTRSV performance comparison\n')

	def DrawComparisonTable(self, filename):
		perf_dataset = pd.read_csv(filename)
		winner_df = perf_dataset.idxmin(axis=1)
		winner_counts = winner_df.value_counts()
		norm_winner_counts = winner_df.value_counts(normalize=True)*100
		print("   ----------------------------------------------------------------------------------------------------")
		print("   |%15s%35s%32s%15s |" % ("Architecture |","SpTRSV implementation |","Winner for # of matrices |",\
			"Percentage"))
		print("   ----------------------------------------------------------------------------------------------------")
		print("   |%15s%35s%30d%s%13.2f %% |" % ("CPU |","MKL(seq) |", winner_counts['mkl_seq']," |",norm_winner_counts['mkl_seq']))
		print("   |%15s%35s%30d%s%13.2f %% |" % ("|","MKL(par) |", winner_counts['mkl_par']," |",norm_winner_counts['mkl_par']))
		print("   ----------------------------------------------------------------------------------------------------")
		print("   |%15s%35s%30d%s%13.2f %% |" % ("GPU |","cuSPARSE(v1) |", winner_counts['cusparse_v1']," |",norm_winner_counts['cusparse_v1']))
		print("   |%15s%35s%30d%s%13.2f %% |" % ("|","cuSPARSE(v2)(level-sch.) |", winner_counts['cusparse_v2_lvl']," |",norm_winner_counts['cusparse_v2_lvl']))
		print("   |%15s%35s%30d%s%13.2f %% |" % ("|","cuSPARSE(v2)(no level sch.) |", winner_counts['cusparse_v2_nolvl']," |",norm_winner_counts['cusparse_v2_nolvl']))
		print("   |%15s%35s%30d%s%13.2f %% |" % ("|","Sync-Free |", winner_counts['syncfree']," |",norm_winner_counts['syncfree']))
		print("   ----------------------------------------------------------------------------------------------------")

	def DrawStatsTable(self, filename):
		stats_dataset = pd.read_csv(filename)
		ds_median = stats_dataset.median()
		ds_min = stats_dataset.min()
		ds_max = stats_dataset.max()
		min_rows = ds_min['rows']/1000
		median_rows = ds_median['rows']/1000
		max_rows = ds_max['rows']/1000000
		min_nnzs = ds_min['nnzs']/1000
		median_nnzs = ds_median['nnzs']/1000
		max_nnzs = ds_max['nnzs']/1000000

		print('   ---------------------------------------------------------------------')
		print("   |%20s%16s%16s%16s"%(" |","Minimum |", "Median |","Maximum |"))
		print('   ---------------------------------------------------------------------')
		print("   |%20s%13.2fK%s%13.2fK%s%13.2fM%s"%("Number of rows |",min_rows," |", median_rows," |",max_rows, " |"))
		print('   ---------------------------------------------------------------------')
		print("   |%20s%13.3fK%s%13.3fK%s%13.3fM%s"%("Number of nonzeros |",min_nnzs, " |",median_nnzs, " |", max_nnzs," |"))
		print('   ---------------------------------------------------------------------')
		


	def DrawFigure(self, filename):
		perf_data = pd.read_csv(filename)
		perf_data.to_json("temp.json", orient='records')
		with open("temp.json", "r") as filename:
			V100_Gold_dataset_json = json.load(filename)
		V100_Gold_json_formatted = []
		for i in range(0, 37):
			V100_Gold_json_formatted.append({
			"Platform 1": V100_Gold_dataset_json[i]["Platform"],
			"Matrix 1": V100_Gold_dataset_json[i]["Matrix ID"],
			"Execution Time 1": V100_Gold_dataset_json[i]["Execution Time"],
			"Degree of Parallelism 1":V100_Gold_dataset_json[i]["Degree of Parallelism"],
			"Winner 1":V100_Gold_dataset_json[i]["Winner"],
			"Platform 2": V100_Gold_dataset_json[i+37]["Platform"],
			"Matrix 2": V100_Gold_dataset_json[i+37]["Matrix ID"],
			"Execution Time 2": V100_Gold_dataset_json[i+37]["Execution Time"],
			"Degree of Parallelism 2":V100_Gold_dataset_json[i]["Degree of Parallelism"],
			"Winner 2": V100_Gold_dataset_json[i+37]["Winner"]})

		V100_Gold_json_formatted = sorted(V100_Gold_json_formatted, key = lambda i: (i['Winner 1'], i['Degree of Parallelism 1']))
		V100_Gold_json_sorted = []
		V100_Gold_Matrix = []

		for i in range(0, 37):
			V100_Gold_json_sorted.append({
			"Platform": V100_Gold_json_formatted[i]["Platform 1"],
			"Matrix ID": V100_Gold_json_formatted[i]["Matrix 1"],
			"Degree of Parallelism": V100_Gold_json_formatted[i]["Degree of Parallelism 1"],
			"Execution Time": V100_Gold_json_formatted[i]["Execution Time 1"],
			})
			V100_Gold_Matrix.append(V100_Gold_json_formatted[i]["Matrix 1"])

		for i in range(0, 37):
		    V100_Gold_json_sorted.append({
		        "Platform": V100_Gold_json_formatted[i]["Platform 2"],
		        "Matrix ID": V100_Gold_json_formatted[i]["Matrix 2"],
		        "Degree of Parallelism": V100_Gold_json_formatted[i]["Degree of Parallelism 2"],
		        "Execution Time": V100_Gold_json_formatted[i]["Execution Time 2"],
		    })
		with open("temp2.json", "w") as file2:
			json.dump(V100_Gold_json_sorted, file2)

		V100_Gold = pd.read_json('temp2.json', orient='records')
		plt.figure(figsize=(15,5))
		p1 = sns.barplot(x="Matrix ID",y="Execution Time",hue="Platform", data=V100_Gold,palette = "magma", edgecolor = 'w', order=V100_Gold_Matrix)
		
		sns.set(font_scale = 1.3)
		sns.set_style("white")
		p1.set_yscale("log")
		p1.set_xticklabels(p1.get_xticklabels(), rotation=90)
		ax1 = p1.axes
		ax1.set(xticklabels=V100_Gold["Degree of Parallelism"])
		ax1.axvline(12.5, ls='--', lw=1.8)
		ax1.text(1.0, 200, "GPU winners: 24")
		ax1.text(1.0, 120, "CPU winners: 13")
		p1.set_xlabel("Matrix degree of parallelism (DoP)")
		p1.set_ylabel("Lower triangular solve time (msec)")
		legend = p1.legend()
		legend.texts[0].set_text("NVIDIA V100")
		legend.texts[1].set_text("Intel Gold")
		plt.legend(loc='upper right')
		plt.setp(ax1.xaxis.get_majorticklabels(), ha='center')
		fig1 = p1.get_figure()
		fig1.set_rasterized(True)
		fig1.savefig('./datasets/figure2.eps', bbox_inches='tight',rasterized=True)
		print("Figure 2 saved in datasets directory as figure2.eps")
		plt.show()
				

class FeatureSelection():
	def __init__( self ):
		print('Feature Selection\n')

	def PrintAllFeatures(self, filename):
		features =	pd.read_csv(filename)
		for col in features.columns:
			print(col)

	def FeatureRanking(self, filename):
		features_data =	pd.read_csv(filename)
		features = features_data.drop(['winner'], axis = 1)
		target = features_data['winner']
		features=features[:-2]
		target=target[:-2]
		KBestFeatures = SelectKBest(score_func=chi2, k=30)
		fit = KBestFeatures.fit(features, target)
		rank = [i+1 for i in range(30)]
		rank_dict = {'Rank':rank}
		rank_df = pd.DataFrame(data=rank_dict)
		feature_dict = {'Feature':features.columns, 'Score':fit.scores_}
		feature_df = pd.DataFrame(data=feature_dict)
		desc = ['Number of rows', 'Number of non-zeros','Number of levels', \
				'Maximum row length count', 'Maximum column length count', "Minimum column length count", \
				'Minimum row length count', 'Maximum non-zeros per level row-wise', \
				'Maximum non-zeros per level column-wise', 'Maximum row length', \
				'Maximum column length', 'Mean row-length',\
				'Maximum rows per level','Median rows per level', \
				'Median row length', 'Median column length', \
				'Mean non-zeros per level row-wise', 'Standard deviation rows per level', \
				'Standard deviation non-zeros per level row-wise', 'Standard deviation rows length', \
				'Standard deviation column length','Mean rows per level', 'Mean max column length per level', \
				'Mean mean column length per level', 'Mean std. deviation column length per level', \
				'Mean maximum row length per level','Mean standard deviation row length per level',\
				'Mean mean row length per level','Mean minimum row length per level',\
				'Mean median row length per level']
		feature_df['Description'] = desc
		feature_df_sorted = feature_df.nlargest(30, 'Score')
		feature_df_sorted.reset_index(drop=True,inplace=True)
		feature_df_sorted.index += 1
		print(feature_df_sorted.to_string(index=True))

class Prediction():
	def __init__( self ):
		print('Prediction\n')

	def CrossValidation(self, filename, mode):
		training_data = pd.read_csv(filename)

		if mode == 1:  # Traning set for 10 features
			X = training_data.drop(['min_rl_cnt','mean_rpl','median_rpl','max_cl','lvls','std_rpl', \
				'mean_max_cl_pl','mean_mean_cl_pl','max_rl','mean_std_cl_pl','mean_max_rl_pl',\
				'std_cl','mean_std_rl_pl','mean_mean_rl_pl','mean_median_rl_pl','mean_min_rl_pl',\
				'mean_rl','median_rl','median_cl','std_rl','mkl_seq','mkl_par','cusparse_v1',\
				'cusparse_v2_lvl','cusparse_v2_nolvl','syncfree','winner','CPU winner','GPU winner',\
				'2nd','3rd','4th','5th','6th'], axis=1)
		else: # Traning set for 30 features
			X = training_data.drop(['mkl_seq','mkl_par','cusparse_v1','cusparse_v2_lvl', \
				'cusparse_v2_nolvl','syncfree','winner','CPU winner','GPU winner','2nd',\
				'3rd','4th','5th','6th'], axis=1)
		y = training_data['winner']
		sc = StandardScaler()
		X_scaled = sc.fit_transform(X)
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=44)
		rfc_algo_selection = RandomForestClassifier(n_estimators=300)
		rfc_algo_selection.fit(X_train, y_train)
		pred_rfc_algo_selection = rfc_algo_selection.predict(X_test)
		seed = 10
		cv_results = []
		accuracy = 'accuracy'
		precision = 'precision_weighted'
		recall = 'recall_weighted'
		f1_score = 'f1_weighted'
		test_precision = 'test_precision_weighted'
		test_recall = 'test_recall_weighted'
		test_f1 = 'test_f1_weighted'
		test_accuracy = 'test_accuracy'
		warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
		scoring = [accuracy, precision, recall,f1_score]
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		with warnings.catch_warnings():
			scores = model_selection.cross_validate(rfc_algo_selection, X_scaled, y, cv=kfold,scoring=scoring)
		cv_results.append(scores[test_accuracy])
		cv_results.append(scores[test_precision])
		cv_results.append(scores[test_recall])
		cv_results.append(scores[test_f1])
		print('Mean accuracy: %0.1f %%' % (cv_results[0].mean()*100.0))
		print('Mean precision: %0.1f %%' % (cv_results[1].mean()*100.0))
		print('Mean recall: %0.1f %%' % (cv_results[2].mean()*100.0))
		print('Mean f1-score: %0.1f %%' % (cv_results[3].mean()*100.0))
		print('Median accuracy: %0.1f %%' % (np.median(cv_results[0])*100.0))
		print('Median precision: %0.1f %%' % (np.median(cv_results[1])*100.0))
		print('Median recall: %0.1f %%' % (np.median(cv_results[2])*100.0))
		print('Median f1-score: %0.1f %%\n' % (np.median(cv_results[3])*100.0))
		labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
		ax1 = sns.boxplot(y=cv_results,x=labels, showmeans=True, fliersize=1,meanprops={"marker":"D","markerfacecolor":"yellow", "markeredgecolor":"none"})
		sns.set(font_scale=1.3)
		sns.set_style("white")
		vals = ax1.get_yticks()
		ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
		myfigure = ax1.get_figure()

		if mode == 1:
			myfigure.savefig('./datasets/figure6.png',bbox_inches='tight')
			print("Figure 8 saved in datasets as figure8.eps")
			print("Note: Statistics can slightly vary from Figure 8 and from run-to-run")	
		else:
			myfigure.savefig('./datasets/figure7.eps',bbox_inches='tight')
			myfigure.show()
			print("Figure 7 saved in datasets as figure7.eps")
			print("Note: Statistics can slightly vary from Figure 7 and from run-to-run")
		plt.show()
class Performance():
	def __init__( self ):
		print('Performance Results\n')

	def Speedup(self, filename):
		training_data = pd.read_csv(filename)
		X = training_data.drop(['mkl_seq','mkl_par','cusparse_v1','cusparse_v2_lvl', \
				'cusparse_v2_nolvl','syncfree','winner','CPU winner','GPU winner','2nd',\
				'3rd','4th','5th','6th'], axis=1)
		y = training_data['winner']
		sc = StandardScaler()
		X_scaled = sc.fit_transform(X)
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=44)
		rfc_algo_selection = RandomForestClassifier(n_estimators=300)
		rfc_algo_selection.fit(X_train, y_train)
		pred_rfc_algo_selection = rfc_algo_selection.predict(X_test)
		seed = 10
		precision = 'precision_weighted'
		recall = 'recall_weighted'
		f1_score = 'f1_weighted'
		scoring = [precision, recall,f1_score]
		kfold = model_selection.KFold(n_splits=10)
		cross_validate_pred = model_selection.cross_val_predict(rfc_algo_selection, X_scaled, y, cv=kfold)
		MKL_seq = training_data['mkl_seq']
		MKL_par = training_data['mkl_par']
		cus1 = training_data['cusparse_v1']
		cus2_lvl = training_data['cusparse_v2_lvl']
		cus2_nolvl = training_data['cusparse_v2_nolvl']
		syncfree = training_data['syncfree']
		algo_labels = {0:'MKL(seq)', 1:'MKL(par)', 2:'cuSPARSE(v1)', \
						3:'cuSPARSE(v2)(level-sch.)',4:'cuSPARSE(v2)(no level-sch.)',5:'Sync-Free'}
		Gain_vs_MKL_seq = []
		Gain_vs_MKL_par = []
		Gain_vs_cus1 = []
		Gain_vs_cus2_lvl = []
		Gain_vs_cus2_nolvl = []
		Gain_vs_syncfree = []
		i = 0

		for val in cross_validate_pred:
		    if val == 1:
		        predicted_time = MKL_seq[i]
		    if val == 2:
		        predicted_time = MKL_par[i]
		    if val == 3:
		        predicted_time = cus1[i]
		    if val == 4:
		        predicted_time = cus2_lvl[i]
		    if val == 5:
		        predicted_time = cus2_nolvl[i]
		    if val == 6:
		        predicted_time = syncfree[i]
		    
		    Gain_vs_MKL_seq.append(MKL_seq[i]/predicted_time)
		    Gain_vs_MKL_par.append(MKL_par[i]/predicted_time)
		    Gain_vs_cus1.append(cus1[i]/predicted_time)
		    Gain_vs_cus2_lvl.append(cus2_lvl[i]/predicted_time)
		    Gain_vs_cus2_nolvl.append(cus2_nolvl[i]/predicted_time)            
		    Gain_vs_syncfree.append(syncfree[i]/predicted_time)
		    i = i + 1

		predicted_speedup=[]
		predicted_speedup.append(Gain_vs_MKL_seq)
		predicted_speedup.append(Gain_vs_MKL_par)
		predicted_speedup.append(Gain_vs_cus1)
		predicted_speedup.append(Gain_vs_cus2_lvl)
		predicted_speedup.append(Gain_vs_cus2_nolvl)
		predicted_speedup.append(Gain_vs_syncfree)

		speedup_g2 = []
		speedup_l1 = []
		counter = 0
		counter_l = 0
		counter_l95 = 0

		for i in range(6):
		    for x in predicted_speedup[i]:
		        if x >= 1:
		            counter = counter + 1
		        if x < 1:
		            counter_l = counter_l + 1
		        if x < 0.95:
		            counter_l95 = counter_l95 + 1
		    speedup_g2.append(counter/998*100)
		    speedup_l1.append(counter_l/998*100)
		    counter = 0
		    counter_l = 0
		    counter_l95 = 0

		sns.set(font_scale=1.0)
		sns.set_style("white")
		fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 4.5))
		fig.set_rasterized(True)
		k = 0

		for i in range(2):    
			for j in range(3):
				#my_bins = [0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,int(np.max(predicted_speedup[k]))]
				max_ps = np.max(predicted_speedup[k])
				my_bins = np.arange(0, 75)
				clrs=['#CB4335' if (x < 1) else '#2874A6' for x in my_bins]				
				plot = sns.distplot(predicted_speedup[k], \
					bins=my_bins, ax=ax[i][j],kde=False)
				sns.color_palette("husl", 8)
				ax1 = plot.axes
				for rec, clr in zip(ax1.patches, clrs):
					rec.set_color(clr)
				
				props = dict(boxstyle='round', facecolor='none', alpha=0.5)
				ax1.text(0.55, 0.70, ">=1: %.1f%%"%(speedup_g2[k]), transform=ax1.transAxes, fontsize=12,
				verticalalignment='top', bbox=props)
				ax1.text(0.55, 0.85, "Mean: %.1f"%(sp.stats.hmean(predicted_speedup[k])), transform=ax1.transAxes, fontsize=12,
				verticalalignment='top', bbox=props)
				z_critical = sp.stats.norm.ppf(q = 0.95)  # Get the z-critical value*
				pop_stdev = np.std(predicted_speedup[k])
				hmean = sp.stats.hmean(predicted_speedup[k])
				mean_m_x = [(hmean-x) for x in predicted_speedup]
				mean_m_x = [np.sqrt(x*x) for x in mean_m_x]
				sample_size = len(predicted_speedup[k])
				h_std = np.sum(mean_m_x)/sample_size
				margin_of_error = z_critical * (pop_stdev/np.sqrt(sample_size))
				plot.set_yscale("log")
				#if k >= 3:
				plot.set_xlabel("Speedup")
				plot.set_title(algo_labels[k],loc="left")
				if k == 0 or k == 3:
				    plot.set_ylabel('Number of matrices')
				k = k + 1		        
		plt.tight_layout()
		warnings.filterwarnings("ignore")
		with warnings.catch_warnings():
			fig.savefig('./datasets/figure9.pdf',bbox_inches='tight',rasterized=True)
			print("Figure 9 saved in datasets as figure9.eps")
			print("Note: Statistics can slightly vary from Figure 9 and from run-to-run")
		#plt.show()

	def Overheads(self, filename_training, filename_overhead):
		training_data=pd.read_csv(filename_training)
		overhead_data=pd.read_csv(filename_overhead)
		FE_wo_ilu = overhead_data['FE_oh_wo']      # Feature extraction (FE) overhead without ILU factorization time included
		FE_w_ilu = overhead_data['FE_oh_w']        # Feature extraction (FE) ovheread with ILU factorization time included
		m=overhead_data['m']                       # Number of rows
		MKL_seq = training_data['mkl_seq']
		MKL_par = training_data['mkl_par']
		cus1 = training_data['cusparse_v1']
		cus2_lvl = training_data['cusparse_v2_lvl']
		cus2_nolvl = training_data['cusparse_v2_nolvl']
		syncfree = training_data['syncfree'] 
		seed = 250
		precision = 'precision_weighted'
		recall = 'recall_weighted'
		f1_score = 'f1_weighted'
		scoring = [precision, recall,f1_score]
		X = training_data.drop(['mkl_seq','mkl_par','cusparse_v1','cusparse_v2_lvl','cusparse_v2_nolvl','syncfree','winner','CPU winner','GPU winner','2nd','3rd','4th','5th','6th'], axis=1)
		y = training_data['winner']
		sc = StandardScaler()
		X_scaled = sc.fit_transform(X)
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=44)
		rfc_algo_selection = RandomForestClassifier(n_estimators=300)
		rfc_algo_selection.fit(X_train, y_train)
		kfold = model_selection.KFold(n_splits=10)
		cross_validate_pred = model_selection.cross_val_predict(rfc_algo_selection, X_scaled, y, cv=kfold)
		
		L_calls_vs_FE_wo_100K = []     # FE overhead in terms of lower triangular solve iterations without ILU factorization time included for matrices upto 100K rows
		L_calls_vs_FE_w_100K = []      # FE overhead in terms of lower triangular solve iterations with ILU factorization time included for matrices upto 100K rows
		L_calls_vs_FE_wo_1000K = []    # FE overhead in terms of lower triangular solve iterations without ILU factorization time included for matrices from 100K-1000K rows
		L_calls_vs_FE_w_1000K = []     # FE overhead in terms of lower triangular solve iterations with ILU factorization time included for matrices from 100K-1000K rows
		L_calls_vs_FE_wo_g1000K = []   # FE overhead in terms of lower triangular solve iterations without ILU factorization time included for matrices > 1000K rows
		L_calls_vs_FE_w_g1000K = []    # FE overhead in terms of lower triangular solve iterations with ILU factorization time included for matrices > 1000K rows

		oh_FE_wo_100K = []             # FE overhead without ILU factorization time included for matrices upto 100K
		oh_FE_w_100K = []              # FE overhead with ILU factorization time included for matrices upto 100K
		oh_FE_wo_1000K = []            # FE overhead without ILU factorization time included for matrices upto 100K-1000K
		oh_FE_w_1000K = []             # FE overhead with ILU factorization time included for matrices upto 100K-1000K
		oh_FE_wo_g1000K = []           # FE overhead without ILU factorization time included for matrices > 1000K
		oh_FE_w_g1000K = []            # FE overhead without ILU factorization time included for matrices > 1000K

		oh_MKLs_wo_100K = []           # MKL(ser) overhead without ILU factorization time included for matrices upto 100K
		oh_MKLs_w_100K = []            # MKL(ser) overhead with ILU factorization time included for matrices upto 100K
		oh_MKLp_wo_100K = []           # MKL(par) overhead without ILU factorization time included for matrices upto 100K
		oh_MKLp_w_100K = []            # MKL(par) overhead with ILU factorization time included for matrices upto 100K
		oh_CUS1_wo_100K = []           # cuSPARSE(v1) overhead without ILU factorization time included for matrices upto 100K
		oh_CUS1_w_100K = []            # cuSPARSE(v1) overhead with ILU factorization time include for matrices upto 100K
		oh_CUS2lvl_wo_100K = []        # cuSPARSE(v2)(level-sch.) overhead without ILU factorization time included for matrices upto 100K
		oh_CUS2lvl_w_100K = []         # cuSPARSE(v2)(level-sch.) overhead with ILU factorization time included for matrices upto 100K         
		oh_CUS2nolvl_wo_100K = []      # cuSPARSE(v2)(no level-sch.) overhead without ILU factorization time included for matrices upto 100K
		oh_CUS2nolvl_w_100K = []       # cuSPARSE(v2)(level-sch.) overhead with ILU factorization time included for matrices upto 100K
		oh_SyncFree_wo_100K = []       # SyncFree overhead without ILU factorization time included for matrices upto 100K
		oh_SyncFree_w_100K = []        # cuSPARSE(v2)(level-sch.) overhead with ILU factorization time included for matrices upto 100K

		oh_MKLs_wo_1000K = []          # MKL(ser) overhead without ILU factorization time included for matrices from 100K-1000K                
		oh_MKLs_w_1000K = []		   # MKL(ser) overhead with ILU factorization time included for matrices from 100K-1000K
		oh_MKLp_wo_1000K = []		   # MKL(par) overhead without ILU factorization time included for matrices from 100K-1000K
		oh_MKLp_w_1000K = []		   # MKL(par) overhead with ILU factorization time included for matrices from 100K-1000K
		oh_CUS1_wo_1000K = []		   # cuSPARSE(v1) overhead without ILU factorization time included for matrices from 100K-1000K
		oh_CUS1_w_1000K = []		   # cuSPARSE(v1) overhead with ILU factorization time include for matrices from 100K-1000K
		oh_CUS2lvl_wo_1000K = []	   # cuSPARSE(v2)(level-sch.) overhead without ILU factorization time included for matrices from 100K-1000K
		oh_CUS2lvl_w_1000K = []		   # cuSPARSE(v2)(level-sch.) overhead with ILU factorization time included for matrices from 100K-1000K         
		oh_CUS2nolvl_wo_1000K = []	   # cuSPARSE(v2)(no level-sch.) overhead without ILU factorization time included for matrices from 100K-1000K
		oh_CUS2nolvl_w_1000K = []	   # cuSPARSE(v2)(level-sch.) overhead with ILU factorization time included for matrices from 100K-1000K
		oh_SyncFree_wo_1000K = []	   # SyncFree overhead without ILU factorization time included for matrices from 100K-1000K
		oh_SyncFree_w_1000K = []	   # cuSPARSE(v2)(level-sch.) overhead with ILU factorization time included for matrices from 100K-1000K

		oh_MKLs_wo_g1000K = []  	   # MKL(ser) overhead without ILU factorization time included for matrices > 1000K       
		oh_MKLs_w_g1000K = []		   # MKL(ser) overhead with ILU factorization time included for matrices > 1000K
		oh_MKLp_wo_g1000K = []		   # MKL(par) overhead without ILU factorization time included for matrices > 1000K
		oh_MKLp_w_g1000K = []          # MKL(par) overhead with ILU factorization time included for matrices > 1000K
		oh_CUS1_wo_g1000K = []         # cuSPARSE(v1) overhead without ILU factorization time included for matrices > 1000K
		oh_CUS1_w_g1000K = []          # cuSPARSE(v1) overhead with ILU factorization time include for matrices > 1000K
		oh_CUS2lvl_wo_g1000K = []      # cuSPARSE(v2)(level-sch.) overhead without ILU factorization time included for matrices > 1000K
		oh_CUS2lvl_w_g1000K = []       # cuSPARSE(v2)(level-sch.) overhead with ILU factorization time included for matrices > 1000K         
		oh_CUS2nolvl_wo_g1000K = []    # cuSPARSE(v2)(no level-sch.) overhead without ILU factorization time included for matrices > 1000K
		oh_CUS2nolvl_w_g1000K = []     # cuSPARSE(v2)(level-sch.) overhead with ILU factorization time included for matrices > 1000K
		oh_SyncFree_wo_g1000K = []     # SyncFree overhead without ILU factorization time included for matrices > 1000K
		oh_SyncFree_w_g1000K = []      # cuSPARSE(v2)(level-sch.) overhead with ILU factorization time included for matrices > 1000K

		oh_MKLs_wo_100K_ana = []       # MKL(ser) algorithm analysis overhead without ILU factorization time included for matrices upto 100K
		oh_MKLs_w_100K_ana = []        # MKL(ser) algorithm analysis overhead with ILU factorization time included for matrices upto 100K
		oh_MKLp_wo_100K_ana = []       # MKL(par) algorithm analysis overhead without ILU factorization time included for matrices upto 100K
		oh_MKLp_w_100K_ana = []        # MKL(par) algorithm analysis overhead with ILU factorization time included for matrices upto 100K
		oh_CUS1_wo_100K_ana = []       # cuSPARSE(v1) algorithm analysis overhead without ILU factorization time included for matrices upto 100K
		oh_CUS1_w_100K_ana = []        # cuSPARSE(v1) algorithm analysis overhead with ILU factorization time include for matrices upto 100K
		oh_CUS2lvl_wo_100K_ana = []    # cuSPARSE(v2)(level-sch.) algorithm analysis overhead without ILU factorization time included for matrices upto 100K
		oh_CUS2lvl_w_100K_ana = []     # cuSPARSE(v2)(level-sch.) algorithm analysis overhead with ILU factorization time included for matrices upto 100K         
		oh_CUS2nolvl_wo_100K_ana = []  # cuSPARSE(v2)(no level-sch.) algorithm analysis overhead without ILU factorization time included for matrices upto 100K
		oh_CUS2nolvl_w_100K_ana = []   # cuSPARSE(v2)(level-sch.) algorithm analysis overhead with ILU factorization time included for matrices upto 100K
		oh_SyncFree_wo_100K_ana = []   # SyncFree algorithm analysis overhead without ILU factorization time included for matrices upto 100K
		oh_SyncFree_w_100K_ana = []    # cuSPARSE(v2)(level-sch.) algorithm analysis overhead with ILU factorization time included for matrices upto 100K

		oh_MKLs_wo_1000K_ana = []      # MKL(ser) algorithm analysis overhead without ILU factorization time included for matrices from 100K-1000K
		oh_MKLs_w_1000K_ana = []       # MKL(ser) algorithm analysis overhead with ILU factorization time included for matrices from 100K-1000K
		oh_MKLp_wo_1000K_ana = []      # MKL(par) algorithm analysis overhead without ILU factorization time included for matrices from 100K-1000K
		oh_MKLp_w_1000K_ana = []       # MKL(par) algorithm analysis overhead with ILU factorization time included for matrices from 100K-1000K
		oh_CUS1_wo_1000K_ana = []      # cuSPARSE(v1) algorithm analysis overhead without ILU factorization time included for matrices from 100K-1000K
		oh_CUS1_w_1000K_ana = []       # cuSPARSE(v1) algorithm analysis overhead with ILU factorization time include for matrices from 100K-1000K
		oh_CUS2lvl_wo_1000K_ana = []   # cuSPARSE(v2)(level-sch.) algorithm analysis overhead without ILU factorization time included for matrices from 100K-1000K
		oh_CUS2lvl_w_1000K_ana = []    # cuSPARSE(v2)(level-sch.) algorithm analysis overhead with ILU factorization time included for matrices from 100K-1000K         
		oh_CUS2nolvl_wo_1000K_ana = [] # cuSPARSE(v2)(no level-sch.) algorithm analysis overhead without ILU factorization time included for matrices from 100K-1000K
		oh_CUS2nolvl_w_1000K_ana = []  # cuSPARSE(v2)(level-sch.) algorithm analysis overhead with ILU factorization time included for matrices from 100K-1000K
		oh_SyncFree_wo_1000K_ana = []  # SyncFree algorithm analysis overhead without ILU factorization time included for matrices from 100K-1000K
		oh_SyncFree_w_1000K_ana = []   # cuSPARSE(v2)(level-sch.) algorithm analysis overhead with ILU factorization time included for matrices from 100K-1000K

		oh_MKLs_wo_g1000K_ana = []     # MKL(ser) algorithm analysis overhead without ILU factorization time included for matrices > 1000K
		oh_MKLs_w_g1000K_ana = []      # MKL(ser) algorithm analysis overhead with ILU factorization time included for matrices > 1000K
		oh_MKLp_wo_g1000K_ana = []     # MKL(par) algorithm analysis overhead without ILU factorization time included for matrices > 1000K
		oh_MKLp_w_g1000K_ana = []      # MKL(par) algorithm analysis overhead with ILU factorization time included for matrices > 1000K
		oh_CUS1_wo_g1000K_ana = []     # cuSPARSE(v1) algorithm analysis overhead without ILU factorization time included for matrices > 1000K
		oh_CUS1_w_g1000K_ana = []      # cuSPARSE(v1) algorithm analysis overhead with ILU factorization time include for matrices > 1000K
		oh_CUS2lvl_wo_g1000K_ana = []  # cuSPARSE(v2)(level-sch.) algorithm analysis overhead without ILU factorization time included for matrices > 1000K
		oh_CUS2lvl_w_g1000K_ana = []   # cuSPARSE(v2)(level-sch.) algorithm analysis overhead with ILU factorization time included for matrices > 1000K         
		oh_CUS2nolvl_wo_g1000K_ana = [] # cuSPARSE(v2)(no level-sch.) algorithm analysis overhead without ILU factorization time included for matrices > 1000K
		oh_CUS2nolvl_w_g1000K_ana = [] # cuSPARSE(v2)(level-sch.) algorithm analysis overhead with ILU factorization time included for matrices > 1000K
		oh_SyncFree_wo_g1000K_ana = [] # SyncFree algorithm analysis overhead without ILU factorization time included for matrices > 1000K
		oh_SyncFree_w_g1000K_ana = []  # cuSPARSE(v2)(level-sch.) algorithm analysis overhead with ILU factorization time included for matrices > 1000K


		emp_oh_wo_100K = 0             # Empirical execution overhead without ILU factorization time included for matrices upto 100K
		emp_oh_wo_1000k = 0            # Empirical execution overhead without ILU factorization time included for matrices from 100K-1000K
		emp_oh_wo_g1000k = 0           # Empirical execution overhead without ILU factorization time included for matrices > 1000K
		emp_oh_w_100K = 0              # Empirical execution overhead with ILU factorization time included for matrices upto 100K
		emp_oh_w_1000k = 0             # Empirical execution overhead with ILU factorization time included for matrices from 100K-1000K
		emp_oh_w_g1000k = 0            # Empirical execution overhead with ILU factorization time included for matrices > 1000K
		
		i = 0
		for val in cross_validate_pred:
		    if val == 1:
		        predicted_time = MKL_seq[i]
		    if val == 2:
		        predicted_time = MKL_par[i]
		    if val == 3:
		        predicted_time = cus1[i]
		    if val == 4:
		        predicted_time = cus2_lvl[i]
		    if val == 5:
		        predicted_time = cus2_nolvl[i]
		    if val == 6:
		        predicted_time = syncfree[i]
		    if m[i] < 100000:
		        L_calls_vs_FE_wo_100K.append(FE_wo_ilu[i]*1000/predicted_time)
		        L_calls_vs_FE_w_100K.append(FE_w_ilu[i]*1000/predicted_time)
		        oh_MKLs_wo_100K.append((overhead_data['MKL(seq) Ana'][i]+overhead_data['MKL(seq) 10 iter'][i]))
		        oh_MKLs_w_100K.append((overhead_data['MKL(seq) Ana'][i]+overhead_data['MKL(seq) 10 iter'][i]+\
		                               overhead_data['MKL(seq) ilu'][i]))
		        oh_MKLp_wo_100K.append((overhead_data['MKL(par) Ana'][i]+overhead_data['MKL(par) 10 iter'][i]))
		        oh_MKLp_w_100K.append((overhead_data['MKL(par) Ana'][i]+overhead_data['MKL(par) 10 iter'][i]+\
		                               overhead_data['MKL(par) ilu'][i]))
		        oh_CUS1_wo_100K.append((overhead_data['cuSPARSE(v1) ana'][i]+overhead_data['cuSPARSE(v1)  10 iter'][i]))
		        oh_CUS1_w_100K.append((overhead_data['cuSPARSE(v1) ana'][i]+overhead_data['cuSPARSE(v1)  10 iter'][i]+\
		                               overhead_data['cuSPARSE(v1) ilu'][i]))
		        oh_CUS2lvl_wo_100K.append((overhead_data['cusparse(v2)ana'][i]+overhead_data['cuSPARSE(v2)lvl'][i]))
		        oh_CUS2lvl_w_100K.append((overhead_data['cusparse(v2)ana'][i]+overhead_data['cuSPARSE(v2)lvl'][i]+\
		                                  +overhead_data['cuSPARSE(v2)iluAna'][i]+overhead_data['cuSPARSE(v2)iu'][i]))
		        oh_CUS2nolvl_wo_100K.append((overhead_data['cuSPARSE(v2)nolvl  10 iter'][i]))
		        oh_CUS2nolvl_w_100K.append((overhead_data['cuSPARSE(v2)nolvl  10 iter'][i]))
		        oh_SyncFree_wo_100K.append((overhead_data['Sync-Free ana'][i]+overhead_data['Sync-Free  10 iter'][i]))
		        oh_SyncFree_w_100K.append((overhead_data['SycnFree_LU'][i]+overhead_data['Sync-Free ana'][i]+\
		                                   overhead_data['Sync-Free  10 iter'][i]))
		        oh_FE_wo_100K.append(overhead_data['FE_oh_wo'][i])
		        oh_FE_w_100K.append(overhead_data['FE_oh_w'][i])
		        
		        oh_MKLs_wo_100K_ana.append((overhead_data['MKL(seq) Ana'][i]))
		        oh_MKLs_w_100K_ana.append((overhead_data['MKL(seq) Ana'][i]+overhead_data['MKL(seq) ilu'][i]))
		        oh_MKLp_wo_100K_ana.append((overhead_data['MKL(par) Ana'][i]))
		        oh_MKLp_w_100K_ana.append((overhead_data['MKL(par) Ana'][i]+overhead_data['MKL(par) ilu'][i]))
		        oh_CUS1_wo_100K_ana.append((overhead_data['cuSPARSE(v1) ana'][i]))
		        oh_CUS1_w_100K_ana.append((overhead_data['cuSPARSE(v1) ana'][i]+overhead_data['cuSPARSE(v1) ilu'][i]))
		        oh_CUS2lvl_wo_100K_ana.append((overhead_data['cusparse(v2)ana'][i]))
		        oh_CUS2lvl_w_100K_ana.append((overhead_data['cusparse(v2)ana'][i]+\
		                                      overhead_data['cuSPARSE(v2)iluAna'][i]+overhead_data['cuSPARSE(v2)iu'][i]))
		        oh_CUS2nolvl_wo_100K_ana.append(0)
		        oh_CUS2nolvl_w_100K_ana.append(0)
		        oh_SyncFree_wo_100K_ana.append((overhead_data['Sync-Free ana'][i]))
		        oh_SyncFree_w_100K_ana.append((overhead_data['SycnFree_LU'][i]+overhead_data['Sync-Free ana'][i]))
		        
		    if m[i] >= 100000 and m[i] < 1000000:
		        L_calls_vs_FE_wo_1000K.append(FE_wo_ilu[i]*1000/predicted_time)
		        L_calls_vs_FE_w_1000K.append(FE_w_ilu[i]*1000/predicted_time)
		        
		        oh_MKLs_wo_1000K.append((overhead_data['MKL(seq) Ana'][i]+overhead_data['MKL(seq) 10 iter'][i]))
		        oh_MKLs_w_1000K.append((overhead_data['MKL(seq) Ana'][i]+overhead_data['MKL(seq) 10 iter'][i]+\
		                                overhead_data['MKL(seq) ilu'][i]))
		        oh_MKLp_wo_1000K.append((overhead_data['MKL(par) Ana'][i]+overhead_data['MKL(par) 10 iter'][i]))
		        oh_MKLp_w_1000K.append((overhead_data['MKL(par) Ana'][i]+overhead_data['MKL(par) 10 iter'][i]+\
		                                overhead_data['MKL(par) ilu'][i]))
		        oh_CUS1_wo_1000K.append((overhead_data['cuSPARSE(v1) ana'][i]+\
		                                 overhead_data['cuSPARSE(v1)  10 iter'][i]))
		        oh_CUS1_w_1000K.append((overhead_data['cuSPARSE(v1) ana'][i]+\
		                                overhead_data['cuSPARSE(v1)  10 iter'][i]+overhead_data['cuSPARSE(v1) ilu'][i]))
		        oh_CUS2lvl_wo_1000K.append((overhead_data['cusparse(v2)ana'][i]+overhead_data['cuSPARSE(v2)lvl'][i]))
		        oh_CUS2lvl_w_1000K.append((overhead_data['cusparse(v2)ana'][i]+\
		                                   overhead_data['cuSPARSE(v2)lvl'][i]+\
		                                   overhead_data['cuSPARSE(v2)iluAna'][i]+overhead_data['cuSPARSE(v2)iu'][i]))
		        oh_CUS2nolvl_wo_1000K.append((overhead_data['cuSPARSE(v2)nolvl  10 iter'][i]))
		        oh_CUS2nolvl_w_1000K.append((overhead_data['cuSPARSE(v2)nolvl  10 iter'][i]))
		        oh_SyncFree_wo_1000K.append((overhead_data['Sync-Free ana'][i]+overhead_data['Sync-Free  10 iter'][i]))
		        oh_SyncFree_w_1000K.append((overhead_data['SycnFree_LU'][i]+\
		                                    overhead_data['Sync-Free ana'][i]+overhead_data['Sync-Free  10 iter'][i]))
		        oh_FE_wo_1000K.append((overhead_data['FE_oh_wo'][i]))
		        oh_FE_w_1000K.append((overhead_data['FE_oh_w'][i]))
		        
		        oh_MKLs_wo_1000K_ana.append((overhead_data['MKL(seq) Ana'][i]))
		        oh_MKLs_w_1000K_ana.append((overhead_data['MKL(seq) Ana'][i]+overhead_data['MKL(seq) ilu'][i]))
		        oh_MKLp_wo_1000K_ana.append((overhead_data['MKL(par) Ana'][i]))
		        oh_MKLp_w_1000K_ana.append((overhead_data['MKL(par) Ana'][i]+overhead_data['MKL(par) ilu'][i]))
		        oh_CUS1_wo_1000K_ana.append((overhead_data['cuSPARSE(v1) ana'][i]))
		        oh_CUS1_w_1000K_ana.append((overhead_data['cuSPARSE(v1) ana'][i]+overhead_data['cuSPARSE(v1) ilu'][i]))
		        oh_CUS2lvl_wo_1000K_ana.append((overhead_data['cusparse(v2)ana'][i]))
		        oh_CUS2lvl_w_1000K_ana.append((overhead_data['cusparse(v2)ana'][i]+\
		                                       overhead_data['cuSPARSE(v2)iluAna'][i]+\
		                                       overhead_data['cuSPARSE(v2)iu'][i]))
		        oh_CUS2nolvl_wo_1000K_ana.append(0)
		        oh_CUS2nolvl_w_1000K_ana.append(0)
		        oh_SyncFree_wo_1000K_ana.append((overhead_data['Sync-Free ana'][i]))
		        oh_SyncFree_w_1000K_ana.append((overhead_data['SycnFree_LU'][i]+overhead_data['Sync-Free ana'][i]))
		  
		        #emp_oh_wo_1000K.append(oh_MKLs_wo_1000K[i]+oh_MKLp_wo_1000K[i]+oh_CUS1_wo_1000K[i]+oh_CUS2lvl_wo_1000K[i]+oh_CUS2nolvl_wo_1000K[i]+oh_SyncFree_wo_1000K[i])
		    if m[i] >= 1000000:
		        L_calls_vs_FE_wo_g1000K.append(FE_wo_ilu[i]*1000/predicted_time)
		        L_calls_vs_FE_w_g1000K.append(FE_w_ilu[i]*1000/predicted_time)
		        oh_MKLs_wo_g1000K.append((overhead_data['MKL(seq) Ana'][i]))
		        oh_MKLs_w_g1000K.append((overhead_data['MKL(seq) Ana'][i]+overhead_data['MKL(seq) ilu'][i]))
		        oh_MKLp_wo_g1000K.append((overhead_data['MKL(par) Ana'][i]))
		        oh_MKLp_w_g1000K.append((overhead_data['MKL(par) Ana'][i]+overhead_data['MKL(par) ilu'][i]))
		        oh_CUS1_wo_g1000K.append((overhead_data['cuSPARSE(v1) ana'][i]+overhead_data['cuSPARSE(v1)  10 iter'][i]))
		        oh_CUS1_w_g1000K.append((overhead_data['cuSPARSE(v1) ana'][i]+overhead_data['cuSPARSE(v1) ilu'][i]+overhead_data['cuSPARSE(v1)  10 iter'][i]))
		        oh_CUS2lvl_wo_g1000K.append((overhead_data['cusparse(v2)ana'][i]+overhead_data['cuSPARSE(v2)lvl'][i]))
		        oh_CUS2lvl_w_g1000K.append((overhead_data['cusparse(v2)ana'][i]+overhead_data['cuSPARSE(v1) ilu'][i]+\
		                                    overhead_data['cuSPARSE(v2)iluAna'][i]+overhead_data['cuSPARSE(v2)iu'][i]))
		        oh_CUS2nolvl_wo_g1000K.append((0))
		        oh_CUS2nolvl_w_g1000K.append((0))
		        oh_SyncFree_wo_g1000K.append((overhead_data['Sync-Free ana'][i]))
		        oh_SyncFree_w_g1000K.append((overhead_data['SycnFree_LU'][i]+overhead_data['Sync-Free ana'][i]))
		        oh_FE_wo_g1000K.append(overhead_data['FE_oh_wo'][i])
		        oh_FE_w_g1000K.append(overhead_data['FE_oh_w'][i])
		        
		        oh_MKLs_wo_g1000K_ana.append((overhead_data['MKL(seq) Ana'][i]))
		        oh_MKLs_w_g1000K_ana.append((overhead_data['MKL(seq) Ana'][i]+overhead_data['MKL(seq) ilu'][i]))
		        oh_MKLp_wo_g1000K_ana.append((overhead_data['MKL(par) Ana'][i]))
		        oh_MKLp_w_g1000K_ana.append((overhead_data['MKL(par) Ana'][i]+overhead_data['MKL(par) ilu'][i]))
		        oh_CUS1_wo_g1000K_ana.append((overhead_data['cuSPARSE(v1) ana'][i]))
		        oh_CUS1_w_g1000K_ana.append((overhead_data['cuSPARSE(v1) ana'][i]+overhead_data['cuSPARSE(v1) ilu'][i]))
		        oh_CUS2lvl_wo_g1000K_ana.append((overhead_data['cusparse(v2)ana'][i]))
		        oh_CUS2lvl_w_g1000K_ana.append((overhead_data['cusparse(v2)ana'][i]+overhead_data['cuSPARSE(v2)lvl'][i]+\
		                                        overhead_data['cuSPARSE(v1) ilu'][i]+overhead_data['cuSPARSE(v2)iluAna'][i]+\
		                                        overhead_data['cuSPARSE(v2)iu'][i]))
		        oh_CUS2nolvl_wo_g1000K_ana.append(0)
		        oh_CUS2nolvl_w_g1000K_ana.append(0)
		        oh_SyncFree_wo_g1000K_ana.append((overhead_data['Sync-Free ana'][i]))
		        oh_SyncFree_w_g1000K_ana.append((overhead_data['SycnFree_LU'][i]+overhead_data['Sync-Free ana'][i]))
		  
		        #emp_oh_wo_g1000K.append(oh_MKLs_wo_g1000K[i] + oh_MKLp_wo_g1000K[i] + oh_CUS1_wo_g1000K[i] + oh_CUS2lvl_wo_g1000K[i] + oh_CUS2nolvl_wo_g1000K[i] + oh_SyncFree_wo_g1000K[i])

		    i = i + 1
		emp_oh_wo_100K = (np.sum(oh_MKLs_wo_100K)+np.sum(oh_MKLp_wo_100K)+np.sum(oh_CUS1_wo_100K) + \
		np.sum(oh_CUS2lvl_wo_100K) + np.sum(oh_CUS2nolvl_wo_100K) + np.sum(oh_SyncFree_wo_100K))\
		/(len(oh_MKLs_wo_100K)*1000)

		emp_oh_wo_1000K = (np.sum(oh_MKLs_wo_1000K)+np.sum(oh_MKLp_wo_1000K)+np.sum(oh_CUS1_wo_1000K) + \
		np.sum(oh_CUS2lvl_wo_1000K) + np.sum(oh_CUS2nolvl_wo_1000K) + np.sum(oh_SyncFree_wo_1000K))\
		/(len(oh_MKLs_wo_1000K)*1000)

		emp_oh_wo_g1000K = (np.sum(oh_MKLs_wo_g1000K)+np.sum(oh_MKLp_wo_g1000K)+np.sum(oh_CUS1_wo_g1000K) + \
		np.sum(oh_CUS2lvl_wo_g1000K) + np.sum(oh_CUS2nolvl_wo_g1000K) + np.sum(oh_SyncFree_wo_g1000K))\
		/(len(oh_MKLs_wo_g1000K)*1000)

		emp_oh_w_100K = (np.sum(oh_MKLs_w_100K)+np.sum(oh_MKLp_w_100K)+np.sum(oh_CUS1_w_100K) + \
		np.sum(oh_CUS2lvl_w_100K) + np.sum(oh_CUS2nolvl_w_100K) + np.sum(oh_SyncFree_w_100K))/(len(oh_MKLs_w_100K)*1000)

		emp_oh_w_1000K = (np.sum(oh_MKLs_w_1000K)+np.sum(oh_MKLp_w_1000K)+np.sum(oh_CUS1_w_1000K) + \
		np.sum(oh_CUS2lvl_w_1000K) + np.sum(oh_CUS2nolvl_w_1000K) + np.sum(oh_SyncFree_w_1000K))\
		/(len(oh_MKLs_w_1000K)*1000)

		emp_oh_w_g1000K = (np.sum(oh_MKLs_w_g1000K)+np.sum(oh_MKLp_w_g1000K)+np.sum(oh_CUS1_w_g1000K) + \
		np.sum(oh_CUS2lvl_w_g1000K) + np.sum(oh_CUS2nolvl_w_g1000K) + np.sum(oh_SyncFree_w_g1000K))\
		/(len(oh_MKLs_w_g1000K)*1000)

		emp_oh_wo_g1000K_ana = (np.sum(oh_MKLs_wo_g1000K_ana)+np.sum(oh_MKLp_wo_g1000K_ana)+np.sum(oh_CUS1_wo_g1000K_ana) + \
		np.sum(oh_CUS2lvl_wo_g1000K_ana) + np.sum(oh_CUS2nolvl_wo_g1000K_ana) + np.sum(oh_SyncFree_wo_g1000K_ana))\
		/(len(oh_MKLs_wo_g1000K_ana)*1000)

		emp_oh_w_g1000K_ana = (np.sum(oh_MKLs_w_g1000K_ana)+np.sum(oh_MKLp_w_g1000K_ana)+np.sum(oh_CUS1_w_g1000K_ana) + \
		np.sum(oh_CUS2lvl_w_g1000K_ana) + np.sum(oh_CUS2nolvl_w_g1000K_ana) + np.sum(oh_SyncFree_w_g1000K_ana))\
		/(len(oh_MKLs_w_g1000K_ana)*1000)		    
		
		Overhead_wo_100K_bar = (np.sum(oh_FE_wo_100K)/len(oh_FE_wo_100K), emp_oh_wo_100K, \
                   np.sum(oh_MKLs_wo_100K_ana)/(len(oh_MKLs_wo_100K_ana)*1000),\
                   np.sum(oh_MKLp_wo_100K_ana)/(len(oh_MKLp_wo_100K_ana)*1000),\
                   np.sum(oh_CUS1_wo_100K_ana)/(len(oh_MKLs_wo_100K_ana)*1000),\
                   np.sum(oh_CUS2lvl_wo_100K_ana)/(len(oh_CUS2lvl_wo_100K_ana)*1000),\
                   np.sum(oh_CUS2lvl_wo_100K_ana)/(len(oh_CUS2lvl_wo_100K_ana)*1000),\
                   np.sum(oh_SyncFree_wo_100K_ana)/(len(oh_SyncFree_wo_100K_ana)*1000))

		Overhead_w_100K_bar = (np.sum(oh_FE_w_100K)/len(oh_FE_w_100K), emp_oh_w_100K, \
                   np.sum(oh_MKLs_w_100K_ana)/(len(oh_MKLs_w_100K_ana)*1000),\
                   np.sum(oh_MKLp_w_100K_ana)/(len(oh_MKLp_w_100K_ana)*1000),\
                   np.sum(oh_CUS1_w_100K_ana)/(len(oh_CUS1_w_100K_ana)*1000),\
                   np.sum(oh_CUS2lvl_w_100K_ana)/(len(oh_CUS2lvl_w_100K_ana)*1000),\
                   np.sum(oh_CUS2lvl_w_100K_ana)/(len(oh_CUS2lvl_w_100K_ana)*1000),\
                   np.sum(oh_SyncFree_w_100K_ana)/(len(oh_SyncFree_w_100K_ana)*1000))

		Overhead_wo_1000K_bar = (np.sum(oh_FE_wo_1000K)/len(oh_FE_wo_1000K), emp_oh_wo_1000K, \
                   np.sum(oh_MKLs_wo_1000K_ana)/(len(oh_MKLs_wo_1000K_ana)*1000),\
                   np.sum(oh_MKLp_wo_1000K_ana)/(len(oh_MKLp_wo_1000K_ana)*1000),\
                   np.sum(oh_CUS1_wo_1000K_ana)/(len(oh_MKLs_wo_1000K_ana)*1000),\
                   np.sum(oh_CUS2lvl_wo_1000K_ana)/(len(oh_CUS2lvl_wo_1000K_ana)*1000),\
                   np.sum(oh_CUS2lvl_wo_1000K_ana)/(len(oh_CUS2lvl_wo_1000K_ana)*1000),\
                   np.sum(oh_SyncFree_wo_1000K_ana)/(len(oh_SyncFree_wo_1000K_ana)*1000))

		Overhead_w_1000K_bar = (np.sum(oh_FE_w_1000K)/len(oh_FE_w_1000K), emp_oh_w_1000K, \
                   np.sum(oh_MKLs_w_1000K_ana)/(len(oh_MKLs_w_1000K_ana)*1000),\
                   np.sum(oh_MKLp_w_1000K_ana)/(len(oh_MKLp_w_1000K_ana)*1000),\
                   np.sum(oh_CUS1_w_1000K_ana)/(len(oh_CUS1_w_1000K_ana)*1000),\
                   np.sum(oh_CUS2lvl_w_1000K_ana)/(len(oh_CUS2lvl_w_1000K_ana)*1000),\
                   np.sum(oh_CUS2lvl_w_1000K_ana)/(len(oh_CUS2lvl_w_1000K_ana)*1000),\
                   np.sum(oh_SyncFree_w_1000K_ana)/(len(oh_SyncFree_w_1000K_ana)*1000))

		Overhead_wo_g1000K_bar = (np.sum(oh_FE_wo_g1000K)/len(oh_FE_wo_g1000K), emp_oh_wo_g1000K, \
                   np.sum(oh_MKLs_wo_g1000K_ana)/(len(oh_MKLs_wo_g1000K_ana)*1000),\
                   np.sum(oh_MKLp_wo_g1000K_ana)/(len(oh_MKLp_wo_g1000K_ana)*1000),\
                   np.sum(oh_CUS1_wo_g1000K_ana)/(len(oh_MKLs_wo_g1000K_ana)*1000),\
                   np.sum(oh_CUS2lvl_wo_g1000K_ana)/(len(oh_CUS2lvl_wo_g1000K_ana)*1000),\
                   np.sum(oh_CUS2lvl_wo_g1000K_ana)/(len(oh_CUS2lvl_wo_g1000K_ana)*1000),\
                   np.sum(oh_SyncFree_wo_g1000K_ana)/(len(oh_SyncFree_wo_g1000K_ana)*1000))

		Overhead_w_g1000K_bar = (np.sum(oh_FE_w_g1000K)/len(oh_FE_w_g1000K), emp_oh_w_g1000K, \
                   np.sum(oh_MKLs_w_g1000K_ana)/(len(oh_MKLs_w_g1000K_ana)*1000),\
                   np.sum(oh_MKLp_w_g1000K_ana)/(len(oh_MKLp_w_g1000K_ana)*1000),\
                   np.sum(oh_CUS1_w_g1000K_ana)/(len(oh_CUS1_w_g1000K_ana)*1000),\
                   np.sum(oh_CUS2lvl_w_g1000K_ana)/(len(oh_CUS2lvl_w_g1000K_ana)*1000),\
                   np.sum(oh_CUS2lvl_w_g1000K_ana)/(len(oh_CUS2lvl_w_g1000K_ana)*1000),\
                   np.sum(oh_SyncFree_w_g1000K_ana)/(len(oh_SyncFree_w_g1000K_ana)*1000))

		print('Number of lower triangular solve iterations (LTI) to amortize feature extraction overhead (FEO) without ILU')
		print('%40s =%20d' % ('1K-100K Min LTI to amortize FEO',np.ceil(np.min(L_calls_vs_FE_wo_100K))))
		print('%40s =%20d' % ('1K-100K Mean LTI to amortize FEO',np.ceil(np.mean(L_calls_vs_FE_wo_100K))))
		print('%40s =%20d' % ('1K-100K Max LTI to amortize FEO',np.ceil(np.max(L_calls_vs_FE_wo_100K))))
		print('%40s =%20d' % ('100K-1000K Min LTI to amortize FEO',np.ceil(np.min(L_calls_vs_FE_wo_1000K))))
		print('%40s =%20d' % ('100K-1000K Mean LTI to amortize FEO',np.ceil(np.mean(L_calls_vs_FE_wo_1000K))))
		print('%40s =%20d' % ('100K-1000K Max LTI to amortize FEO',np.ceil(np.max(L_calls_vs_FE_wo_1000K))))
		print('%40s =%20d' % ('> 1000K Min LTI to amortize FEO',np.ceil(np.min(L_calls_vs_FE_wo_g1000K))))
		print('%40s =%20d' % ('> 1000K Mean LTI to amortize FEO',np.ceil(np.mean(L_calls_vs_FE_wo_g1000K))))
		print('%40s =%20d' % ('> 1000K Max LTI to amortize FEO',np.ceil(np.max(L_calls_vs_FE_wo_g1000K))))
		print('')
		#print('Number of lower triangular solve iterations (LTI) to amortize feature extraction overhead (FEO) with ILU')
		#print('1K-100K Min LTI to amortize FEO=%20d' % np.ceil(np.min(L_calls_vs_FE_w_100K)))
		#print('1K-100K Mean LTI to amortize FEO=%20d' % np.ceil(np.mean(L_calls_vs_FE_w_100K)))
		#print('1K-100K Max LTI to amortize FEO=%20d' % np.ceil(np.max(L_calls_vs_FE_w_100K)))
		#print('100K-1000K Min LTI to amortize FEO=%20d' % np.ceil(np.min(L_calls_vs_FE_w_1000K)))
		#print('100K-1000K Mean LTI to amortize FEO=%20d' % np.ceil(np.mean(L_calls_vs_FE_w_1000K)))
		#print('100K-1000K Max LTI to amortize FEO=%20d' % np.ceil(np.max(L_calls_vs_FE_w_1000K)))
		##print('> 1000K Min LTI to amortize FEO=%20d' % np.ceil(np.min(L_calls_vs_FE_w_g1000K)))
		#print('> 1000K Mean LTI to amortize FEO=%20d' % np.ceil(np.mean(L_calls_vs_FE_w_g1000K)))
		#print('> 1000K Max LTI to amortize FEO=%20d' % np.ceil(np.max(L_calls_vs_FE_w_g1000K)))
		
		f, ax = plt.subplots(2, 3,figsize=(15, 6))
		N = 8
		width = 0.55
		x = ('Framework','Agressive user','MKL(seq)','MKL(par)','cuSPARSE(v1)',\
     		'cuSPARSE(v2)\n(level-sch.)','cuSPARSE(v2)\n(no level-sch.)','Sync-Free')
		ind = np.arange(N)
		x1 = ('','','','','','','','')
		p11 = ax[0,0].bar(ind, Overhead_wo_100K_bar, width,color='maroon')
		p12 = ax[0,1].bar(ind, Overhead_wo_1000K_bar, width,color='maroon')
		p13 = ax[0,2].bar(ind, Overhead_wo_g1000K_bar, width,color='maroon')
		p14 = ax[1,0].bar(ind, Overhead_w_100K_bar, width,color='maroon')
		p15 = ax[1,1].bar(ind, Overhead_w_1000K_bar, width,color='maroon')
		p16 = ax[1,2].bar(ind, Overhead_w_g1000K_bar, width,color='maroon')
		p11[0].set_color('b')
		p12[0].set_color('b')
		p13[0].set_color('b')
		p14[0].set_color('b')
		p15[0].set_color('b')
		p16[0].set_color('b')
		label_font = 12
		ax[0,0].set_ylabel('Execution time (sec)',fontsize=12)
		ax[0,0].set_yscale('log')
		ax[0,0].set_xticks(np.arange(len(x)))
		ax[0,0].set_xticklabels(x1,rotation=90,fontsize=label_font)
		ax[0,0].set_title('Overhead (w/o ILU) 1K-100K',loc="left")
		ax[0,0].set_xlabel('(a)')
		ax[0,1].set_yscale('log')
		ax[0,1].set_xticks(np.arange(len(x)))
		ax[0,1].set_xticklabels(x1,rotation=90,fontsize=label_font)
		ax[0,1].set_title('Overhead (w/o ILU) 100K-1000K',loc="left")
		ax[0,1].set_xlabel('(b)')
		ax[0,2].set_yscale('log')
		ax[0,2].set_xticks(np.arange(len(x)))
		ax[0,2].set_xticklabels(x1,rotation=90,fontsize=label_font)
		ax[0,2].set_title('Overhead (w/o ILU) >1000K',loc="left")
		ax[0,2].set_xlabel('(c)')
		ax[1,0].set_ylabel('Execution time (sec)',fontsize=12)
		ax[1,0].set_yscale('log')
		ax[1,0].set_xticks(np.arange(len(x)))
		ax[1,0].set_xticklabels(x,rotation=90,fontsize=label_font)
		ax[1,0].set_title('Overhead (w ILU) 1K-100K',loc="left")
		ax[1,0].set_xlabel('(d)')
		ax[1,1].set_yscale('log')
		ax[1,1].set_xticks(np.arange(len(x)))
		ax[1,1].set_xticklabels(x,rotation=90,fontsize=label_font)
		ax[1,1].set_title('Overhead (w ILU) 100K-1000K',loc="left")
		ax[1,1].set_xlabel('(e)')
		ax[1,2].set_yscale('log')
		ax[1,2].set_xticks(np.arange(len(x)))
		ax[1,2].set_xticklabels(x,rotation=90,fontsize=label_font)
		ax[1,2].set_title('Overhead (w ILU) >1000K',loc="left")
		ax[1,2].set_xlabel('(f)')
		plt.tight_layout()
		f.savefig('./datasets/figure10.pdf',bbox_inches='tight')
		print("Figure 10 saved in datasets as figure10.eps")
		print("Note: Mean LTI to amortize FEO statistic for matrices with > 1000K row can slightly vary from line 3 page 13 and from run-to-run")



###############################################################
### main code of the program
###############################################################
if __name__ == "__main__":
	print("SpTRSV framework artifact evaluation Script")

	if len(sys.argv) > 1:
		option = sys.argv[1]

		if option == "figure2":
			figure1 = CPUGPUComparison()
			print("Generating Figure 2. SpTRSV performance on Intel Xeon Gold (6148) CPU and an NVIDIA V100 GPU (32GB, PCIe)")
			figure1.DrawFigure('./datasets/CPU_GPU_best_SpTRSV_37_matrices.csv')

		if option == "figure7":
			figure7 = Prediction()
			print("Generating Figure 7. Model cross validation scores with 30 features in the feature set")
			figure7.CrossValidation('./datasets/Training_data.csv',2)

		if option == "figure8":
			figure6 = Prediction()
			print("Generating Figure 8. Model cross validation scores with 10 features in the feature set")
			figure6.CrossValidation('./datasets/Training_data.csv',1)

		if option == "figure9":
			figure7 = Performance()
			print("Generating Figure 9. Speedup gained by predicted over lazy choice algorithm. >= 1 indicates speedup of greater or equal to 1. Mean refers to average speedup (harmonic mean) achieved by the framework over the lazy choice.")
			figure7.Speedup('./datasets/Training_data.csv')

		if option == "figure10":
			figure8 = Performance()
			print("Generating Figure 10. Mean overhead of framework versus mean empirical execution time for aggressive and lazy users. 1K-100K, 100K-1000K and >1000K refer to matrix size ranges.")
			figure8.Overheads('./datasets/Training_data.csv','./datasets/Overhead.csv')

		if option == "table1":
			table1 = CPUGPUComparison()
			print("\nTable 1. SpTRSV winning algorithm breakdown for 37 matrices in Figure 2\n")
			table1.DrawComparisonTable('./datasets/CPU_GPU_SpTRSV_perf_data_37_matrices.csv')

		if option == "table2":
			featurescores = FeatureSelection()
			print("\nTable 2. Selected feature set for the prediction framework\n")
			featurescores.FeatureRanking('./datasets/Features.csv')

		if option == "table3":
			table3 = CPUGPUComparison()
			print("\nTable 3. SpTRSV winning algorithm breakdown for the 998 matrices from SuiteSparse\n")
			table3.DrawComparisonTable('./datasets/CPU_GPU_SpTRSV_comparison_full_dataset.csv')

		if option == "table4":
			table4 = CPUGPUComparison()
			print("\nTable 4. Number of rows and nonzero statistics for the 998 matrices from SuiteSparse\n")
			table4.DrawStatsTable('./datasets/CPU_GPU_SpTRSV_comparison_full_dataset.csv')

		if option == "printallfeatures":
			feature_sel = FeatureSelection()
			feature_sel.PrintAllFeatures('./datasets/Features.csv')

		
			
			
