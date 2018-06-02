#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:37:43 2018

@author: lunar
"""
import os
os.chdir('/home/lunar/Desktop/research_project/results/Pipeline')
from packages import *

class eliminative_selection():
    def __init__(self, X_d, y_d, seed_subsets = {}, dataname='3-class', n=4, c=19):
        self.X = X_d
        self.y = y_d
        self.n = n
        self.dataname = dataname
        
        
        self.seed_subsets = seed_subsets
        self.sort_modes = ['Mean', 'imp']
        self.models = ['xgb', 'forest']
        
        self.final_results = {}
        for g in self.seed_subsets.keys():
            self.final_results[g] = {}
            for i, m in enumerate(self.models):
                self.final_results[g][m]={}
                for s in self.sort_modes:
                    self.final_results[g][m][s]=None
        print("Initialized new class, data-type:{}".format(self.dataname))
    
    
#-------------------------------------------------------------------------------------------------------------    
    def model1(self, ):
        f = lambda x: x>12 and 3 or int(6-0.3*x)
        if len(set(list(self.y))) > 2:
            objective = 'multi:softmax'
        else:
            objective = 'binary:logitraw'
        xgb = XGBClassifier(n_estimators=256, objective=objective, learning_rate=1, min_child_weight=1.2,
                               random_state=42, nthread=4, max_depth=f(self.n), silent=1, subsample=1, 
                      reg_lambda=1.3, scale_pos_weight=0.3,)
        return xgb
        
    def model2(self):
        forest = RandomForestClassifier(n_estimators=256,  min_samples_split=2,  
                                         max_depth=int(self.X.shape[0]/8), n_jobs=4, class_weight='balanced')
        return forest
    
    
#-------------------------------------------------------------------------------------------------------------        
    def single_run(self, subset_name, sort_method='imp', model_type='xgb'):
        
        subset = self.seed_subsets[subset_name]
        
        if model_type=='xgb':
            model=self.model1()
        elif model_type=='forest':
            model=self.model2()
            
        loo = LeaveOneOut()
        loo.get_n_splits(self.X[subset])
        
        outputs = {'cumm_predicts': np.zeros(self.X.shape[0]), 'cumm_residues': np.zeros(self.X.shape[0]),
                         'loo_acc': None, 'gene_subset': None, 'feat_imps':None, 'final_stats':None}
        
        feat_imp_  = []
        i = 0
        
        single_run_start = time.time()
        #print(subset_name, sort_method)
        
        
        for train_index, test_index in loo.split(self.X[subset]):
            X_train, X_test = self.X[subset].values[train_index], self.X[subset].values[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            model.fit(X_train, y_train)
            outputs['cumm_predicts'][i] = model.predict(X_test)
            outputs['cumm_residues'][i] = y_test
            feat_imp_.append(model.feature_importances_)
            i += 1
            
        s = outputs['cumm_predicts']-outputs['cumm_residues']
        outputs['loo_acc'] = 1-np.count_nonzero(s)/len(s)

        feat_imps = pd.DataFrame({key: value for (key, value) in enumerate(feat_imp_)}, index=self.X[subset].columns)
        feat_imps['Mean'] = feat_imps.std(axis=1)
        feat_imps['std'] = feat_imps.mean(axis=1)
        feat_imps['imp'] = feat_imps.ix[:,:-2].mean(axis=1)/(feat_imps.ix[:,:-2].std(axis=1)+2/3*feat_imps['std'].mean(axis=0))
        
        feat_imps = feat_imps.sort_values(by='{}'.format(sort_method), ascending=True)
        outputs['feat_imps']=feat_imps.ix[:,-3:] # we dont include all importancies, only statistics over them
        outputs['gene_subset'] = feat_imps.index[int(len(feat_imps.index)/(self.n+1/3))+3:]
        t = single_run_start - time.time()
        #print('{}:{}:{}; Run with n={} completed in {}s'.format(subset_name, model_type, sort_method, self.n, t)) 
        return outputs
    
#-------------------------------------------------------------------------------------------------------------    
    def iterative_run(self, subset_name, c=3, sort_method='imp', model_type='xgb'):
        
        results = {'loo_accs': [], 'gene_subsets': [], 'predicts': [], 'actuals': [], 'stats' : []}
        self.final_results[subset_name][model_type][sort_method]=results
        
        initial_n = self.n
        initial_gene_subset = self.seed_subsets[subset_name]
        
        for i in range(c):
            outputs = self.single_run(subset_name, sort_method=sort_method, model_type=model_type)
            self.final_results[subset_name][model_type][sort_method]['loo_accs'].append(outputs['loo_acc'])
            self.final_results[subset_name][model_type][sort_method]['gene_subsets'].append(outputs['gene_subset'])
            self.final_results[subset_name][model_type][sort_method]['predicts'].append(outputs['cumm_predicts'])
            self.final_results[subset_name][model_type][sort_method]['actuals'].append(outputs['cumm_residues'])
            self.final_results[subset_name][model_type][sort_method]['stats'].append(outputs['feat_imps'])
            self.seed_subsets[subset_name] = outputs['gene_subset']
            self.n += +1 
        print('n={}, model is {}, sort is {}, genes:'.format(self.n, model_type, sort_method), outputs['gene_subset'])
        self.n = initial_n
        self.seed_subsets[subset_name] = initial_gene_subset
        
        return 0

    
#-------------------------------------------------------------------------------------------------------------
    def pipeline(self, cs=11):
        # this function should return results for both sorting methods, for both models. Overall 2x2=4 iterative runs                
        for g in self.seed_subsets.keys():
            for i, m in enumerate(self.models):
                for s in self.sort_modes:  
                    self.iterative_run(c=cs, subset_name=g, sort_method=s, model_type=m)
       
        with open('../../results/Result_outputs/{}{}_1.csv'.format(str(self.dataname), 'run1'), 'wb') as out:
            pickle.dump(self.final_results, out, protocol=pickle.HIGHEST_PROTOCOL)      
    
#--------------------------------------------------------------------------------------------------------------

    def utils_plot_accuracy_curve(self, u=2, u2=20):
        z = 0
        fig,ax = plt.subplots(u,2,figsize=(20,12))
        #fig.tight_layout()
        for i, g in enumerate(self.seed_subsets.keys()):
                    for j, m in enumerate(self.models):
                        for k, s in enumerate(self.sort_modes):
                            plt.subplot(u,2,z+1)
                            plot_accuracy_curve(self.final_results[g][m][s], base=g, 
                                                model_type=m, sort_method=s)
                            z += 1

    def utils_best_subsets(self,):
        
        def from_results_best_subset(res):
            i = np.argmax(res['loo_accs'])
            return res['gene_subsets'][i]
        
        for i, g in enumerate(self.seed_subsets.keys()):
                    for j, m in enumerate(self.models):
                        for k, s in enumerate(self.sort_modes):
                            o = from_results_best_subset(self.final_results[g][m][s])
                            print(':'.join([g,m,s]),o)
                            self.final_results[g][m][s]['best_subset'] = from_results_best_subset(self.final_results[g][m][s])

    def utils_intersect_subsets(class_data,):
        permutation_list = []
        intersections = {}
        for g in class_data.seed_subsets.keys():
            for m in class_data.models:
                for s in class_data.sort_modes:
                    permutation_list.append([':'.join([g,m,s]), class_data.final_results[g][m][s]['best_subset']])
        for i, j in itertools.permutations(permutation_list,2):
            intersections[i[0]+'|'+j[0]] = np.intersect1d(i[1][0],j[1][0])
        return intersections








