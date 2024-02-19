# https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890) 
  
# data (as pandas dataframes) 
X = aids_clinical_trials_group_study_175.data.features 
y = aids_clinical_trials_group_study_175.data.targets 
  
# # metadata 
# print(aids_clinical_trials_group_study_175.metadata) 
  
# # variable information 
# print(aids_clinical_trials_group_study_175.variables) 
"""
       name     role  ... units missing_values
0    pidnum       ID  ...  None             no
1       cid   Target  ...  None             no
2      time  Feature  ...  None             no
3       trt  Feature  ...  None             no
4       age  Feature  ...  None             no
5      wtkg  Feature  ...  None             no
6      hemo  Feature  ...  None             no
7      homo  Feature  ...  None             no
8     drugs  Feature  ...  None             no
9    karnof  Feature  ...  None             no
10   oprior  Feature  ...  None             no
11      z30  Feature  ...  None             no
12   zprior  Feature  ...  None             no
13  preanti  Feature  ...  None             no
14     race  Feature  ...  None             no
15   gender  Feature  ...  None             no
16     str2  Feature  ...  None             no
17    strat  Feature  ...  None             no
18  symptom  Feature  ...  None             no
19    treat  Feature  ...  None             no
20   offtrt  Feature  ...  None             no
21     cd40  Feature  ...  None             no
22    cd420  Feature  ...  None             no
23     cd80  Feature  ...  None             no
24    cd820  Feature  ...  None             no

"""

# for col in X.columns:
#     print(col, X[col].dtype) # either int64 or float64


