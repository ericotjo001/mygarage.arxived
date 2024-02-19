"""
This project uses real data for demonstration.
We will structure the code linearly so that the flow of the process is easy to see.
 (c.f. projects/ezmed where we structure the project properly with config-controller-pipeline.)

https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175
"""

import os, json

def main():
    print('Welcome!')

    from src.utils import manage_dirs_p
    DIRS = manage_dirs_p(**{
        "projectname" : "aidsct",
        "label": "numproto-000",
        })

    from ucimlrepo import fetch_ucirepo 
    aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890) 

    df_X = aids_clinical_trials_group_study_175.data.features 
    # y = aids_clinical_trials_group_study_175.data.targets # binary: the patient died/not.
    # In this project, we don't need the original target variable.
    # Instead, we test if "drugs" (binary, history of IV drug use) 
    # affects "karnof" (Karnofsky score on a scale of 0-100)
    print('n data:', len(df_X))

    DV = "karnof"
    IV = "drugs" # history of IV drug use (0=no, 1=yes)
    EXCLUDE = [DV, IV, "pidnum", "cid"]
    NUMERICAL_FEATURES = [a for a in list(df_X.columns) if a not in EXCLUDE] 
    config = {'DV': DV, 'IV': IV, 'DV_description': "Karnofsky score",
        'idx2IV': [0,1], # or like ["placebo", "drugX"]
        'NUMERICAL_FEATURES': NUMERICAL_FEATURES, 'TOKEN_FEATURES': [],}

    # compute means for one way ANOVA
    from src.data.utils import compute_means    
    means = compute_means(df_X, DV, IV, config['idx2IV'])
    with open(DIRS['MEANS_DIR'],'w') as f:
        json.dump(means, f)

    from src.data.utils import package_df
    data_package = package_df(df_X, config)

    from src.prototype import NumPrototype
    POLES = [-1.0,1.0]
    POLES2FACTOR = {-1.0: 0, 1.0: 1} # "drugs" variable
    model = NumPrototype(POLES, POLES2FACTOR, n_gen=len(df_X)*5)

    model.setup_data(data_package)

    # here, we first choose which of config['idx2IV'] will be the treatment group  
    # whose statistical significance we want to compute against the control group
    TARGET_IV_idx = 1 
    model.fit(TARGET_IV_idx)        

    model.gen(DIRS['MAIN_RESULT_DIR_FOLDER'])

    model.vis(DIRS)

if __name__ == '__main__':
    main()


"""
print(df_X.variables)  
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
