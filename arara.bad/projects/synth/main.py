"""
"""

import os
import numpy as np
import pandas as pd

N_DATA = 128
N_FEATURES = 7
def get_synth_df(drug_str=1.0):
    n_control = int(N_DATA/2)
    df = {
        f'x{str(i)}': [1.0 + 0.001*(i+1)*j for j in range(N_DATA)] for i in range(N_FEATURES)}

    group_ = lambda x: "drug" if x%2==1 else "placebo"
    BG = [0.-drug_str*(k%2) for k in range(N_DATA)] 
    TreatmentGroup = [group_(k) for k in range(N_DATA)] 
    df.update({
        "BG": BG, # blood glucose
        'TreatmentGroup': TreatmentGroup,
    })

    df = pd.DataFrame(df)
    for i in range(N_FEATURES):
        df['BG'] = df['BG'] + df[f'x{str(i)}']
    return df

def main():
    print('Welcome!')

    from src.utils import manage_dirs_p

    # print(df_X) # ok

    def run_one_expt(drug_str):
        DIRS = manage_dirs_p(**{
            "projectname" : "synth",
            "label": f"linear_{str(drug_str)}-000",
            })
        df_X = get_synth_df(drug_str=drug_str)
        DV = "BG"
        IV = "TreatmentGroup" 
        EXCLUDE = [DV, IV]
        NUMERICAL_FEATURES = [a for a in list(df_X.columns) if a not in EXCLUDE] 

        from src.data.utils import package_df
        config = {'DV': DV, 'IV': IV, 'DV_description': "Blood Glucose",
            'idx2IV': ["placebo","drug"], 
            'NUMERICAL_FEATURES': NUMERICAL_FEATURES, 'TOKEN_FEATURES': [],}
        data_package = package_df(df_X, config)

        from src.prototype import NumPrototype
        POLES = [-1.0,1.0]
        POLES2FACTOR = {-1.0: "control", 1.0: "drug"} # "drugs" variable
        model = NumPrototype(POLES, POLES2FACTOR, n_gen=len(df_X)*5)

        model.setup_data(data_package)

        # here, we first choose which of config['idx2IV'] will be the treatment group  
        # whose statistical significance we want to compute against the control group
        TARGET_IV_idx = 1 
        model.fit(TARGET_IV_idx)        
        model.gen(DIRS['MAIN_RESULT_DIR_FOLDER'])
        model.vis(DIRS)

    for d in np.linspace(0,1,11):
        drug_str = np.round(d, 2)
        run_one_expt(drug_str)
        print('drug_str:', drug_str, "# done")

if __name__ == '__main__':
    main()