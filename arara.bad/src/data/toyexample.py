from .utils import *

"""
This toy data consists of only NUMERICAL variables.

"""
def get_toy_data(**kwargs):
    """ This is the ezmed data.
    Default looks like this:
        blood-pressure    drugs  feature-0  feature-1  feature-2  feature-3
    0            1.011  placebo     -0.336     -0.957     11.914      9.873
    1            0.800  placebo      0.656     -0.921      8.063      8.607
    2            1.010  placebo      1.164     -0.166     -0.440      9.214
    3            1.084  placebo      0.300      0.914      4.126      9.197
    4            0.968  placebo      0.720      2.227      7.791     11.007
    5            0.986  placebo     -1.172     -0.599     11.499     12.047
    6            0.800  placebo      0.183      0.161     13.155     11.798
    7            1.165    drugX     -0.714      1.408      7.072      9.238
    8            1.200    drugX     -2.188      0.658      6.691     12.478
    9            0.995    drugX      0.263     -0.216      6.677     13.588
    10           1.173    drugX     -1.622      1.989     10.188     12.554
    11           1.149    drugX     -0.635      2.377      9.424      9.300
    12           1.157    drugX      1.047      0.322     13.823      9.969
    13           1.101    drugX      1.159      1.732      9.211      7.456   
    """

    # Main independent variable
    IV = 'drugs' 
    if 'IV' in kwargs: IV = kwargs['IV']

    GROUP_N = {'placebo': 7,'drugX':7}
    if 'GROUP_N' in kwargs: GROUP_N = kwargs['GROUP_N']

    # Main dependent variable
    DV = "blood-pressure"
    if 'DV' in kwargs: DV = kwargs['DV']

    # Manipulate DV based on features and IV
    def f(df, i):
        C, D1, D2 = 0.05, 0.5, 0.6
        if 'C' in kwargs: C = kwargs['C']
        if 'D1' in kwargs: D1 = kwargs['D1']
        if 'D2' in kwargs: D2 = kwargs['D2']

        if df.at[i,'drugs'] == 'drugX':
            df.at[i,'blood-pressure'] += C
        if df.at[i, 'feature-0']>0:
            df.at[i,'blood-pressure'] -= D1*C
        if df.at[i, 'feature-1']>2:
            df.at[i,'blood-pressure'] -= D2*C

        dv_mean, dv_delta = 1.0, 0.2
        df.at[i,'blood-pressure'] = np.clip(df.at[i,'blood-pressure'],
            dv_mean- dv_delta, dv_mean+dv_delta)

    group_relation = f
    if 'group_relation' in kwargs: group_relation = kwargs['group_relation']

    ####### Other features #######
    N_FEATURES = 4
    if 'N_FEATURES' in kwargs: N_FEATURES = kwargs['N_FEATURES']

    FEATURE_MEANS = [0.,1.,10.,10.]
    if 'FEATURE_MEANS' in kwargs: FEATURE_MEANS = kwargs['FEATURE_MEANS']

    FEATURE_SD = [1.,1., 5., 2.,]
    if 'FEATURE_SD' in kwargs: FEATURE_SD = kwargs['FEATURE_SD']

    # CORRUPTION_RATES = [None for _ in range(N_FEATURES)]
    # if 'CORRUPTION_RATES' in kwargs: CORRUPTION_RATES = kwargs['CORRUPTION_RATES']

    def construct_toy_data():
        IV_DATA = []
        for group_,n in GROUP_N.items():
            IV_DATA = IV_DATA + [group_ for _ in range(n)] 
        n_data = len(IV_DATA)

        dv_mean, dv_delta = 1.0, 0.2
        DV_DATA = np.random.uniform(dv_mean-dv_delta, dv_mean+dv_delta ,size=(n_data,))
        DV_DATA = np.round(DV_DATA, 3)

        df = {DV: DV_DATA, IV: IV_DATA, }
        for i in range(N_FEATURES):
            mu, sd = FEATURE_MEANS[i], FEATURE_SD[i]
            FEATURE_DATA = np.random.normal(mu, sd, size=(n_data,))
            df['feature-'+str(i)] = np.round(FEATURE_DATA, 3)
        return pd.DataFrame(df)

    df = construct_toy_data()
    for i in range(len(df)):
        group_relation(df, i)

    config = { 
        "IV": IV, 
        "idx2IV": [group_ for group_ in GROUP_N],
        "DV": DV,
        "DV_description": "blood-pressure (normalized)",
        "NUMERICAL_FEATURES": [f"feature-{i}" for i in range(4)],
        "TOKEN_FEATURES": [],
    }
    return df, config

def get_toy_data_(**kwargs):
    df, config = get_toy_data(**kwargs)
    return package_df(df, config)

def toyexample_demo(parser):
    parser.add_argument('--C', default=0.2, type=float, help=None)
    parser.add_argument('--D1', default=0.5, type=float, help=None)
    parser.add_argument('--D2', default=0.6, type=float, help=None)
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary

    df, config = get_toy_data(**kwargs)
    print(df)
    
    data_package = package_df(df, config=config)
    print(data_package)
