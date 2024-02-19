""" NumPrototype
Prototype for data with the following assumptions:
1. only numerical data
2. only two groups
"""
from .utils import *

class NumPrototype():
    def __init__(self, POLES, POLES2FACTOR, n_gen=256, n_repeat=17, n_partitions=7):
        self.POLES = POLES
        self.POLES2FACTOR = POLES2FACTOR 
        self.N_PARTITIONS = n_partitions
        self.n_repeat = n_repeat
        self.n_gen = n_gen
        self.alpha = 0.05

    def setup_data(self, data_package):
        """ data_package:
            { 'X': (n, 1 + n_features) np.array, 
              'Y': (n, 1) np.array,
              'config': config } 

        X data format is Dv+Num. See src/data/__init__.py
    
        config is like {
            'IV': 'drugs', 
            'idx2IV': ['placebo', 'drugX'], 
            'DV': 'blood-pressure', 
            'NUMERICAL_FEATURES': ['feature-0', 'feature-1', 'feature-2', 'feature-3'], 'TOKEN_FEATURES': []
        }    
        """
        self.data_package = data_package
        conf = data_package['config']

    def simple_polarization(self, iv, right=1.0, left=-1.0):
        ####### POLARIZATION #######
        # With polarization, we will be able to generate more data later.
        # This is the most primitive form our idea, the complete form will use a full-blown generative AI        
        if iv==self.TARGET_IV: return right
        return left

    def simple_polarization_(self, *args, **kwargs):
        return np.vectorize(self.simple_polarization)(*args, **kwargs)

    def fetch_target_idxs(self, target_column):
        return np.where(target_column==self.TARGET_IV)[0]

    def fit(self, TARGET_IV_idx):
        dpack = self.data_package
        X = dpack['X']
        Y = dpack['Y']
        conf = dpack['config']
        # print(X.shape, Y.shape) # (n, n_f), (n,)

        self.TARGET_IV = conf['idx2IV'][TARGET_IV_idx]
        self.target_idxs = self.fetch_target_idxs(X[:,0])

        X_ = np.array(X)
        X_[:,0] = self.simple_polarization_(X_[:,0], right=self.POLES[1],
            left=self.POLES[0])
        self.X_ = X_

        self.origin_data = {}
        for p in self.POLES:
            idxs = np.where(X_[:,0]==p)[0]
            if p==self.POLES[1]: 
                assert(np.all(idxs == self.target_idxs))
            self.origin_data[p] = Y[idxs]

        from sklearn.mixture import BayesianGaussianMixture
        n_components = X.shape[1]
        self.gmodel = BayesianGaussianMixture(n_components=n_components)
        self.gmodel.fit(X_)

        # from sklearn.svm import SVR
        # self.rmodel = SVR()
        # from sklearn.ensemble import GradientBoostingRegressor
        # self.rmodel = GradientBoostingRegressor()
        # from sklearn.ensemble import AdaBoostRegressor
        # self.rmodel = AdaBoostRegressor()
        from sklearn.ensemble import RandomForestRegressor
        self.rmodel = RandomForestRegressor()
        self.rmodel.fit(X_,Y)

        self.delta = self.POLES[1] - self.POLES[0]

    def gen(self, MAIN_RESULT_DIR_FOLDER):
        dpack = self.data_package
        X = dpack['X']
        Y = dpack['Y']

        idxs = self.target_idxs
        n_gen = self.n_gen
        
        for j in range(self.n_repeat):
            gX, gX_labels = self.gmodel.sample(n_samples=n_gen)
            # gX_labels are cluster labels, not Y
            gY = self.rmodel.predict(gX)

            gen_data = {}
            gen_polar = {}
            counts, edges = np.histogram(gX[:,0], bins=self.N_PARTITIONS)
            # print(edges) # there are self.N_PARTITIONS+1 edges

            df = pd.DataFrame(np.concatenate((gX,np.expand_dims(gY,axis=1)),axis=1))
            for i,c in enumerate(counts):
                pmid = 0.5*(edges[i]+edges[i+1])
                idxs = df[0].between(edges[i], edges[i+1])

                # if not np.any(idxs): continue
                if np.sum(idxs)<2: continue # temporary

                # print(df[idxs].to_numpy()) # ok
                gen_data[pmid] = df[idxs].to_numpy()[:,-1] # gY
                # print(gen_data[p].shape) # ok
                gen_polar[pmid] = df[idxs].to_numpy()[:, 0]

            PACKAGE_DIR = os.path.join(MAIN_RESULT_DIR_FOLDER, f'{j}-gen.data')
            package = {'gen_data': gen_data, 'gen_polar': gen_polar}
            joblib.dump(package, PACKAGE_DIR)

    def vis(self, DIRS):
        dpack = self.data_package
        X = dpack['X']
        Y = dpack['Y']
        # conf = dpack['config']

        for i in range(self.n_repeat):
            self.plot_polarized_scatter(i, DIRS)        

        self.plot_pcurve(DIRS)


    def plot_polarized_scatter(self, i, DIRS):     
        dpack = self.data_package
        conf = dpack['config']
        DV_description = conf['DV_description'] if 'DV_description' in conf else conf['DV']

        PACKAGE_DIR = os.path.join(DIRS['MAIN_RESULT_DIR_FOLDER'], f'{i}-gen.data')
        package = joblib.load(PACKAGE_DIR)
        gen_data = package['gen_data']
        gen_polar = package['gen_polar']


        SCATTER_VIS_DIR = os.path.join(DIRS['MAIN_VIS_DIR_FOLDER'], f'scatter-{i}.png')
        plt.rc('font', **{'size': 17})
        plt.figure()
        plt.gcf().add_subplot(1,1,1)
        assert(len(self.origin_data)==2)
        main_colors_ = ( (0.0,0.57,0.77, 1), (0.77,0.57,0., 1),)
        for i, (p,y) in enumerate(self.origin_data.items()):
            # print(p, y)
            label = self.POLES2FACTOR[p]
            x_ = [p + np.random.normal(0,0.05*self.delta) for _ in range(len(y))]
            plt.gca().scatter(x_, y, s=36, c=[main_colors_[i] for _ in range(len(x_))], 
                alpha=0.77, label=label)

        n_ = len(gen_data)
        colors = np.array([1.0,0.0,0.0])
        for i,(p,y) in enumerate(gen_data.items()):
            c_ = colors + 0
            c_[0] = c_[0] - i/n_
            c_[2] = c_[2] + i/n_
            plt.gca().scatter(gen_polar[p], y, marker='+', s=17, alpha=0.27, 
                c=[c_ for _ in range(len(y))])

        plt.gca().set_xlabel("polarity")
        plt.gca().set_ylabel(DV_description)
        plt.legend()
        plt.tight_layout()
        plt.savefig(SCATTER_VIS_DIR)
        plt.close()

    def plot_pcurve(self, DIRS):
        PCURVE_VIS_DIR = os.path.join(DIRS['MAIN_VIS_DIR_FOLDER'], 'pvalues.png')
        PCURVE_RESULT_DIR = os.path.join(DIRS['MAIN_VIS_DIR_FOLDER'], 'pvalues.json')

        F_original, p_original = ss.f_oneway(self.origin_data[self.POLES[0]], 
            self.origin_data[self.POLES[1]]) 
        ppackage = self.get_ppackage(DIRS)

        mean_polar = np.array([np.mean(pack_['polarizations']) for ip, pack_ in ppackage.items()])
        mean_F = np.array([np.mean(pack_['Fvalues']) for ip, pack_ in ppackage.items()])
        # sd_F = np.array([np.var(pack_['Fvalues'])**0.5 for ip, pack_ in ppackage.items()])
        mean_logp = np.array([np.mean(-np.log10(pack_['pvalues'])) 
            for ip, pack_ in ppackage.items()])
        sd_logp = np.array([np.var(-np.log10(pack_['pvalues']))**0.5 
            for ip, pack_ in ppackage.items()])
        mean_pvalue = np.array([np.mean(pack_['pvalues']) for ip, pack_ in ppackage.items()])
        sd_pvalue = np.array([np.var(pack_['pvalues'])**0.5 for ip, pack_ in ppackage.items()])

        plt.rc('font', **{'size': 17})

        plt.figure(figsize=(9,5))
        plt.gcf().add_subplot(1,1,1)
        greyplot = (0.17,0.17,0.17)
        plt.gca().plot(mean_polar, mean_logp, linewidth=0.77, marker='.', color=greyplot)
        plt.gca().fill_between(mean_polar, mean_logp-sd_logp, mean_logp + sd_logp, color=(1.,0.47,0.27), alpha=0.1)        
        plt.gca().set_ylabel("-log(p-value)")
        plt.gca().set_xlabel("polarization")

        plt.gca().scatter([self.POLES[1]], [-np.log10(p_original)], marker='^',color=(0.77,0.17,0.97), 
            label="Original data")

        NLP = -np.log10(self.alpha)
        plt.gca().axhline(y=NLP, linestyle='dashed', color=greyplot, 
            linewidth=0.77, label=r'$\alpha$'+f"={self.alpha}")
        if np.all(mean_logp<NLP):
            plt.gca().set_ylim([None, NLP*1.1])

        offset_to_right = 0.35
        lgnd = plt.legend(prop={'size': 11}, framealpha=0.77, 
            loc='center left', bbox_to_anchor=(1 + offset_to_right, 0.37))

        plt.gca().twinx()
        lightpink = (1,0.77,1.0, 0.42)
        textpink = (1.0,0.27,1.0, 0.42)
        plt.plot(mean_polar, mean_pvalue, marker='x', linewidth=1, c=lightpink,
            markeredgecolor=textpink)
        plt.gca().fill_between(mean_polar, mean_pvalue- sd_pvalue , mean_pvalue + sd_pvalue, color=lightpink, alpha=0.1)        
        plt.gca().tick_params(axis='y', colors=textpink)
        plt.gca().set_ylabel('p-value', c=textpink)

        plt.gca().plot([self.POLES[1]], [p_original], marker='+',c=textpink)

        plt.tight_layout()
        plt.savefig(PCURVE_VIS_DIR)

        out_package = {
            'F_original': F_original, 
            'p_original': p_original,
            'mean_polar': list(mean_polar), 
            'mean_pvalue': list(mean_pvalue),
            'mean_logp': list(mean_logp),
            'mean_F': list(mean_F),
        }
        with open(PCURVE_RESULT_DIR,'w') as f:
            json.dump(out_package, f, indent=4)

    def get_ppackage(self, DIRS):
        dpack = self.data_package
        Y = dpack['Y']
        ppackage = {}
        for i in range(self.n_repeat):
            PACKAGE_DIR = os.path.join(DIRS['MAIN_RESULT_DIR_FOLDER'], f'{i}-gen.data')
            package = joblib.load(PACKAGE_DIR)
            gen_data = package['gen_data']
            # gen_polar = package['gen_polar']

            idx2p = [p for p in gen_data]

            control_group_dist = [(p-self.POLES[0])**2 for p in gen_data] 
            control_group_idx = np.argmin(control_group_dist)
            control_p = idx2p[control_group_idx]
            y_control = gen_data[control_p]
            y_control = np.concatenate([y_control, Y])

            alt_group_dist = [(p-self.POLES[1])**2 for p in gen_data]
            alt_group_idx = np.argmin(alt_group_dist)
            alt_p = idx2p[alt_group_idx]
            assert(alt_p in gen_data)

            for i_partition, (p,y) in enumerate(gen_data.items()):
                if not i_partition in ppackage:
                    ppackage[i_partition] = {'Fvalues':[], 'pvalues':[], 'polarizations': []}

                F, pvalue = ss.f_oneway(y_control, y)
                ppackage[i_partition]['Fvalues'].append(F)
                ppackage[i_partition]['pvalues'].append(pvalue)
                ppackage[i_partition]['polarizations'].append(p)

        """ length of ppackage is N_PARTITIONS """
        # for i_partition, pack_ in ppackage.items():
        #     n_ = len(pack_["Fvalues"]) # equals to n_repeat
        #     print(i_partition, n_, ) 
        return ppackage
