import os, argparse, json

def ezmed_end_to_end(parser):
    parser.add_argument('--C', default=None, type=float, help="strength of effect")
    args, unknown = parser.parse_known_args()
    kwargs = vars(args) # is a dictionary

    DIRS = manage_dirs_p(**kwargs)
    conf = Config(DIRS, **kwargs)
    controller = Controller(conf)
    pipe = controller.select_pipeline()

    pipe.fitgenvis()

def manage_dirs_p(**kwargs):
    from src.utils import manage_dirs
    DIRS = manage_dirs(**kwargs)  

    PROJECT_DIR = os.path.join(DIRS['WORKSPACE_DIR'], kwargs['projectname'])
    os.makedirs(PROJECT_DIR, exist_ok=True)
    PROJECT_LABEL_DIR = os.path.join(PROJECT_DIR, kwargs['label'])
    os.makedirs(PROJECT_LABEL_DIR, exist_ok=True)
    LABEL_CONFIG_DIR = os.path.join(PROJECT_LABEL_DIR, 'config.json')

    MAIN_RESULT_DIR_FOLDER = os.path.join(PROJECT_LABEL_DIR,'main.results')
    os.makedirs(MAIN_RESULT_DIR_FOLDER, exist_ok=True)
    MAIN_VIS_DIR_FOLDER = os.path.join(PROJECT_LABEL_DIR,'main.vis')
    os.makedirs(MAIN_VIS_DIR_FOLDER, exist_ok=True)

    DIRS.update({
        'PROJECT_DIR': PROJECT_DIR,
        'PROJECT_LABEL_DIR': PROJECT_LABEL_DIR,
        'LABEL_CONFIG_DIR': LABEL_CONFIG_DIR,

        'MAIN_RESULT_DIR_FOLDER': MAIN_RESULT_DIR_FOLDER,
        'MAIN_VIS_DIR_FOLDER': MAIN_VIS_DIR_FOLDER,
        })
    return DIRS

class Config():
    def __init__(self, DIRS, adjust=True, **kwargs): 
        self.DIRS = DIRS
        self.kwargs = kwargs
        
        self.C = 0.05

        self.adjustables_ = ['C']

        if adjust:
            self.adjust_config()
            self.print_config()

    def print_config(self):
        config_record = {}
        for x,y in vars(self).items():
            if x in ['DIRS', 'kwargs']: continue
            print(f'[{x}]: {y}')
            config_record[x] = y

        with open(self.DIRS['LABEL_CONFIG_DIR'],'w') as f:
            json.dump(config_record,f, indent=4)            

    def adjust_config(self):        
        adjustables = {x:y for x,y in self.kwargs.items() if x in self.adjustables_}

        for x,y in adjustables.items():
            if y is not None:
                setattr(self, x, y)


class Controller():
    def __init__(self, config):
        self.config = config
        self.label_ = config.kwargs['label'].split('-')[0]                

    def select_by_label(self, selection=None):
        conf = self.config
        kwargs = conf.kwargs
        label_ = self.label_

        if label_ in ['medi0000']:
            if selection == 'pipeline':
                return ezPipeline00(self)
            elif selection == 'data':
                if label_ in ['medi0000']:
                    from src.data import get_toy_data_
                    GROUP_N = {'placebo': 17,'drugX':17}
                    data_package = get_toy_data_(GROUP_N=GROUP_N, C=conf.C) 
                    # { 'X': (n, 1 + n_features) np.array, 
                    #   'Y': (n, 1) np.array,
                    #   'config': config }
                    return data_package
                else:
                    raise NotImplementedError("customize here")
            elif selection == 'model':
                if label_ in ['medi0000']:
                    from src.prototype import NumPrototype
                    POLES = [-1.0,1.0]
                    POLES2FACTOR = {-1.0:"placebo", 1.0:"drugX"}
                    model = NumPrototype(POLES, POLES2FACTOR)
                    return model
                else:
                    raise NotImplementedError("customize here")
            else:
                raise NotImplementedError('not recognized selection')
        else:
            raise NotImplementedError('not recognized label_')

    def select_pipeline(self,):
        return self.select_by_label(selection="pipeline")

    def select_data(self,):
        return self.select_by_label(selection="data")

    def select_model(self,):
        return self.select_by_label(selection="model")        

class ezPipeline00():
    def __init__(self, controller):
        self.controller = controller

    def fitgenvis(self):
        controller = self.controller
        conf = controller.config
        DIRS = conf.DIRS

        data_package = controller.select_data()

        model = controller.select_model()
        model.setup_data(data_package)

        # here, we first choose which of config['idx2IV'] will be the treatment group  
        # whose statistical significance we want to compute against the control group
        TARGET_IV_idx = 1 
        model.fit(TARGET_IV_idx)        

        model.gen(DIRS['MAIN_RESULT_DIR_FOLDER'])

        model.vis(DIRS)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)
    parser.add_argument('-m','--mode', default=None, type=str, help=None)
    parser.add_argument('--projectname', default="ezmed", type=str, help=None)
    parser.add_argument('--label', default="anything-777", type=str, help=None)
    # parser.add_argument('--id', nargs='+', default=['a','b']) # for list args
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary

    if kwargs['mode'] == 'e2e':
        ezmed_end_to_end(parser)
    else:
        raise NotImplementedError('Unrecognized mode,')