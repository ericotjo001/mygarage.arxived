"""
python -m main --mode e2e
python -m main --mode e2e --label ggwp-000
python -m main --mode e2e --label ababa_0001-000
python -m main --mode e2e --label ababa_0001-000 --max_epochs 10 --devices 4
"""
import os, argparse, json

def get_home_path():
    if "HOMEPATH" in os.environ: # Windows
        HOME_ = os.environ["HOMEPATH"]
    elif "HOME" in os.environ:
        HOME_ = os.environ["HOME"] # Linux
    else:
        raise Exception('please check how to access your OS home path')    
    return HOME_

def manage_dirs(**kwargs):
    WORKSPACE_DIR = kwargs['WORKSPACE_DIR'] if 'WORKSPACE_DIR' in kwargs else None
    if WORKSPACE_DIR is None:
        HOME_ = get_home_path()
        WORKSPACE_DIR =  os.path.join(HOME_, "Desktop", "anything.ws") 
    print("= + = + = + "*7)
    if not os.path.exists(WORKSPACE_DIR):
        print(f'Setting up workspace at {WORKSPACE_DIR}')
    else:
        print(f'Current workspace: {WORKSPACE_DIR}')
    os.makedirs(WORKSPACE_DIR,exist_ok=True)
    print("All the results you obtain will be found inside the above directory")
    print(" = + = + = + "*7, "\n")

    ####### results #######
    CKPT_DIR = os.path.join(WORKSPACE_DIR, 'checkpoint')
    DATA_CACHE_DIR = os.path.join(WORKSPACE_DIR, 'data.cache')
    os.makedirs(CKPT_DIR,exist_ok=True)
    os.makedirs(DATA_CACHE_DIR,exist_ok=True)

    DIRS = {
        'WORKSPACE_DIR': WORKSPACE_DIR,
        'CKPT_DIR': CKPT_DIR,
        'DATA_CACHE_DIR': DATA_CACHE_DIR,
    }

    DIRS = manage_project_dir(DIRS, **kwargs)
    DIRS = manage_label_dir(DIRS, **kwargs)
    return DIRS

def manage_project_dir(DIRS, **kwargs):
    if not 'projectname' in kwargs: return DIRS
    if kwargs['projectname'] is None: return DIRS
    CKPT_DIR = DIRS['CKPT_DIR']
    DATA_CACHE_DIR = DIRS['DATA_CACHE_DIR']

    PROJECT_DIR = os.path.join(CKPT_DIR, kwargs['projectname'])
    os.makedirs(PROJECT_DIR,exist_ok=True)
    PROJECT_DATA_CACHE_DIR = os.path.join(DATA_CACHE_DIR, kwargs['projectname'])
    os.makedirs(PROJECT_DATA_CACHE_DIR,exist_ok=True)
    DIRS.update({
        'PROJECT_DIR':PROJECT_DIR,
        'PROJECT_DATA_CACHE_DIR': PROJECT_DATA_CACHE_DIR,
        })    
    return DIRS

def manage_label_dir(DIRS, **kwargs):
    if 'label' not in kwargs:
        print("No label argument. Perhaps you're doing some pre/post-processing")
        # do something here if necessary
        return DIRS

    if kwargs['label'] is None:
        print("label is None. Perhaps you're doing some pre/post-processing")
        # do something here if necessary
        return DIRS
    if "PROJECT_DIR" not in DIRS:
        return DIRS

    PROJECT_LABEL_DIR = os.path.join(DIRS['PROJECT_DIR'], kwargs['label'])
    os.makedirs(PROJECT_LABEL_DIR,exist_ok=True)

    LABEL_CONFIG_DIR = os.path.join(PROJECT_LABEL_DIR, 'config_record.json')

    DIRS.update({
        'PROJECT_LABEL_DIR': PROJECT_LABEL_DIR,
        'LABEL_CONFIG_DIR': LABEL_CONFIG_DIR
        })
    return DIRS

class Config():
    def __init__(self, DIRS, adjust=True, **kwargs):
        super(Config, self).__init__()
        self.DIRS = DIRS
        self.kwargs = kwargs

        self.devices = 1
        self.max_epochs = 100
        self.epsilon = 1e-5

        self.adjustables_ = ['devices', 'max_epochs']

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
        super(Controller, self).__init__()
        self.config = config
        self.label_ = config.kwargs['label'].split('-')[0]

    def select_by_label(self, selection=None):
        conf = self.config
        kwargs = conf.kwargs
        label_ = self.label_

        if label_ == 'ababa_0001':
            if selection == 'pipeline':
                return SomePipeline(self)
            elif selection == 'model':
                return "model:ABABA"
            else:
                raise NotImplementedError('not recognized selection')
        else:
            raise NotImplementedError('not recognized label_')

    def select_pipeline(self,):
        return self.select_by_label(selection="pipeline")

    def select_model(self, ):
        return self.select_by_label(selection="model")  

class SomePipeline():
    def __init__(self, controller):
        super(SomePipeline, self).__init__()
        self.controller = controller

    def trainval(self):
        conf = self.controller.config

        print('trainval...')
        model = self.controller.select_model()
        print('model:', model)

        print(f"this pipeline:{type(self)}", )
        print(f"  max epochs:{conf.max_epochs}")
        print(f"  devices   :{conf.devices}")


def end2end(parser):
    parser.add_argument('--label', default=None, type=str, help=None)    

    parser.add_argument('--devices', default=None, type=int, help=None)        
    parser.add_argument('--max_epochs', default=None, type=str, help=None)    

    parser.add_argument('--TOGGLES', default=None, type=str, help=None)
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary

    if kwargs['label'] is None:
        print('>>> preprocessing... DONE!')
        return

    DIRS = manage_dirs(**kwargs)
    conf = Config(DIRS, **kwargs)
    controller = Controller(conf)
    pipe = SomePipeline(controller)

    TOGGLES = "11111" if kwargs['TOGGLES'] is None else kwargs['TOGGLES']

    if TOGGLES[0] == "1":
        pipe.trainval()

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)    

    parser.add_argument('--mode', default=None, type=str, help=None)
    parser.add_argument('--projectname', default="myababa", type=str, help=None)
    # parser.add_argument('--id', nargs='+', default=['a','b']) # for list args
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary
    
    if args.mode == 'e2e':
        end2end(parser)
    else:
        raise NotImplementedError('unrecognized mode')
