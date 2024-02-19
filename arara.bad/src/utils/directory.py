import os

def get_home_path():
    if "HOMEPATH" in os.environ: # Windows
        HOME_ = os.environ["HOMEPATH"]
    elif "HOME" in os.environ:
        HOME_ = os.environ["HOME"] # Linux
    else:
        raise Exception('please check how to access your OS home path')    
    return HOME_

def manage_dirs(**kwargs):
    WORKSPACE_DIR = None
    if 'WORKSPACE_DIR' in kwargs:
        if kwargs['WORKSPACE_DIR'] is not None:
            WORKSPACE_DIR = kwargs['WORKSPACE_DIR']

    if WORKSPACE_DIR is None:
        HOME_ = get_home_path()
        WORKSPACE_DIR = os.path.join(HOME_, "Desktop", "arara.ws")
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    frag = "= "*17
    print(f"{frag}\nCURRENT WORKSPACE:{WORKSPACE_DIR}\n{frag}")
  
    DIRS = {
        'WORKSPACE_DIR': WORKSPACE_DIR,
    }
    return DIRS


def manage_dirs_p(**kwargs):
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

    MEANS_DIR = os.path.join(PROJECT_LABEL_DIR, 'means.json')

    DIRS.update({
        'PROJECT_DIR': PROJECT_DIR,
        'PROJECT_LABEL_DIR': PROJECT_LABEL_DIR,
        'LABEL_CONFIG_DIR': LABEL_CONFIG_DIR,

        'MAIN_RESULT_DIR_FOLDER': MAIN_RESULT_DIR_FOLDER,
        'MAIN_VIS_DIR_FOLDER': MAIN_VIS_DIR_FOLDER,

        'MEANS_DIR': MEANS_DIR,
        })
    return DIRS
