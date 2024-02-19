from .toyexample import *

""" DATA FORMAT
Standardized the order of features in X: (IV, NUMERICAL_FEATURES, TOKEN_FEATURES) 

Dv+Num: X is [DV, NUMERICAL[1,2,...]]

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)
    parser.add_argument('--mode', default=None, type=str, help=None)
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary

    if kwargs['mode'] == 'toyexample_demo':
        toyexample_demo(parser)
    else: 
        raise NotImplementedError("unrecognized mode")
