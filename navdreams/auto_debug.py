from IPython import get_ipython

def enable_auto_debug():
    ipython = get_ipython()
    if ipython is None:
        print("WARNING: Auto debugging can not be enabled, please run this script with ipython")
        return
    else:
        ipython.magic("pdb 1")
