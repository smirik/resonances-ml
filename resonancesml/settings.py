import os
from os.path import join as opjoin
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SYN_CATALOG_PATH = opjoin(PROJECT_DIR, '..', 'input', 'all.syn')
CAT_CATALOG_PATH = opjoin(PROJECT_DIR, '..', 'input', 'allnum.cat')
