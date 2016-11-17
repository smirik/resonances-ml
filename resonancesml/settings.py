import os
from os.path import join as opjoin
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_CATALOGS_DIR = opjoin(PROJECT_DIR, '..', 'input', 'catalogs')
SYN_CATALOG_PATH = opjoin(_CATALOGS_DIR, 'all.syn')
CAT_CATALOG_PATH = opjoin(_CATALOGS_DIR, 'allnum.cat')
PRO_CATALOG_PATH = opjoin(_CATALOGS_DIR, 'allnum.pro')

_RESONANCE_TABLES_DIR = opjoin(PROJECT_DIR, '..', 'input', 'resonance_tables')
RESONANCE_TABLES = {
    'Jupiter Saturn': opjoin(_RESONANCE_TABLES_DIR, 'matrix-js.res')
}
_LIBRATIONS_DIR = opjoin(PROJECT_DIR, '..', 'input', 'librations')
LIBRATIONS_FOLDERS = {
    '4J-2S-1': opjoin(_LIBRATIONS_DIR, '4J-2S-1'),
    'all-JS': opjoin(_LIBRATIONS_DIR, 'libration_asteroids_JS')
}
