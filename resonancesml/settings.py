import logging
from typing import Dict

import yaml
import sys
import os
from os.path import join as opjoin

PROJECT_DIR = opjoin(os.path.dirname(os.path.abspath(__file__)), '..')
_CATALOGS_DIR = opjoin(PROJECT_DIR, 'input', 'catalogs')
SYN_CATALOG_PATH = opjoin(_CATALOGS_DIR, 'all.syn')
CAT_CATALOG_PATH = opjoin(_CATALOGS_DIR, 'allnum.cat')
PRO_CATALOG_PATH = opjoin(_CATALOGS_DIR, 'allnum.pro')


def _merge(source: Dict, destination: Dict) -> Dict:
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _merge(value, node)
        else:
            destination[key] = value
    return destination


def _env_var_eval(val: Dict):
    """
    Evaluates environment variables for values starts from '$'
    """
    logger = logging.getLogger(__name__)
    for key, value in val.items():
        if isinstance(value, dict):
            _env_var_eval(value)
        elif isinstance(value, str) and value[:1] == '$':
            val[key] = os.environ.get(value[1:], None)
            if not val[key]:
                logger.warning('Environment variable %s is not defined' % value[1:])
    return val


class _ParamBridge:
    """
    Parses file pointed in variable config_path then it parses file pointed in
    variable local_config_path and merge parsed data from this files.

    If an option is in both files her value will be get from local_config_path.

    If an option has value starts from '$' it will be determined by environment
    variable pointed in this value. For example: 'margin: $MARGIN' in this case
    value of margin will be equal to value of environment variable MARGIN.

    Supported only yaml files.
    """
    def __init__(self, config_path: str, local_config_path: str = None):
        with open(config_path) as f:
            try:
                self._params = yaml.load(f)
            except Exception as e:
                raise e

        if local_config_path and os.path.exists(local_config_path):
            local_params = {}
            with open(local_config_path) as f:
                try:
                    local_params = yaml.load(f)
                except Exception as e:
                    raise e

            if local_params:
                self._params = _merge(local_params, self._params)
        self._params = _env_var_eval(self._params)

    def __getitem__(self, key: str):
        return getattr(self, key, self._params[key])


_config_path = None
_params = None


def set_config_path(value: str):
    global _config_path
    if not _config_path:
        _config_path = value
    else:
        raise Exception('Cannot be invoked twice')


def params() -> _ParamBridge:
    """
    Returns instance of parameters are built from file pointed by _config_path or from config.yml
    """
    global _params
    global _config_path

    if _params is None:
        local_config_filename = None
        if 'pytest' in ' '.join(sys.argv):
            config_filename = 'config_unittest.yml'
        else:
            config_filename = 'config.yml'
            local_config_filename = 'local_config.yml'
        if not _params:
            if local_config_filename:
                local_config_path = opjoin(PROJECT_DIR, local_config_filename)
            else:
                local_config_path = None

            if not _config_path:
                _config_path = opjoin(PROJECT_DIR, config_filename)
            _params = _ParamBridge(_config_path, local_config_path)
    return _params
