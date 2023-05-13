import os.path
from typing import Any, Dict, List, Union

import yaml

import src.config as cfg

class RunpoolSync(object):
    # class IterLog:
    #     def __init__(self, iter_num: int, best_: List[Dict[str, Any]]) -> None:
    #         self._iter_num = iter_num
    #         self._changes = changes

    #     def yaml(self) -> Dict[str, Any]:
    #         return {
    #             'iter_num': self._iter_num, 'changes': self._changes,
    #         }

    # class UpdateLog:
    #     def __init__(self, iter_num: int, changes: List[Dict[str, Any]]) -> None:
    #         self._iter_num = iter_num
    #         self._changes = changes

    #     def yaml(self) -> Dict[str, Any]:
    #         return {
    #             'iter_num': self._iter_num, 'changes': self._changes,
    #         }

    _SPEC_INFO_KEY          = 'info'
    _SPEC_ITER_CNT_KEY      = 'iter_cnt'
    _SPEC_TOTAL_GEN_CNT_KEY = 'total_gen_cnt'
    _SPEC_ITER_LOG_KEY      = 'iter_log'
    _SPEC_UPDATE_LOG_KEY    = 'update_log'
    _SPEC_ITER_2_DATA_KEY   = 'iter_2_data'
    _SPEC_FIT_DATAS_KEY     = 'fit_datas'
    _SPEC_VALID_DATAS_KEY   = 'valid_datas'

    def __init__(self,
        iter_num: int,
        runpool_dir: str,
        evaluate_cfg: cfg.EsnEvaluateConfigField,
        evo_cfg: cfg.IEvoConfig,
    ) -> None:
        self._iter_num = iter_num
        self._runpool_dir = runpool_dir
        self._evaluate_cfg = evaluate_cfg
        self._evo_cfg = evo_cfg

        self._iter_dir = os.path.join(self._runpool_dir, f'iter_{self._iter_num}')

        self._spec_path = os.path.join(self._runpool_dir, 'spec.yaml')
        self._spec = self._load_spec()
        if self._spec is None or len(self._spec) == 0:
            self._init_spec()
        else:
            self._update_spec()

    @property
    def iter_num(self) -> int: return self._iter_num

    @property
    def iter_dir(self) -> str: return self._iter_dir

    @property
    def runpool_dir(self) -> str: return self._runpool_dir

    @property
    def spec(self) -> Dict[str, Any]: return self._spec

    def sync_spec(self) -> None:
        old_spec = self._load_spec()
        if self._spec != old_spec:
            return self._flush_spec()

    def _load_spec(self) -> Dict[str, Any]:
        ret = None
        if not os.path.exists(self._spec_path):
            return ret

        with open(self._spec_path, 'r') as f:
            ret = yaml.full_load(f)
        return ret

    def _flush_spec(self) -> None:
        with open(self._spec_path, 'w') as f:
            yaml.safe_dump(self._spec, f)

    def _init_spec(self) -> None:
        self._spec = {
            self._SPEC_INFO_KEY: None,
            self._SPEC_ITER_CNT_KEY: 0,
            self._SPEC_TOTAL_GEN_CNT_KEY: 0,
            self._SPEC_ITER_LOG_KEY: [],
            self._SPEC_UPDATE_LOG_KEY: [],
            self._SPEC_ITER_2_DATA_KEY: [],
            self._SPEC_FIT_DATAS_KEY: [],
            self._SPEC_VALID_DATAS_KEY: [],
        }
        self._update_spec_info()

    def _update_spec(self) -> None:
        self._spec[self._SPEC_ITER_CNT_KEY] += 1
        self._spec[self._SPEC_TOTAL_GEN_CNT_KEY] += self._evo_cfg.MaxGenNum

        changes = self._get_changes()
        if changes is not None and len(changes) > 0:
            self._spec[self._SPEC_UPDATE_LOG_KEY].append(changes)

        self._update_spec_info()


    def _get_changes(self) -> Union[Dict, None]:
        # TODO :
        return None

    def _update_spec_info(self) -> None:
        self._spec[self._SPEC_INFO_KEY] = {}

        self._spec[self._SPEC_INFO_KEY]['eval'] = self._evaluate_cfg.yaml()

        evo_info = {}
        if self._evo_cfg.HromoLen is not None: evo_info['hromo_len'] = self._evo_cfg.HromoLen
        if self._evo_cfg.RandSeed is not None: evo_info['rand_seed'] = self._evo_cfg.RandSeed
        self._spec[self._SPEC_INFO_KEY]['evo'] = evo_info

