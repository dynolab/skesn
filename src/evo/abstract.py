from types import FunctionType
from typing import Any, Union

class Scheme(object):
    def run(self,
        **kvargs,
    ) -> None: pass

    def save(self,
        dirname: str,
        **kvargs,
    ) -> None: pass

    # def restore_result(self,
    #     result: Any,
    # ) -> None: pass

    def restore_result(self,
        runpool_dir: Union[str,None]=None,
        iter_num: Union[int,None]=None,
    ) -> None: pass

    def get_name(self) -> str: pass

    def get_evaluate_f(self) -> FunctionType: pass
