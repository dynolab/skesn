from typing import Any


class Scheme(object):
    def run(self, **kvargs) -> None: pass

    def save(self, dirname: str, **kvargs) -> None: pass

    def restore_result(self, result: Any) -> None: pass

    def get_name(self) -> str: pass
