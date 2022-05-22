import config

from pshipilov_dev.src.evo.evo_scheme import EvoScheme

from deap import base

class EsnEvoScheme(EvoScheme):
    def __init__(self,
        name: str,
        cfg: config.EvoSchemeConfigField,
        esn_cfg: config.EsnConfigField,
        esn_creator_f,
        ind_creator_f=None,
        ) -> None:

        self._esn_cfg = esn_cfg

        super().__init__(
            name,
            cfg,
            esn_creator_f,
            ind_creator_f,
        )
