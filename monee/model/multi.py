from .core import MultiGridBranchModel


class PowerToGas(MultiGridBranchModel):
    def __init__(self, efficiency) -> None:
        super().__init__()

        self.efficiency = efficiency

    def equations(self, grids, from_models, to_model, **kwargs):
        power_load = [model for model in from_models if "p_w" in model.vars][0]
        heat_exchanger = [model for model in to_model if "q_w" in model.vars][0]

        return power_load.vars["p_w"] * self.efficiency == heat_exchanger.vars["q_w"]


class PowerToHeat(MultiGridBranchModel):
    def __init__(self, efficiency) -> None:
        super().__init__()

        self.efficiency = efficiency

    def equations(self, grids, from_models, to_model, **kwargs):
        power_load = [model for model in from_models if "p_w" in model.vars][0]
        heat_exchanger = [model for model in to_model if "q_w" in model.vars][0]

        return power_load.vars["p_w"] * self.efficiency == heat_exchanger.vars["q_w"]
