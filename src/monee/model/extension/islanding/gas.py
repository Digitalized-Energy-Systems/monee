"""Gas-carrier islanding mode and grid-forming source model."""

from __future__ import annotations

from monee.model.core import ChildModel, Const, Var, model
from monee.model.grid import GasGrid

from .core import GridFormingMixin, PressureGatedIslandingMode


@model
class GridFormingSource(ChildModel, GridFormingMixin):
    """Grid-forming source: pins pressure (and t_pu/t_k) on the junction, leaves
    mass_flow_kgs as a Var to absorb island imbalance."""

    def __init__(
        self,
        pressure_pu: float = 1.0,
        t_k: float = 356.0,
        mass_flow_max_kgs: float = 1e6,
        nominal_mass_flow_kgs: float | None = None,
    ) -> None:
        super().__init__()
        self.mass_flow_kgs = Var(
            0, min=-mass_flow_max_kgs, max=mass_flow_max_kgs, name="gf_mass_flow"
        )
        self._pressure_pu = pressure_pu
        self._t_k = t_k
        self._nominal_mass_flow_kgs = nominal_mass_flow_kgs

    def overwrite(self, node_model, grid) -> None:
        # Pin only when leading an island without an ext grid (stamped by
        # IslandingMode.stamp_gf_leadership); a second absolute pressure pin
        # in the same hydraulic island over-determines the drop equations.
        if not getattr(self, "_gf_leading", True):
            return
        node_model.pressure_pu = Const(self._pressure_pu)
        node_model.pressure_squared_pu = Const(self._pressure_pu**2)
        node_model.t_pu = Const(self._t_k / grid.t_ref_k)
        node_model.t_k = Const(self._t_k)

    def equations(self, grid, node_model, **kwargs):
        # A former only needs a free balancing Var while it IS the island
        # reference. Where an ext grid leads the component, stamp_gf_leadership
        # marks EVERY former non-leading, overwrite() pins nothing, and this
        # returning [] leaves mass_flow_kgs appearing in the node balance alone —
        # a degenerate free injection that no equation and no objective pin
        # (plain energy flow carries no objective). The LP then returns an
        # arbitrary split: on the simbench MES a 38-unit promoted fleet
        # delivered 0.0013 of its 0.0118 kg/s while the slack covered the rest.
        # With a nominal, a non-leading former holds the setpoint of the Source
        # it was promoted from — which is what it physically is until an island
        # forms. None keeps the fully-free Var (previous behaviour).
        if self._nominal_mass_flow_kgs is None or getattr(self, "_gf_leading", True):
            return []
        return [self.mass_flow_kgs == self._nominal_mass_flow_kgs]


class GasIslandingMode(PressureGatedIslandingMode):
    r"""Gas islanding: connectivity flow plus :math:`pressure_{pu} \le 2 \cdot e` on regular
    junctions (GF junctions already pin pressure via overwrite())."""

    carrier_grid_type = GasGrid
    var_prefix = "gas"

    def __init__(self, big_m_conn: float | None = None) -> None:
        self.big_m_conn = big_m_conn
