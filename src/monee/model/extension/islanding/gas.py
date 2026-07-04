"""Gas-carrier islanding mode and grid-forming source model."""

from __future__ import annotations

from monee.model.core import ChildModel, Const, Intermediate, Var, model
from monee.model.grid import GasGrid
from monee.model.network import Network

from .core import GridFormingMixin, IslandingMode


@model
class GridFormingSource(ChildModel, GridFormingMixin):
    """Grid-forming source: pins pressure (and t_pu/t_k) on the junction, leaves
    mass_flow_kgs as a Var to absorb island imbalance."""

    def __init__(
        self,
        pressure_pu: float = 1.0,
        t_k: float = 356.0,
        mass_flow_max_kgs: float = 1e6,
    ) -> None:
        super().__init__()
        self.mass_flow_kgs = Var(
            0, min=-mass_flow_max_kgs, max=mass_flow_max_kgs, name="gf_mass_flow"
        )
        self._pressure_pu = pressure_pu
        self._t_k = t_k

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
        return []


class GasIslandingMode(IslandingMode):
    r"""Gas islanding: connectivity flow plus :math:`pressure_{pu} \le 2 \cdot e` on regular
    junctions (GF junctions already pin pressure via overwrite())."""

    carrier_grid_type = GasGrid
    var_prefix = "gas"
    gated_child_attrs = ("mass_flow_kgs",)

    def __init__(self, big_m_conn: float | None = None) -> None:
        self.big_m_conn = big_m_conn

    def prepare(self, network: Network) -> None:
        self.prepare_common(network)
        for node in network.nodes:
            if isinstance(node.grid, GasGrid) and node.active:
                node.model.e_gas = Var(1, min=0, max=1, integer=True, name="e_gas")
                is_gf = any(
                    isinstance(c.model, GridFormingMixin) and c.active
                    for c in network.childs_by_ids(node.child_ids)
                )
                if is_gf:
                    node.model.c_src_gas = Var(1, min=0, name="c_src_gas")
        for branch in network.branches:
            if isinstance(branch.grid, GasGrid) and branch.active:
                branch.model.c_gas_fwd = Var(0, min=0, name="c_gas_fwd")
                branch.model.c_gas_rev = Var(0, min=0, name="c_gas_rev")

    def add_physical_constraints(
        self, network, gf_nodes, regular_nodes, e_vars
    ) -> list:
        eqs = []
        for node in regular_nodes:
            e = e_vars[node.id]
            # 2.0 is the existing model bound; non-binding when e=1.
            # Gate in the squared domain when the model carries it: on
            # squared-pressure formulations (MIQCQP) pressure_pu is only the
            # sqrt(pressure_squared_pu) intermediate, which no quadratic
            # solver writer accepts; p<=2e <=> p^2<=4e for binary e, and the
            # squared form is linear (gas) or quadratic (water) instead.
            # An Intermediate is not a solver variable yet — formulations that
            # work in plain pressure space keep the squared attr as one; use
            # pressure_pu directly there.
            p2 = getattr(node.model, "pressure_squared_pu", None)
            if p2 is not None and not isinstance(p2, Intermediate):
                eqs.append(p2 <= 4.0 * e)
            else:
                eqs.append(node.model.pressure_pu <= 2.0 * e)
        return eqs
