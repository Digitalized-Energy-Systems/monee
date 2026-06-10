"""Native JSON (de)serialization for monee :class:`~monee.model.network.Network`.

The native format is a plain ``dict`` (JSON-friendly) with this shape::

    {
        "version": 2,
        "grids":    {name: {"model_type": str, "values": {...}}},
        "nodes":    [{"id", "grid_id", "child_ids", "position",
                      "active", "name", "values", "model_type"}],
        "childs":   [{"id", "active", "name", "values", "model_type"}],
        "branches": [{"id", "from_node", "to_node", "grid_id", "grid_ids",
                      "active", "name", "values", "model_type"}],
        "compounds":[{"id", "connected_to", "values", "model_type"}],
    }

Each model attribute in a ``values`` block is encoded with full fidelity:
plain scalars pass through unchanged, while :class:`~monee.model.core.Var`,
:class:`~monee.model.core.Const` and :class:`~monee.model.core.Intermediate`
are tagged with a reserved ``__type__`` key so they round-trip exactly
(``Var`` keeps its bounds, ``integer`` flag and ``name``).

The reader stays backward compatible with the legacy format produced by older
versions and by :mod:`monee.io.matpower` (untagged ``{"value", "max", "min"}``
Var dicts).

Both the public, solver-visible attributes of a model *and* its JSON-encodable
private attributes (e.g. a storage's ``_p_max`` / ``_lossy``) are serialized, so
models whose behaviour is configured through private state round-trip faithfully.
Private attributes holding object references (grids, sub-models) are skipped on
purpose — they are rebuilt by the compound ``create()`` / branch ``init()`` hooks.

Known limitations (not represented in the native format): user-supplied
``network.constraint`` / ``network.objective`` callables, network-level
extensions registered via ``add_extension`` (linepack, LTC, islanding), and
non-default formulations applied with ``apply_formulation`` — the latter are
re-derived from the default formulation rules on load.
"""

import inspect
import json
import numbers

from monee.model import Network
from monee.model.core import (
    Compound,
    Const,
    Intermediate,
    PostProcess,
    Var,
    component_list,
)

#: Bumped whenever the on-disk structure changes in a non-additive way.
FORMAT_VERSION = 2

#: Reserved key used to tag the concrete type of an encoded value object.
_TYPE_KEY = "__type__"


class PersistenceException(Exception):
    pass


# --------------------------------------------------------------------------- #
# Value (de)serialization                                                      #
# --------------------------------------------------------------------------- #
def _encodable(value):
    """Return ``True`` if *value* can be losslessly written to the native format."""
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (Var, Const, Intermediate, PostProcess)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_encodable(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _encodable(v) for k, v in value.items()
        )
    # numbers.Number also covers numpy integer/float scalars.
    return isinstance(value, numbers.Number)


def _encode_value(value):
    """Encode a single model attribute value to a JSON-serializable form."""
    if isinstance(value, Var):
        return {
            _TYPE_KEY: "Var",
            "value": value.value,
            "max": value.max,
            "min": value.min,
            "integer": value.integer,
            "name": value.name,
        }
    if isinstance(value, Const):
        return {_TYPE_KEY: "Const", "value": value.value}
    if isinstance(value, Intermediate):
        return {_TYPE_KEY: "Intermediate", "value": value.value}
    if isinstance(value, PostProcess):
        # Only the computed value is portable; the lambda is re-attached on the
        # next solve (its model's equations()/__init__ recreate it).
        return {_TYPE_KEY: "PostProcess", "value": value.value}
    if isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(v) for v in value]
    return value


def _decode_value(value):
    """Inverse of :func:`_encode_value`, tolerant of the legacy format."""
    if isinstance(value, dict):
        tag = value.get(_TYPE_KEY)
        if tag == "Var":
            return Var(
                value["value"],
                max=value.get("max"),
                min=value.get("min"),
                integer=value.get("integer", False),
                name=value.get("name"),
            )
        if tag == "Const":
            return Const(value["value"])
        if tag == "Intermediate":
            return Intermediate(value["value"])
        if tag == "PostProcess":
            # Carry the stored value via a constant lambda; the model's next
            # solve re-attaches the real computation.
            stored = value["value"]
            return PostProcess(lambda _vals, _v=stored: _v, value=stored)
        if tag is not None:
            raise PersistenceException(f"Unknown encoded value type: {tag!r}")

        # --- legacy (untagged) handling ------------------------------------ #
        # Older files and matpower import emit bare Var dicts without a tag.
        if "value" in value and ("max" in value or "min" in value):
            return Var(
                value["value"],
                max=value.get("max"),
                min=value.get("min"),
                integer=value.get("integer", False),
                name=value.get("name"),
            )
        if set(value.keys()) == {"value"}:
            # Legacy Intermediate/Const both serialized as {"value": x}; restore
            # the numeric value as an Intermediate rather than dropping it.
            return Intermediate(value["value"])
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    return value


def _encode_values(model_dict):
    return {k: _encode_value(v) for k, v in model_dict.items()}


def _decode_values(values_dict):
    return {k: _decode_value(v) for k, v in values_dict.items()}


def _json_default(obj):
    """Fallback encoder for stray objects (e.g. numpy scalars) in ``values``."""
    if isinstance(obj, (Var, Const, Intermediate, PostProcess)):
        return _encode_value(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return vars(obj)


# --------------------------------------------------------------------------- #
# Deserialization (dict -> Network)                                            #
# --------------------------------------------------------------------------- #
def init_model(model_type, preprocessed_dict):
    model_type_dict = {
        component_cls.__name__: component_cls for component_cls in component_list
    }
    if model_type not in model_type_dict:
        raise PersistenceException(
            f"The type {model_type} is not known! Maybe you forgot to decorate "
            f"your model class with @model?"
        )
    model_cls = model_type_dict[model_type]

    # Required params get 0; optional params keep their default. Real attr
    # values arrive via setattr, bypassing constructor transforms so the
    # serialized state round-trips exactly.
    init_kwargs = {}
    for argname, param in list(
        inspect.signature(model_cls.__init__).parameters.items()
    )[1:]:
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # *args / **kwargs accept nothing by default.
            continue
        init_kwargs[argname] = (
            0 if param.default is inspect.Parameter.empty else param.default
        )
    model = model_cls(**init_kwargs)
    for key, value in preprocessed_dict.items():
        setattr(model, key, value)
    return model


# Kept for backward compatibility (older callers/imports).
def preprocess_dict(model_dict):
    return _decode_values(model_dict)


def native_dict_to_network(dict_struct) -> Network:
    network = Network(None)

    # Build grid objects without mutating the caller's dict.
    grid_by_name = {}
    for name, grid_entry in dict_struct["grids"].items():
        grid_by_name[name] = init_model(
            grid_entry["model_type"], _decode_values(grid_entry["values"])
        )

    for child_dict in dict_struct["childs"]:
        model = init_model(
            child_dict["model_type"], _decode_values(child_dict["values"])
        )
        child_id = network.child(
            model,
            overwrite_id=child_dict["id"],
            name=child_dict.get("name"),
        )
        if not child_dict.get("active", True):
            network.deactivate_by_id(type(network.child_by_id(child_id)), child_id)

    for node_dict in dict_struct["nodes"]:
        model = init_model(
            node_dict["model_type"], _decode_values(node_dict["values"])
        )
        position = node_dict.get("position")
        node_id = network.node(
            model,
            child_ids=list(node_dict["child_ids"]),
            grid=grid_by_name[node_dict["grid_id"]],
            overwrite_id=node_dict["id"],
            name=node_dict.get("name"),
            position=tuple(position) if isinstance(position, list) else position,
        )
        node = network.node_by_id(node_id)
        node.active = node_dict.get("active", True)

    for compound_dict in dict_struct.get("compounds", []):
        model = init_model(
            compound_dict["model_type"], _decode_values(compound_dict["values"])
        )
        compound_id = network.compound(
            model,
            overwrite_id=compound_dict.get("id"),
            **compound_dict["connected_to"],
        )
        compound = network.compound_by_id(compound_id)
        compound.name = compound_dict.get("name")
        if not compound_dict.get("active", True):
            network.deactivate_by_id(Compound, compound_id)

    for branch_dict in dict_struct["branches"]:
        model = init_model(
            branch_dict["model_type"], _decode_values(branch_dict["values"])
        )
        grid = _resolve_branch_grid(branch_dict, grid_by_name)
        branch_id = network.branch(
            model,
            from_node_id=branch_dict["from_node"],
            to_node_id=branch_dict["to_node"],
            grid=grid,
            name=branch_dict.get("name"),
        )
        branch = network.branch_by_id(branch_id)
        branch.active = branch_dict.get("active", True)

    return network


def _resolve_branch_grid(branch_dict, grid_by_name):
    """Return the grid argument for re-creating a branch.

    Single-grid branches get their concrete grid. Multi-grid (coupling) branches
    return ``None`` so :meth:`Network.branch` rebuilds the grid-type mapping from
    the endpoint node grids.
    """
    grid_ids = branch_dict.get("grid_ids")
    if grid_ids is None:
        grid_id = branch_dict.get("grid_id")
        grid_ids = [grid_id] if grid_id is not None else []
    distinct = [grid_by_name[g] for g in dict.fromkeys(grid_ids)]
    if len(distinct) == 1:
        return distinct[0]
    return None


def load_to_network(file) -> Network:
    with open(file, encoding="utf-8") as read_fp:
        dict_struct = json.load(read_fp)
    return native_dict_to_network(dict_struct)


# --------------------------------------------------------------------------- #
# Serialization (Network -> dict)                                              #
# --------------------------------------------------------------------------- #
def iter_concrete_grids(grid):
    """Yield the concrete grid object(s) backing a component.

    Single-grid components carry their grid directly.  Multi-grid branches
    (coupling points that span e.g. a power and a gas grid) carry a ``dict``
    mapping grid type -> grid (and sometimes a ``list`` of grids).  Yield every
    distinct concrete grid that exposes a ``name``, in a stable order.
    """
    if isinstance(grid, dict):
        seen = set()
        for value in grid.values():
            candidates = value if isinstance(value, (list, tuple)) else [value]
            for candidate in candidates:
                if (
                    getattr(candidate, "name", None) is not None
                    and id(candidate) not in seen
                ):
                    seen.add(id(candidate))
                    yield candidate
    elif getattr(grid, "name", None) is not None:
        yield grid


def fetch_grid_to_dict(grid_dict, grid_from_model):
    for grid in iter_concrete_grids(grid_from_model):
        if grid.name not in grid_dict:
            grid_dict[grid.name] = grid
        elif grid_dict[grid.name] is not grid:
            raise PersistenceException(
                f"You must not define multiple grids with the same name: {grid.name}"
            )


def model_to_dict(model):
    """Encode a model's full state.

    All public (solver-visible) attributes are serialized; private attributes
    are serialized only when JSON-encodable, since non-encodable ones hold object
    references that are rebuilt by ``create()`` / ``init()`` on load. A public
    attribute that cannot be encoded is a genuine gap and raises.
    """
    result = {}
    for key, value in model.__dict__.items():
        if _encodable(value):
            result[key] = _encode_value(value)
        elif not key.startswith("_"):
            raise PersistenceException(
                f"Cannot serialize public attribute {key!r} of "
                f"{type(model).__name__}: unsupported type {type(value).__name__}."
            )
    return result


def child_to_dict(child):
    return dict(
        id=child.id,
        active=child.active,
        name=child.name,
        values=model_to_dict(child.model),
        model_type=type(child.model).__name__,
    )


def compound_to_dict(compound):
    return dict(
        id=compound.id,
        active=compound.active,
        name=compound.name,
        values=model_to_dict(compound.model),
        model_type=type(compound.model).__name__,
        connected_to=compound.connected_to,
    )


def branch_to_dict(branch, grids):
    fetch_grid_to_dict(grids, branch.grid)
    grid_ids = [grid.name for grid in iter_concrete_grids(branch.grid)]
    return dict(
        id=branch.id,
        from_node=branch.id[0],
        to_node=branch.id[1],
        # grid_id keeps a single representative for backward compatibility;
        # grid_ids preserves the full (possibly multi-grid) mapping.
        grid_id=grid_ids[0] if grid_ids else None,
        grid_ids=grid_ids,
        active=branch.active,
        name=branch.name,
        values=model_to_dict(branch.model),
        model_type=type(branch.model).__name__,
    )


def node_to_dict(node, grids):
    fetch_grid_to_dict(grids, node.grid)
    return dict(
        id=node.id,
        grid_id=node.grid.name,
        child_ids=node.child_ids,
        position=node.position,
        active=node.active,
        name=node.name,
        values=model_to_dict(node.model),
        model_type=type(node.model).__name__,
    )


def network_to_native_dict(network: Network) -> dict:
    """Serialize *network* to the native, JSON-friendly ``dict`` form."""
    grids = {}
    node_dict_list = [
        node_to_dict(node, grids)
        for node in network.nodes
        if not network.is_blacklisted(node)
    ]
    branch_dict_list = [
        branch_to_dict(branch, grids)
        for branch in network.branches
        if not network.is_blacklisted(branch)
    ]
    child_dict_list = [
        child_to_dict(child)
        for child in network.childs
        if not network.is_blacklisted(child)
    ]
    compound_dict_list = [
        compound_to_dict(compound)
        for compound in network.compounds
        if not network.is_blacklisted(compound)
    ]
    return dict(
        version=FORMAT_VERSION,
        grids={
            k: {"values": _encode_values(vars(v)), "model_type": type(v).__name__}
            for k, v in grids.items()
        },
        nodes=node_dict_list,
        childs=child_dict_list,
        branches=branch_dict_list,
        compounds=compound_dict_list,
    )


def write_omef_network(file, network: Network):
    """Serialize *network* to *file* as native JSON."""
    to_serialize = network_to_native_dict(network)
    with open(file, "w", encoding="utf-8") as write_fp:
        json.dump(to_serialize, write_fp, indent=3, default=_json_default)
