import json
from monee.model.core import Network, Var, component_list
import inspect


class PersistenceException(Exception):
    pass


def init_model(model_type, preprocessed_dict):
    model = None
    model_type_dict = {
        component_cls.__name__: component_cls for component_cls in component_list
    }
    if model_type in model_type_dict:
        model_cls = model_type_dict[model_type]
        # model = model_cls.__new__(model_cls)
        model = model_cls(
            **{
                argname: 0
                for argname, _ in list(
                    inspect.signature(model_cls.__init__).parameters.items()
                )[1:]
            }
        )
        for key, value in preprocessed_dict.items():
            setattr(model, key, value)
    else:
        raise PersistenceException(
            f"The type {model_type} is not known! Maybe you forgot to decorate your model class with @model?"
        )
    return model


def preprocess_dict(model_dict):
    result = {}
    for k, v in model_dict.items():
        if type(v) == dict:
            if "max" in v and "min" in v and "value" in v:
                result[k] = Var(v["value"], v["max"], v["min"])
        else:
            result[k] = v
    return result


def native_dict_to_network(dict_struct) -> Network:
    network = Network(None)

    grid_by_name = dict_struct["grids"]
    for k, v in grid_by_name.items():
        values_grid_dict = v["values"]
        preprocessed_dict = preprocess_dict(values_grid_dict)
        model = init_model(v["model_type"], values_grid_dict)
        grid_by_name[k] = model
        if network.default_grid_model is None:
            network.default_grid_model = model

    childs = dict_struct["childs"]
    nodes = dict_struct["nodes"]
    branches = dict_struct["branches"]
    for child_dict in childs:
        values_child_dict = child_dict["values"]
        preprocessed_dict = preprocess_dict(values_child_dict)
        model = init_model(child_dict["model_type"], preprocessed_dict)
        network.child(
            model,
            overwrite_id=child_dict["id"],
        )
    for node_dict in nodes:
        values_node_dict = node_dict["values"]
        preprocessed_dict = preprocess_dict(values_node_dict)
        model = init_model(node_dict["model_type"], preprocessed_dict)
        network.node(
            model,
            child_ids=node_dict["child_ids"],
            grid=grid_by_name[node_dict["grid_id"]],
            overwrite_id=node_dict["id"],
        )
    for branch_dict in branches:
        values_branch_dict = branch_dict["values"]
        preprocessed_dict = preprocess_dict(values_branch_dict)
        model = init_model(branch_dict["model_type"], preprocessed_dict)
        network.branch(
            model,
            from_node_id=branch_dict["from_node"],
            to_node_id=branch_dict["to_node"],
            grid=grid_by_name[branch_dict["grid_id"]],
        )

    return network


def load_to_network(file) -> Network:
    dict_struct = None
    with open(file, "r") as read_fp:
        dict_struct = json.load(read_fp)

    return native_dict_to_network(dict_struct)


def write_omef_network(file, network: Network):
    grids = {}
    nodes = network.nodes
    branches = network.branches
    childs = network.childs
    compounds = network.compounds

    node_dict_list = []
    branch_dict_list = []
    child_dict_list = []
    compound_dict_list = []
    for node in nodes:
        if not network.is_blacklisted(node):
            node_dict_list.append(node_to_dict(node, grids))
    for branch in branches:
        if not network.is_blacklisted(branch):
            branch_dict_list.append(branch_to_dict(branch, grids))
    for child in childs:
        if not network.is_blacklisted(child):
            child_dict_list.append(child_to_dict(child))
    for compound in compounds:
        compound_dict_list.append(compound_to_dict(compound))

    to_serialize = dict(
        grids={
            k: {"values": v.__dict__, "model_type": type(v).__name__}
            for (k, v) in grids.items()
        },
        nodes=node_dict_list,
        childs=child_dict_list,
        branches=branch_dict_list,
    )
    with open(file, "w") as write_fp:
        json.dump(to_serialize, write_fp, indent=3, default=vars)


def child_to_dict(child):
    return dict(
        id=child.id,
        values=model_to_dict(child.model),
        model_type=type(child.model).__name__,
    )


def compound_to_dict(compound):
    return dict(
        id=compound.id,
        values=model_to_dict(compound.model),
        model_type=type(compound.model).__name__,
        connected_to=compound.node_name_to_id_dict,
    )


def fetch_grid_to_dict(grid_dict, grid_from_model):
    if not grid_from_model.name in grid_dict:
        grid_dict[grid_from_model.name] = grid_from_model
    elif grid_dict[grid_from_model.name] is not grid_from_model:
        raise Exception(
            f"You must not define multiple grids with the same name: {grid_from_model.name}"
        )


def branch_to_dict(branch, grids):
    fetch_grid_to_dict(grids, branch.grid)

    return dict(
        id=branch.id,
        from_node=branch.id[0],
        to_node=branch.id[1],
        grid_id=branch.grid.name,
        values=model_to_dict(branch.model),
        model_type=type(branch.model).__name__,
    )


def node_to_dict(node, grids):
    fetch_grid_to_dict(grids, node.grid)

    return dict(
        id=node.id,
        grid_id=node.grid.name,
        child_ids=node.child_ids,
        values=model_to_dict(node.model),
        model_type=type(node.model).__name__,
    )


def model_to_dict(model):
    base_dict = model.vars
    result = dict(base_dict)
    return result
