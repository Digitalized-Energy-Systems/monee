def _get(obj, attr, default=None):
    """Safe attribute read that also turns empty references into *default*."""
    value = getattr(obj, attr, default)
    return default if value in ("", None) else value


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _class_name(obj) -> str:
    return type(obj).__name__


def _resolve_two_endpoints(endpoints, resolve, report, count_msg, unresolved_msg):
    """Validate a 2-endpoint branch and resolve both nodes.

    Returns ``(node_a, node_b)`` on success. Calls ``report.skip`` with the
    appropriate message and returns ``None`` when there are not exactly two
    *endpoints* or either endpoint does not resolve to a node.
    """
    if len(endpoints) != 2:
        report.skip(count_msg)
        return None
    node_a = resolve(endpoints[0])
    node_b = resolve(endpoints[1])
    if node_a is None or node_b is None:
        report.skip(unresolved_msg)
        return None
    return node_a, node_b


def _resolve_single_node(endpoints, resolve, report, unresolved_msg):
    """Resolve the first endpoint to a node.

    Returns the node id, or ``None`` (after calling ``report.skip`` with
    *unresolved_msg*) when there is no endpoint or it does not resolve.
    """
    node = resolve(endpoints[0]) if endpoints else None
    if node is None:
        report.skip(unresolved_msg)
        return None
    return node
