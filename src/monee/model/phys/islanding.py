def source_reference_angle(theta_s):
    """Fix the voltage angle at a grid-forming bus: theta_s == 0."""
    return theta_s == 0


def angle_upper_bound_energized(theta_i, theta_max, e_i):
    """theta_i <= theta_max * e_i  (zero angle when de-energised)."""
    return theta_i <= theta_max * e_i


def angle_lower_bound_energized(theta_i, theta_max, e_i):
    """theta_i >= -theta_max * e_i  (zero angle when de-energised)."""
    return theta_i >= -theta_max * e_i


def connectivity_demand_balance(bus_inflow, bus_outflow, e_i):
    """
    Per-bus balance: inflow - outflow == e_i.
    Each energised bus receives exactly 1 unit of connectivity flow;
    de-energised buses receive 0.
    """
    return bus_inflow - bus_outflow == e_i


def connectivity_super_source_supply(super_outflow, total_energized_buses):
    """
    Super-source supply equals the total number of energised buses:
    sum_s c_0s == sum_i e_i.
    """
    return super_outflow == total_energized_buses


def connectivity_arc_capacity_line(c_ij, y_ij, big_m_conn):
    """c_ij <= M_conn * y_ij  (no flow on open/off branches)."""
    return c_ij <= big_m_conn * y_ij


def connectivity_arc_capacity_source(c_0s, g_s, big_m_conn):
    """c_0s <= M_conn * g_s  (no flow from disabled grid-forming sources)."""
    return c_0s <= big_m_conn * g_s
