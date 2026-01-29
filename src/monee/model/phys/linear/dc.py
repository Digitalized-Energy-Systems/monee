# dc_power_flow_islanding.py
#
# Islanding-safe DC switching model via a *single-commodity connectivity flow*
# from a super-source (node 0) to all energized buses.
#
# Assumptions:
# - You have binaries:
#     e_i ∈ {0,1}    bus energized
#     y_ij ∈ {0,1}   line (i,j) closed
#     g_s ∈ {0,1}    grid-forming source / blackstart / substation enabled (at bus s)
# - You add a super-source node "0" (not a real bus).
# - You have directed connectivity-flow variables:
#     c_ij ≥ 0 for each directed arc i→j corresponding to a physical line
#     c_0s ≥ 0 for each arc 0→s to each candidate source bus s
#
# Each function below returns exactly ONE equation/constraint.


# --- Slack / reference handling (safe for multiple islands) ---


def source_reference_angle(theta_s):
    """
    Fix each candidate grid-forming source bus angle.
    Safe with islanding: each energized island must contain at least one such source.
    """
    return theta_s == 0


def angle_upper_bound_energized(theta_i, theta_max, e_i):
    """
    Angle bound active only when bus is energized:
        theta_i <= theta_max * e_i
    """
    return theta_i <= theta_max * e_i


def angle_lower_bound_energized(theta_i, theta_max, e_i):
    """
    Angle bound active only when bus is energized:
        theta_i >= -theta_max * e_i
    """
    return theta_i >= -theta_max * e_i


def source_implies_bus_energized(e_s, g_s):
    """
    If a grid-forming source is enabled, the bus must be energized:
        e_s >= g_s
    """
    return e_s >= g_s


def bus_must_have_source_if_energized(e_i, g_i):
    """
    Optional strengthening (if you maintain g_i for every bus, with g_i=0 for non-sources):
        e_i <= g_i + (something that connects it)
    Often you will instead rely on the connectivity-flow constraints below.
    This constraint alone just says an energized bus could be a source.
    """
    return e_i <= g_i + (
        1 - g_i
    )  # tautology placeholder; usually omit in favor of flow connectivity


# --- Connectivity (single-commodity flow from super-source 0) ---


def connectivity_demand_balance(bus_inflow, bus_outflow, e_i):
    """
    For each real bus i (excluding super-source 0):
        sum_in c_ji - sum_out c_ij == e_i
    Meaning: each energized bus must receive 1 unit of connectivity flow.
    For de-energized bus (e_i=0), net inflow is 0.
    """
    return bus_inflow - bus_outflow == e_i


def connectivity_super_source_supply(super_outflow, total_energized_buses):
    """
    At the super-source node 0:
        sum_s c_0s == sum_i e_i
    Meaning: the super-source injects exactly as much connectivity flow
    as the total number of energized buses.
    """
    return super_outflow == total_energized_buses


def connectivity_arc_capacity_line(c_ij, y_ij, big_m_conn):
    """
    Capacity on a physical directed arc i→j:
        c_ij <= M_conn * y_ij
    where M_conn is typically (N-1) or N (number of buses).
    """
    return c_ij <= big_m_conn * y_ij


def connectivity_arc_capacity_source(c_0s, g_s, big_m_conn):
    """
    Capacity from super-source 0 to a candidate source bus s:
        c_0s <= M_conn * g_s
    If g_s=0, the source can't supply connectivity flow.
    """
    return c_0s <= big_m_conn * g_s


def connectivity_nonnegativity(c_ij):
    """
    Nonnegativity of connectivity flow on any arc:
        c_ij >= 0
    """
    return c_ij >= 0


def total_energized_buses_definition(total_energized_buses, e_vars):
    """
    If your modeling layer prefers an explicit equality for totals:
        total_energized_buses == sum(e_i)
    """
    return total_energized_buses == sum(e_vars)


def super_outflow_definition(super_outflow, c_0s_vars):
    """
    Explicit definition of super-source outflow:
        super_outflow == sum_s c_0s
    """
    return super_outflow == sum(c_0s_vars)


# --- Switchable DC flow equations (one constraint per function) ---


def line_flow_angle_relation_upper(f_ij, theta_i, theta_j, b_ij, y_ij, big_m_pf):
    """
    Switchable DC flow (upper big-M):
        f_ij - b_ij*(theta_i - theta_j) <= M_pf*(1 - y_ij)
    """
    return f_ij - b_ij * (theta_i - theta_j) <= big_m_pf * (1 - y_ij)


def line_flow_angle_relation_lower(f_ij, theta_i, theta_j, b_ij, y_ij, big_m_pf):
    """
    Switchable DC flow (lower big-M):
        f_ij - b_ij*(theta_i - theta_j) >= -M_pf*(1 - y_ij)
    """
    return f_ij - b_ij * (theta_i - theta_j) >= -big_m_pf * (1 - y_ij)


def line_capacity_upper(f_ij, f_max_ij, y_ij):
    """
    Switchable thermal limit (upper):
        f_ij <= Fmax * y_ij
    """
    return f_ij <= f_max_ij * y_ij


def line_capacity_lower(f_ij, f_max_ij, y_ij):
    """
    Switchable thermal limit (lower):
        f_ij >= -Fmax * y_ij
    """
    return f_ij >= -f_max_ij * y_ij


def nodal_active_power_balance(p_gen_i, p_load_served_i, flow_out_sum, flow_in_sum):
    """
    Active power balance at bus i:
        Pgen_i - Pload_i == sum_out F - sum_in F
    """
    return p_gen_i - p_load_served_i == flow_out_sum - flow_in_sum


# --- Handy utilities (pure Python, not constraints) ---


def big_m_pf_from_angle_bounds(b_ij, theta_max):
    """
    Tight-ish big-M for power-flow equation if angles are bounded:
        |b(θi-θj)| <= 2*b*θmax  =>  M_pf = 2*b*θmax
    """
    return 2 * b_ij * theta_max


def big_m_conn_from_bus_count(n_buses):
    """
    Typical choice for connectivity flow big-M:
        M_conn = N-1  (or N)
    """
    return max(1, n_buses - 1)
