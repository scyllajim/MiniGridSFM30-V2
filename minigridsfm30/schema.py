# MiniGridSFM30-v2 schema
# GridSFM-style schema for pandapower case30.

# ----------------------------
# Node types
# ----------------------------
NODE_TYPES = [
    "bus",
    "generator",
    "load",
    "shunt",
    "branch_ac",
    "branch_tr",
    "cycle",
]

# ----------------------------
# Edge types
# ----------------------------
EDGE_TYPES = [
    ("bus", "endpoint_of", "branch_ac"),
    ("branch_ac", "endpoint_of", "bus"),

    ("bus", "endpoint_of", "branch_tr"),
    ("branch_tr", "endpoint_of", "bus"),

    ("cycle", "in_cycle", "branch_ac"),
    ("branch_ac", "in_cycle", "cycle"),

    ("cycle", "in_cycle", "branch_tr"),
    ("branch_tr", "in_cycle", "cycle"),

    ("generator", "generator_link", "bus"),
    ("bus", "generator_link", "generator"),

    ("load", "load_link", "bus"),
    ("bus", "load_link", "load"),

    ("shunt", "shunt_link", "bus"),
    ("bus", "shunt_link", "shunt"),
]

# ----------------------------
# Bus feature schema
# Official GridSFM base bus schema is close to:
# [base_kV, type, Vmin, Vmax]
#
# In v2, we add Pd/Qd aggregate as useful 30-bus features.
# ----------------------------
BUS_FEATURES = [
    "base_kv",
    "type",
    "min_vm_pu",
    "max_vm_pu",
    "pd_pu",
    "qd_pu",
    "is_slack",
]

# bus.y = [theta_rad, vm_pu]
# Important: official GridSFM uses bus.pred = [theta, V]
BUS_TARGETS = [
    "theta_rad",
    "vm_pu",
]

# ----------------------------
# Generator feature schema
# Similar to official GridSFM generator schema:
# [mbase, _, Pmin, Pmax, _, Qmin, Qmax, Vg, cp2, cp1, cp0]
# ----------------------------
GENERATOR_FEATURES = [
    "mbase",
    "unused_1",
    "pmin_pu",
    "pmax_pu",
    "unused_2",
    "qmin_pu",
    "qmax_pu",
    "vg_pu",
    "cp2",
    "cp1",
    "cp0",
    "is_ext_grid",
]

GENERATOR_TARGETS = [
    "pg_pu",
    "qg_pu",
]

# ----------------------------
# Load feature schema
# Official load feature is essentially [Pd, Qd].
# ----------------------------
LOAD_FEATURES = [
    "pd_pu",
    "qd_pu",
]

# ----------------------------
# Shunt feature schema
# ----------------------------
SHUNT_FEATURES = [
    "bs_pu",
    "gs_pu",
]

# ----------------------------
# Branch AC feature schema
# Similar to official ac_line edge_attr:
# [angmin, angmax, b_fr, b_to, r, x, rate_a, ...]
# Here branch_ac is a node, so we store these on branch_ac.x.
# ----------------------------
BRANCH_AC_FEATURES = [
    "angmin_rad",
    "angmax_rad",
    "b_fr_pu",
    "b_to_pu",
    "r_pu",
    "x_pu",
    "rate_a_pu",
]

# branch_ac.y = [p_from_pu, q_from_pu, p_to_pu, q_to_pu]
BRANCH_AC_TARGETS = [
    "p_from_pu",
    "q_from_pu",
    "p_to_pu",
    "q_to_pu",
]

SN_BASE_MVA = 100.0


def print_schema():
    print("=== MiniGridSFM30-v2 Schema ===")
    print("NODE_TYPES:", NODE_TYPES)
    print("EDGE_TYPES:")
    for et in EDGE_TYPES:
        print(" ", et)
    print("BUS_FEATURES:", BUS_FEATURES)
    print("BUS_TARGETS:", BUS_TARGETS)
    print("GENERATOR_FEATURES:", GENERATOR_FEATURES)
    print("GENERATOR_TARGETS:", GENERATOR_TARGETS)
    print("LOAD_FEATURES:", LOAD_FEATURES)
    print("SHUNT_FEATURES:", SHUNT_FEATURES)
    print("BRANCH_AC_FEATURES:", BRANCH_AC_FEATURES)
    print("BRANCH_AC_TARGETS:", BRANCH_AC_TARGETS)
    print("SN_BASE_MVA:", SN_BASE_MVA)


if __name__ == "__main__":
    print_schema()
