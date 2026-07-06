# MiniGridSFM30-v2

GridSFM-style AC-OPF surrogate for pandapower case30.

Current progress:
- Project environment initialized.
- GridSFM-style schema created.
- pandapower case30 converted to HeteroData.
- Line is represented as branch_ac node.
- bus <-> branch_ac endpoint graph is constructed.
- generator/load/shunt/branch_tr/cycle node types are reserved.

Current graph shape:
- bus: 30 nodes
- generator: 6 nodes = 5 regular generators + 1 ext_grid
- load: 20 nodes
- branch_ac: 41 nodes
- endpoint edges: 82
