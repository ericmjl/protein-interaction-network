"""Make a graph out of a PDB file and saves the graph as a NetworkX pickle."""

import networkx as nx

from proteingraph import read_pdb

p = read_pdb("hiv1_homology_model.pdb")
nx.write_gpickle(p, "hiv1_homology_model.pkl")
