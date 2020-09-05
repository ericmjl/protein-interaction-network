# protein-graph

Computes a molecular graph for protein structures.

## why?

Proteins fold into 3D structures, and have a natural graph representation: amino acids are nodes, and biochemical interactions are edges.

I wrote this package as part of a larger effort to do graph convolutional neural networks on protein structures (represented as graphs). However, that's not the only thing I can foresee doing with this.

One may be interested in the topology of proteins across species and over evolutionary time. This package can aid in answering this question.

## how do I install this package?

Currently only `pip`-installable:

```bash
$ pip install proteingraph
```

## how do I use this package?

This package assumes that you have a standard protein structure file
(e.g. a PDB file).
This may be a file generated after solving the NMR
or crystal structure of a protein,
or it may be generated from homology modelling.

Once that has been generated, the molecular graph can be generated using Python code.

```python
from proteingraph import read_pdb

p = read_pdb('my_model.pdb')
```

The object that is returned is a NetworkX Graph object,
which means all of the graph theoretic methods in there are available.

### converting graphs to tensors

To convert the graph into tensors for use as inputs to graph neural networks,
there are three functions provided.

Here's how they are used,
starting first with converting node metadata to matrices:

```python
from proteingraph.conversion import (
    generate_feature_dataframe,
    format_adjacency,
    generate_adjacency_tensor
)

# You provide a collection of functions
# that take in the node name and metadata dictionary,
# and return a pandas Series:
def my_func(n, d):
    return pd.Series({"key_name": d["key_name"]}, name=n)

def my_func2(n, d):
    return pd.Series(..., name=n)

def myfunc3(n, d):
    return pd.Series(..., name=n)

# If you have a function that depends on outside information,
# be sure to scope the variables correctly
# or use functools.partial to help:
from functools import partial

@partial(argname=some_variable)
def myfunc4(n, d, argname):
    return pd.Series(..., name=n)

# seriously though, give the functions more informative names!

funcs = [
    my_func,
    my_func2,
    my_func3,
]

# Now get a pandas DataFrame version of the tensor
feats = generate_feature_dataframe(p, funcs=funcs)
# You can also return a NumPy array version:
F = generate_feature_dataframe(p, funcs=funcs, return_array=True)

# Same goes for adjacency matrices, or even adjacency tensors!
# To facilitate return as XArray DataArrays (for inspection),
# we provide a `format_adjacency` function.
def my_adj_func(G):
    adj = some_func(G)
    return format_adjacency(G, adj, "xarray_coord_name")

def my_adj_func2(G):
    adj = some_func2(G)
    return format_adjacency(G, adj, "another_xarray_coord_name")

funcs = [
    my_adj_func,
    my_adj_func2,
]

# Now, generate the xarray adjacency tensor
adj_da = generate_adjacency_tensor(G, funcs)
# You can also generate a NumPy array version:
A = generate_adjacency_tensor(G, funcs, return_array=True)
```
