from proteingraph.conversion import generate_feature_dataframe, format_adjacency, generate_adjacency_tensor
import networkx as nx
import math
import pytest
import pandas as pd
import numpy as np


#### NODE FEATURE GENERATION FUNCTION TESTS

@pytest.fixture
def dummy_graph():
    G = nx.Graph()
    nodes = list(range(10))
    metadata = [{"idx": math.sqrt(i)} for i in nodes]
    for n, d in zip(nodes, metadata):
        G.add_node(n, **d)

    G.add_edges_from(zip(nodes[:-1], nodes[1:]))
    return G


def test_generate_feature_dataframe_example(dummy_graph):
    """
    Example-based test for generating a feature dataframe from a graph
    """
    funcs = [
        lambda n, d: pd.Series({"idx": d["idx"]}, name=n)
    ]

    df = generate_feature_dataframe(dummy_graph, funcs)
    assert len(df) == len(dummy_graph)
    assert "idx" in df.columns
    assert len(df.columns) == 1


def test_generate_feature_dataframe_twofuncs(dummy_graph):
    """
    Example-based test that uses two functions.
    """
    funcs = [
        lambda n, d: pd.Series({"idx": d["idx"]}, name=n),
        lambda n, d: pd.Series({"idx_sq": d["idx"]**2}, name=n)
    ]

    df = generate_feature_dataframe(dummy_graph, funcs)
    assert len(df) == len(dummy_graph)
    assert set(["idx", "idx_sq"]) == set(df.columns)
    assert len(df.columns) == 2


def test_generate_feature_bad_name(dummy_graph):
    """
    Example-based test where series is not named.
    """
    funcs = [
        lambda n, d: pd.Series({"idx": d["idx"]}, name="random_name"),
    ]
    with pytest.raises(NameError):
        generate_feature_dataframe(dummy_graph, funcs)

def test_generate_feature_func_two_columns(dummy_graph):
    """
    Example-based test where one func generates two columns.
    """

    funcs = [
        lambda n, d: pd.Series(
            {
                "idx": d["idx"],
                "idx_sq": d["idx"]**2,
            },
            name=n
        )
    ]
    df = generate_feature_dataframe(dummy_graph, funcs)
    assert len(df) == len(dummy_graph)
    assert set(["idx", "idx_sq"]) == set(df.columns)
    assert len(df.columns) == 2


def test_generate_feature_return_array(dummy_graph):
    """Example-based test to test that `return_array=True` returns NumPy array."""
    funcs = [lambda n, d: pd.Series({"idx": d["idx"]}, name=n)]
    arr = generate_feature_dataframe(dummy_graph, funcs, return_array=True)
    assert isinstance(arr, np.ndarray)


#### ADJACENCY-LIKE FUNCTION TESTS


def adjacency_matrix(G):
    """
    Generate adjacency matrix.

    Intended to be a helper function for the tests
    to make test body easier to read.
    """
    adj = nx.adjacency_matrix(G).toarray()
    return format_adjacency(G, adj, "adjacency")


def laplacian_matrix(G):
    """
    Generate laplacian matrix.

    Intended to be a helper function for the tests
    to make test body easier to read.
    """
    adj = nx.laplacian_matrix(G).toarray()
    return format_adjacency(G, adj, "laplacian")


def test_generate_adjacency_tensor(dummy_graph):
    """
    Test that adjacency tensor is generated correctly.
    """
    funcs = [
        adjacency_matrix,
        laplacian_matrix
    ]
    da = generate_adjacency_tensor(dummy_graph, funcs)
    assert da.shape == (len(dummy_graph), len(dummy_graph), len(funcs))
    assert set(da.dims) == set(["n1", "n2", "name"])
    assert set(da.coords["n1"].data) == set(dummy_graph.nodes())
    assert set(da.coords["n2"].data) == set(dummy_graph.nodes())
    assert set(da.coords["name"].data) == set(["laplacian", "adjacency"])


def test_format_adjacency_bad_func(dummy_graph):
    """
    Test that ValueError is raised when we pass in a badly-shaped adjacency matrix.
    """
    adj = np.random.normal(size=(len(dummy_graph)-1, len(dummy_graph)+1))
    with pytest.raises(ValueError):
        format_adjacency(dummy_graph, adj, "dummy_name")


def test_generate_adjacency_tensor_return_array(dummy_graph):
    """
    Test that return_array returns a numpy array.
    """
    funcs = [
        adjacency_matrix,
        laplacian_matrix
    ]
    arr = generate_adjacency_tensor(dummy_graph, funcs, return_array=True)
    assert isinstance(arr, np.ndarray)
