"""Tests for pin.py."""
import os
from pathlib import Path

import pandas as pd
import pytest
from pyprojroot import here

from .pin import (
    compute_chain_pos_aa_mapping,
    compute_distmat,
    get_interacting_atoms,
    get_ring_atoms,
    get_ring_centroids,
    pdb2df,
    read_pdb,
)
from .resi_atoms import (
    AROMATIC_RESIS,
    BOND_TYPES,
    CATION_RESIS,
    HYDROPHOBIC_RESIS,
    NEG_AA,
    PI_RESIS,
    POS_AA,
    RESI_NAMES,
    SULPHUR_RESIS,
)

data_path = Path(__file__).parent / "test_data" / "2VIU.pdb"


def generate_network():
    """Generate PDB network.

    This is a helper function.
    """
    return read_pdb(data_path)


@pytest.fixture(scope="module")
def net():
    """Generate proteingraph from 2VUI.pdb."""
    return generate_network()


@pytest.fixture()
def pdb_df():
    """Generate pdb_df from 2VIU.pdb."""
    return pdb2df(data_path)


def test_bond_types_are_correct(net):
    """Test that the bonds that are specified are correct."""
    # Check that the bonds are correctly
    for _, _, d in net.edges(data=True):
        assert isinstance(d["kind"], list)
        for kind in d["kind"]:
            assert kind in BOND_TYPES


def test_nodes_are_strings(net):
    """
    Checks to make sure that the nodes are a string.

    For expediency, checks only 1/4 of the nodes.
    """
    for n in net.nodes():
        assert isinstance(n, str)


@pytest.mark.skip(reason="Deprecated.")
def test_parse_pdb(net):
    """Test the function parse_pdb."""
    # Asserts that the number of lines in the dataframe is correct.
    assert len(net.dataframe) == 3892, "Error: Function or data has changed!"

    # Asserts that the following columns are all present.
    column_types = {
        "record_name": str,
        "atom_name": str,
        "residue_name": str,
        "chain_id": str,
        "residue_number": int,
        "x_coord": float,
        "y_coord": float,
        "z_coord": float,
        "node_id": str,
    }
    for c in column_types.keys():
        assert (
            c in net.dataframe.columns
        ), "{0} not present in DataFrame columns!".format(c)


def test_compute_distmat(net):
    """Test the function compute_distmat, using dummy data."""
    data = list()
    for i in range(1, 2):
        d = dict()
        d["idx"] = i
        d["x_coord"] = i
        d["y_coord"] = i
        d["z_coord"] = i
        data.append(d)
    df = pd.DataFrame(data)
    distmat = compute_distmat(df)

    # Asserts that the shape is correct.
    assert distmat.shape == (len(data), len(data))


def test_no_self_loops(net):
    """Test that there are no self loops amongst residues."""
    for n in net.nodes():
        assert not net.has_edge(n, n)


def test_get_interacting_atoms(pdb_df):
    """Test the function get_interacting_atoms_."""
    distmat = compute_distmat(pdb_df)
    interacting = get_interacting_atoms(6, distmat)
    # Asserts that the number of interactions found at 6A for 2VIU.
    assert len(interacting[0]) == 156408


def test_add_hydrophobic_interactions(net):
    """Test the function add_hydrophobic_interactions_."""
    resis = get_edges_by_bond_type(net, "hydrophobic")
    for (r1, r2) in resis:
        assert net.nodes[r1]["residue_name"] in HYDROPHOBIC_RESIS
        assert net.nodes[r2]["residue_name"] in HYDROPHOBIC_RESIS


def test_add_disulfide_interactions(net):
    """Test the function add_disulfide_interactions_."""
    resis = get_edges_by_bond_type(net, "disulfide")

    for (r1, r2) in resis:
        assert net.nodes[r1]["residue_name"] == "CYS"
        assert net.nodes[r2]["residue_name"] == "CYS"


@pytest.mark.skip(reason="Not yet implemented.")
def test_delaunay_triangulation(net):
    """
    Test delaunay triangulation.

    I am including this test here that always passes because I don't know how
    best to test it. The code in pin.py uses scipy's delaunay triangulation.
    """
    pass


# 10 March 2016
# This test has been passed out until I figure out what the exact criteria
# for hydrogen bonding is. I intuitively don't think it should be merely 3.5A
# between any two of N, O, S atoms, regardless of whether they are the same
# element or not. Rather, it should be O:->N or N:-->O, or something like that.
@pytest.mark.skip(reason="Not yet implemented.")
def test_add_hydrogen_bond_interactions(net):
    """Tests the function add_hydrogen_bond_interactions_."""
    # net.add_hydrogen_bond_interactions_()
    # resis = net.get_edges_by_bond_type('hbond')
    # assert len(resis) == 86
    pass


from proteingraph.pin import get_edges_by_bond_type


def test_add_aromatic_interactions(net):
    """
    Tests the function add_aromatic_interactions_.

    The test checks that each residue in an aromatic interaction
    is one of the aromatic residues.
    """
    resis = get_edges_by_bond_type(net, "aromatic")
    for n1, n2 in resis:
        assert net.nodes[n1]["residue_name"] in AROMATIC_RESIS
        assert net.nodes[n2]["residue_name"] in AROMATIC_RESIS


def test_add_aromatic_sulphur_interactions(net):
    """Tests the function add_aromatic_sulphur_interactions_."""
    resis = get_edges_by_bond_type(net, "aromatic_sulphur")
    for n1, n2 in resis:
        condition1 = (
            net.nodes[n1]["residue_name"] in SULPHUR_RESIS
            and net.nodes[n2]["residue_name"] in AROMATIC_RESIS
        )

        condition2 = (
            net.nodes[n2]["residue_name"] in SULPHUR_RESIS
            and net.nodes[n1]["residue_name"] in AROMATIC_RESIS
        )

        assert condition1 or condition2


def test_add_cation_pi_interactions(net):
    """Tests the function add_cation_pi_interactions."""
    resis = get_edges_by_bond_type(net, "cation_pi")
    for n1, n2 in resis:
        resi1 = net.nodes[n1]["residue_name"]
        resi2 = net.nodes[n2]["residue_name"]

        condition1 = resi1 in CATION_RESIS and resi2 in PI_RESIS
        condition2 = resi2 in CATION_RESIS and resi1 in PI_RESIS

        assert condition1 or condition2


def test_add_ionic_interactions(net):
    """
    Tests the function add_ionic_interactions_.

    This test checks that residues involved in ionic interactions
    are indeed oppositely-charged.

    Another test is needed to make sure that ionic interactions
    are not missed.
    """
    resis = get_edges_by_bond_type(net, "ionic")
    for n1, n2 in resis:
        resi1 = net.nodes[n1]["residue_name"]
        resi2 = net.nodes[n2]["residue_name"]

        condition1 = resi1 in POS_AA and resi2 in NEG_AA
        condition2 = resi2 in POS_AA and resi1 in NEG_AA

        assert condition1 or condition2


@pytest.mark.skip(reason="Not yet implemented.")
def test_add_ionic_interactions_example():
    """
    Example-based test.

    Check on HIV protease that "B8ARG", "A29ASP" contains both ionic
    and hbond interactions.
    """
    pass


@pytest.mark.skip(reason="Feature array is deprecated.")
def test_feature_array(net):
    """Test the function feature_array."""
    with pytest.raises(AssertionError):
        net.feature_array("atom_name")

    node_features = net.feature_array(kind="node")
    assert len(node_features) == len(net.nodes())

    edge_features = net.feature_array(kind="edge")
    assert len(edge_features) == len(net.edges())


# def test_get_ring_atoms():
#     """
#     Tests the function get_ring_atoms.
#     """
#     ring_atom_TRP = net.get_ring_atoms(net.dataframe, 'TRP')
#     assert len(ring_atom_TRP) == 63
#     ring_atom_HIS = net.get_ring_atoms(net.dataframe, 'HIS')
#     assert len(ring_atom_HIS) == 55


def test_get_ring_centroids(pdb_df):
    """Test the function get_ring_centroids."""
    ring_atom_TYR = get_ring_atoms(pdb_df, "TYR")
    assert len(ring_atom_TYR) == 32
    centroid_TYR = get_ring_centroids(ring_atom_TYR)
    assert len(centroid_TYR) == 16

    ring_atom_PHE = get_ring_atoms(pdb_df, "PHE")
    assert len(ring_atom_PHE) == 36
    centroid_PHE = get_ring_centroids(ring_atom_PHE)
    assert len(centroid_PHE) == 18


node_pairs = []
pdb_dataframe = pdb2df(data_path)
chain_pos_aa = compute_chain_pos_aa_mapping(pdb_dataframe)
for chain, pos_aa in chain_pos_aa.items():
    for pos, aa in pos_aa.items():
        try:
            node1 = f"{chain}{pos}{aa}"
            aa2 = pos_aa[pos + 1]
            node2 = f"{chain}{pos+1}{aa2}"
            if aa in RESI_NAMES and aa2 in RESI_NAMES:
                node_pairs.append((node1, node2))
        except KeyError:
            pass


@pytest.mark.parametrize("node_pair", node_pairs)
def test_backbone_neighbor_connectivity(net, node_pair):
    """Test to ensure that backbone connectivity has been entered correctly."""
    node1, node2 = node_pair
    assert net.has_edge(node1, node2)
