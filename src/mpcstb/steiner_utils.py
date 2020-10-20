import numpy as np
import pandas as pd
import itertools
import time

import networkx as nx
from networkx.algorithms import approximation as apx

import random

import matplotlib.pyplot as plt

# my libraries
from mpcstb import landscape as ld


def make_random_terminals(box_size, Nterminals):
    '''Generates random terminals on a grid of given size'''

    terminals = np.random.randint(1, high=box_size, size=(Nterminals, 2))
    terminals = list(map(tuple, terminals))

    return terminals


def create_graph(box_size, terminals):
    ''' Initialize 2D grid graph with terminals added as node attributes '''

    # create square grid graph
    grid_1d = range(1, int(box_size))
    G = nx.grid_2d_graph(grid_1d, grid_1d)

    # set terminal as attribute
    attribute_dictonary = {}
    for point in terminals:
        attribute_dictonary.update({point: {'terminal': 1}})

    nx.set_node_attributes(G, attribute_dictonary)

    return G


def get_terminals(G):
    '''returns terminals searching in node attributes'''
    return [
        n for n, attr in G.nodes(data=True)
        if (not attr == {}) and attr['terminal'] == 1
    ]


def naive_steiner(G):
    ''' Computes Steiner tree using terminals from node attributes. If number of terminals = 1, outputs
    a graph with one node = terminal (this is not the default of the nx function, which returns an empty graph) '''

    G = G.copy()

    terminals = get_terminals(G)

    if len(terminals) >= 1:
        G_steiner = apx.steinertree.steiner_tree(G, terminals, weight='length')

    if len(terminals) == 1:
        G_steiner = nx.Graph()
        G_steiner.add_node(terminals[0], terminal=1)

    return G_steiner


def get_pos(g):
    '''returns dictionary with nodes physical location'''
    return dict((n, n) for n in g.nodes())


def plot_graph_on_grid(G, box_size, tree=None, fig_size=6):
    '''Plots graph (connected or not) with option to overlay tree'''

    # get terminals from attributes
    terminals = get_terminals(G)

    # axes
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # set limits
    plt.xlim(0, box_size)
    plt.ylim(0, box_size)

    # draw edges and nodes
    nx.draw_networkx_edges(G, get_pos(G))
    nx.draw_networkx_nodes(G, get_pos(G), nodelist=terminals, node_size=50)

    # tick parameters
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    if not tree == None:
        nx.draw(tree,
                pos=get_pos(tree),
                node_size=0,
                ax=ax,
                edge_color='r',
                width=2.5)

    plt.show()


def plot_many_graphs_on_grid(list_G, box_size, fig_size=6):

    # axes
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # set limits
    plt.xlim(0, box_size)
    plt.ylim(0, box_size)

    # tick parameters
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    colors = ["red", "blue", "green", "yellow", "purple", "orange"]

    for G in list_G:
        terminals = get_terminals(G)
        nx.draw_networkx_edges(G, get_pos(G), edge_color=random.choice(colors))
        nx.draw_networkx_nodes(G, get_pos(G), nodelist=terminals, node_size=50)

    plt.show()


def remove_edges_with_landscape(G, landscape, alpha):
    '''Remove edges from G using landscape(x,y) > alpha, and also delete nodes left with no edges attached to them '''

    #copy G
    G = G.copy()

    terminals = get_terminals(G)

    # delete edges where the landscape function > alpha
    for edge in G.edges:
        x, y = 0.5 * (np.array(edge[0]) + np.array(edge[1]))
        if landscape(x, y) > alpha:
            G.remove_edge(edge[0], edge[1])

    #delete nodes left with no edges, unless they are terminals
    for node in G.copy().nodes():
        if G.degree(node) == 0 and not node in terminals:
            G.remove_node(node)

    # delete nodes with degree 1; weird! would be nice to fix this
    #ends = [ x[0] for x in G.degree() if x[1] <= 1]
    #G.remove_nodes_from(ends)

    return G


def extract_proper_clusters(G):
    '''Get clusters of nodes/edges which have terminals on them'''
    clusters = [
        G.subgraph(c).copy()
        for c in nx.connected_components(G)
        if len(get_terminals(G.subgraph(c).copy()))
    ]
    return clusters


def graph_info(G, verbosity=True):
    '''count clusters, nodes, terminals and edges of G'''
    terminals = get_terminals(G)

    n_clusters = len(extract_proper_clusters(G))
    n_nodes, n_edges, n_terminals = len(G.nodes()), len(
        G.edges()), len(terminals)

    if verbosity == True:
        print('Number of clusters: {}; Number of nodes: {}; Number of edges: {}; Number of terminals: {}'\
              .format(n_clusters, n_nodes, n_edges, n_terminals) )

    return n_clusters, n_nodes, n_edges, n_terminals


def metric(clusters, scaling):

    list_n_terminals = np.array(
        list(map(lambda x: len(get_terminals(x)), clusters)))
    list_n_nodes = np.array([len(x) for x in clusters])

    #s1 = np.dot(list_n_nodes, list_n_terminals**scaling)
    s1 = sum((list_n_terminals - 1)**scaling)
    s2a = sum(
        np.array(
            [x * y for x, y in itertools.product(list_n_terminals, repeat=2)]))
    s2b = sum(list_n_terminals**2)
    s2 = 0.5 * (s2a - s2b)

    return (s1 + s2) / 1000


######################################### modules to trim with landscape


def trim_graph_landscape(G, box_size, condition, sigma, verbosity=False):
    '''Trims graph using the landscape function. We scan over a range of values of alpha, and stop when a condition
    over the trimmed graph properties is satisfied. Outputs a list of connected sub-graphs (clusters). '''
    # copy original graph
    G = G.copy()

    # compute properties of original graph
    if verbosity == True:
        print('Properties of the original graph: ')
    n_clusters_0, n_nodes_0, n_edges_0, n_terminals_0 = graph_info(G, verbosity)

    # get terminals from attributes
    terminals = get_terminals(G)

    #compute landscape function
    fLand = ld.find_landscape_V(box_size, terminals, sigma)

    # range of alpha's
    #alpha_list = np.linspace(0.2, 10.2, 51)
    alpha_list = np.linspace(0.01, 1.01, 51)

    #print(alpha_list)

    # scans a range of alpha
    for alpha in alpha_list:
        if verbosity == True:
            print('For alpha = ', alpha)

        #trim graph
        G_trim = remove_edges_with_landscape(G, fLand, alpha)

        # compute properties of trimmed graph
        n_clusters, n_nodes, n_edges, n_terminals = graph_info(
            G_trim, verbosity)

        clusters = extract_proper_clusters(G_trim)
        #print('metric = ', metric(clusters, 1.4))

        # stop if condition is satisfied
        if eval(condition):
            print('Trimming condition fulfilled!')
            aux = graph_info(G_trim, verbosity=True)
            break

    # extract clusters
    clusters = extract_proper_clusters(G_trim)

    #outputs list of clusters
    return clusters


######################################### modules to join the sub-steiners


def shortest_path_node_subgraph(G, node, g):
    '''
    Computes shortest distances between `node` and all terminals in subgraph `g`, and keeps
    the one with smallest length (brute force scanning). Outputs a path i.e. collection of nodes in a particular order.
    '''

    terminals = get_terminals(g)
    #terminals = g.nodes()

    old_path = nx.path_graph(100)
    best_path = old_path

    for terminal in terminals:
        new_path = nx.shortest_path(G, node, terminal, weight='length')

        if len(new_path) < len(best_path):
            best_path = new_path

        old_path = new_path

    return best_path


def join_two_trees(G, tree1, tree2):
    '''Joins two trees by brute force scanning. Outputs joint tree'''

    tree1 = tree1.copy()
    tree2 = tree2.copy()

    terminals_1 = get_terminals(tree1)

    old_bridge = nx.path_graph(100)
    best_bridge = old_bridge

    for node_1 in terminals_1:
        new_bridge = shortest_path_node_subgraph(G, node_1, tree2)

        if len(new_bridge) < len(best_bridge):
            best_bridge = new_bridge

        old_bridge = new_bridge

    nx.add_path(tree2, best_bridge)

    joint_graph = nx.compose_all([tree1, tree2])

    return joint_graph


def join_two_trees_v2(G, tree1, tree2):
    '''Joins two trees by brute force scanning. Outputs joint tree'''

    #tree1 = tree1.copy()
    tree2 = tree2.copy()

    terminals_1 = get_terminals(tree1)
    terminals_2 = get_terminals(tree2)

    all_pairs = list(itertools.product(terminals_1, terminals_2))

    def my_distance(pair):
        return nx.shortest_path(G, *pair, weight='length')

    bridge = sorted(map(my_distance, all_pairs), key=len)[0]

    nx.add_path(tree2, bridge)

    joint_graph = nx.compose_all([tree1, tree2])

    return joint_graph


def join_all_trees(G, partial_trees):
    ''' Joins all trees by sequentially joining individual sub-trees to largest tree. '''

    partial_trees = partial_trees.copy()

    # trees, largest first
    partial_trees.sort(key=len, reverse=True)

    big_tree = partial_trees[0]
    small_trees = partial_trees[1:]

    for tree in small_trees:
        big_tree = join_two_trees_v2(G, tree, big_tree)

    return big_tree


def improved_steiner(G, box_size, condition, sigma, verbosity=False):

    tic = time.time()
    clusters = trim_graph_landscape(G, box_size, condition, sigma, verbosity)
    toc = time.time()
    print('Time spent computing clusters: ', toc - tic)

    tic = time.time()
    partial_steiners = list(map(naive_steiner, clusters))
    toc = time.time()
    print('Time spent computing partial steiners: ', toc - tic)

    tic = time.time()
    full_steiner = join_all_trees(G, partial_steiners)
    toc = time.time()
    print('Time spent joining partial steiners: ', toc - tic)

    print('Length of Steiner tree {}'.format(len(full_steiner)))

    return full_steiner
