import numpy as np
import pandas as pd

import networkx as nx
from networkx.algorithms import approximation as apx

import random

import matplotlib.pyplot as plt

# my libraries
import landscape as ld


def make_random_terminals(box_size, Nterminals):
    
    '''Generates random terminals on a grid of given size'''
    
    terminals = np.random.randint(0, high = box_size, size=(Nterminals, 2)) 
    terminals = list(map(tuple, terminals))
    
    return terminals


def create_graph(box_size, terminals):
    
    ''' Initialize 2D grid graph with terminals added as node attributes '''
    
    # create square grid graph
    grid_1d = range(1,int(box_size))
    G = nx.grid_2d_graph(grid_1d, grid_1d) 

    # set terminal as attribute
    attribute_dictonary = {}
    for point in terminals:
        attribute_dictonary.update({point: {'terminal': 1}})
            
    nx.set_node_attributes(G, attribute_dictonary)
            
    return G

def get_terminals(G):
    
    '''returns terminals searching in node attributes'''    
    return [n for n, attr in G.nodes(data=True) if (not attr == {}) and attr['terminal'] == 1]

def naive_steiner(G):
    
    ''' Computes Steiner tree using terminals from node attributes. If number of terminals = 1, outputs
    a graph with one node = terminal (this is not the default of the nx function, which returns an empty graph) '''
    
    G = G.copy()
    
    terminals = get_terminals(G)
        
    if len(terminals) >= 1:
        G_steiner = apx.steinertree.steiner_tree(G, terminals, weight='length') 
    
    if len(terminals) == 1:
        G_steiner = nx.Graph()
        G_steiner.add_node(terminals[0], terminal = 1)
    
    return G_steiner

def get_pos(g):
    '''returns dictionary with nodes physical location'''
    return dict( (n, n) for n in g.nodes() )

def plot_graph_on_grid(G, box_size, tree = None, fig_size = 6):
    
    '''Plots graph (connected or not) with option to overlay tree'''
    
    # get terminals from attributes
    terminals = get_terminals(G)
    
    # axes 
    fig, ax = plt.subplots(figsize=(fig_size,fig_size));

    # set limits
    plt.xlim(0, box_size); plt.ylim(0, box_size);
    
    # draw edges and nodes
    nx.draw_networkx_edges(G, get_pos(G))
    nx.draw_networkx_nodes(G, get_pos(G), nodelist = terminals, node_size = 50)
    
    # tick parameters
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    if not tree == None:
        nx.draw(tree, pos = get_pos(tree), node_size = 0, ax = ax, edge_color = 'r', width = 2.5)
    
    plt.show()

def remove_edges_with_landscape(G, landscape, alpha):
    
    '''Remove edges from G using landscape(x,y) > alpha, and also delete nodes left with no edges attached to them '''
    
    #copy G
    G = G.copy()
    
    terminals = get_terminals(G)
    
    # delete edges where the landscape function > alpha 
    for edge in G.edges:
        x, y = 0.5*(np.array(edge[0]) + np.array(edge[1]))
        if landscape(x,y) > alpha:
            G.remove_edge(edge[0],edge[1])

    #delete nodes left with no edges, unless they are terminals
    for node in G.copy().nodes():
        if G.degree(node) == 0 and not node in terminals :
            G.remove_node(node)    
        
    return G 
    
def graph_info(G, verbosity = False):
    '''count clusters, nodes, terminals and edges of G'''
    terminals = get_terminals(G)
    
    n_clusters = len([G.subgraph(c).copy() for c in nx.connected_components(G) if len(c) > 1])
    n_nodes, n_edges, n_terminals =  len(G.nodes()), len(G.edges()), len(terminals) 
    
    if verbosity == True:
        print('Number of clusters: {}; Number of nodes: {}; Number of edges: {}; Number of terminals: {}'\
              .format(n_clusters, n_nodes, n_edges, n_terminals) )
    
    return n_clusters, n_nodes, n_edges, n_terminals
    
######################################### modules to trim with landscape 
    
def trim_graph_landscape(G, box_size, condition, verbosity = False):
    
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
    fLand = ld.find_landscape(box_size, terminals, Nx0 = 50)

    # range of alpha's
    alpha_list = np.linspace(0.5,3.0,51)
    
    # scans a range of alpha
    for alpha in alpha_list:
        if verbosity == True:
            print('For alpha = ', alpha)
        
        #trim graph 
        G_trim = remove_edges_with_landscape(G, fLand, alpha)
        
        # compute properties of trimmed graph
        n_clusters, n_nodes, n_edges, n_terminals = graph_info(G_trim, verbosity)
        
        # stop if condition is satisfied
        if eval(condition):
            break
        
    # extract clusters, remove clusters with no terminals and clusters of zero length
    clusters = [G_trim.subgraph(c).copy() for c in nx.connected_components(G_trim)\
                if len(c) > 1 and len(get_terminals(G_trim.subgraph(c).copy())) > 0 ]
        
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
        new_path = nx.shortest_path(G, node, terminal, weight = 'length')
       
        if len(new_path) <  len(best_path):
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
        
        if len(new_bridge) <  len(best_bridge):
            best_bridge = new_bridge
            
        old_bridge = new_bridge
        
    nx.add_path(tree2, best_bridge)
    
    joint_graph = nx.compose_all([tree1, tree2])
    
    return joint_graph 

def join_all_trees(G, partial_trees):
    
    ''' Joins all trees by sequentially joining individual sub-trees to largest tree. '''
    
    partial_trees = partial_trees.copy()
    
    # trees, largest first
    partial_trees.sort(key = len, reverse = True)
    
    big_tree = partial_trees[0]
    small_trees = partial_trees[1:]
    
    for tree in small_trees:
        big_tree = join_two_trees(G, tree, big_tree)
        
    return big_tree
