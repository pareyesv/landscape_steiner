'''
An implementation of STP instances using networkx
'''

from typing import List, Union

import networkx as nx
import pandas as pd

__version__ = "0.1"


class SteinerTreeProblem(nx.Graph):
    '''
    A class to abstract a STP problem as a network x graph

    As the class extends the Graph class in networkx, it uses
    this class utilities to store the information.

    * Update factors (no discount rates) are stored as graph attributes
      with attribute names update_factor-01, update_factor-02, ....
    * The horizon (number of periods) is stored as an graph attribute named
      "periods"
    * Node values are stored as a "prize" node attribute.
    * Weights of the the edges are stored as a "weight" edge attribute.
    * To mark the terminal nodes, there is a node attribute named is_terminal
      which is equal to 1 for terminal nodes and zero otherwise.
    * If the nodes have coordinates, they are stored as node attributes named
      "x" and "y", respectively.

    Notice that time-periods are numbered from 1 to the horizon.

    # Implementation of STP Files

    Currently it supports reading STP Files (partially implemented)
    and writing them to a file.

    Support for STP Files
    - Graph section: Nodes and edges (with weights)
    - Terminals
    - Comments starting with # (everything after that is ignored)
    - Coordinates.

    TODO: create functions to set node and edge attributes and refactor
    '''

    def __init__(
        self,
        update_factors=None,
        stp_file=None,
        root_node=1,
        default_prize=1,
        default_weight=1,
        node_period_attributes: dict = None,
        edge_period_attributes: dict = None,
        graph_period_attributes: dict = None,
        period_suffix_format=r"-{:02d}",
        node_attribute_format="{}",
        edge_attribute_format="{}",
        graph_attribute_format="{}",
        terminal_value=1,
        root_value=2,
        no_terminal_value=0,
    ):

        nx.Graph.__init__(self)

        if update_factors:
            self._uf = update_factors
        else:
            self._uf = [1.0]

        self.period_suffix_format = period_suffix_format
        self.node_attribute_format = node_attribute_format
        self.edge_attribute_format = edge_attribute_format
        self.graph_attribute_format = graph_attribute_format
        self.root_node = root_node
        self.terminal_value = terminal_value
        self.root_value = root_value
        self.no_terminal_value = no_terminal_value

        self.graph["periods"] = self.num_periods()
        self.set_graph_period_attribute("update_factor", self._uf)

        # graph attributes
        if graph_period_attributes is not None:
            for key, value in graph_period_attributes.items():
                self.set_graph_period_attribute(key, value)

        if stp_file:
            self.parse(
                stp_file,
                default_prize=default_prize,
                default_weight=default_weight,
                node_period_attributes=node_period_attributes,
                edge_period_attributes=edge_period_attributes,
            )

    def num_periods(self):
        return len(self._uf)

    def set_graph_period_attribute(self, name: str, values: Union[int, float,
                                                                  List]):
        """Set graph attribute per period.

        Args:
            name (str): name of the attribute
            values (int or float or list): list of values per period, or a number if constant
        """
        if not isinstance(values, list):
            values = [values] * self.num_periods()
        for t in self.iter_periods():
            self.graph[(self.graph_attribute_format.format(name) +
                        self.period_suffix_format).format(t)] = values[t - 1]

    def get_graph_period_attribute(self, name, t=None):
        """Get graph attribute per period.

        Args:
            name (str): name of the attribute
            t (int, optional): Period number. If None, include the period in `name` as suffix.
                                Defaults to None.

        Returns:
            value of the graph attribute
        """
        if t is None:
            return self.graph[name]
        else:
            return self.graph[(self.graph_attribute_format.format(name) +
                               self.period_suffix_format).format(t)]

    def parse(
        self,
        stp_file,
        default_prize=1,
        default_weight=1,
        node_period_attributes: dict = None,
        edge_period_attributes: dict = None,
    ):
        '''
        Parses a STP file.

        stp_file = Filename of the file to read from.
        default_prize = Prize to set for each node as default (STP files do not have weights on nodes)
        default_weight = Cost/Weight to put for default on edges (STP files may not have weight on edges)
        '''
        f = open(stp_file)
        edges = []
        arcs = []
        coords = {}
        terminals = set()
        in_graph, in_terms, in_coords = False, False, False

        for line in f:

            # remove comment
            idx = line.find("#")
            if idx >= 0:
                line = line[0:idx]

            # parse graph section
            if line.startswith("SECTION Graph") or line.startswith(
                    "Section Graph"):
                in_graph, in_terms, in_coords = True, False, False
            if in_graph:
                if line.startswith("Nodes"):
                    n_nodes = int(line.split()[1])
                if line.startswith("Edges"):
                    n_edges = int(line.split()[1])
                else:
                    # STP files specification indicates that  edges no weight,
                    # however many fails contain a weight attribute...
                    if line.startswith("E "):
                        sp_line = line.split()
                        v1, v2 = int(sp_line[1]), int(sp_line[2])
                        if len(sp_line) < 4:
                            edges.append((v1, v2, default_weight))
                        else:
                            edges.append((v1, v2, int(sp_line[3])))
                # STP files also can have "arcs", but they are not supported
                #if line.startswith("A"):
                #    sp_line = line.split()
                #    v1, v2, w = int(sp_line[1]), int(sp_line[2], int(sp_line[3]))
                #    arcs.append( (v1,v2,w))

            # parse terminals
            if line.startswith("SECTION Terminals") or line.startswith(
                    "Section Terminals"):
                in_graph, in_terms, in_coords = False, True, False

            if in_terms:
                if not line.startswith("Terminals") and line.startswith("T"):
                    terminals.add(int(line.split()[1]))

            # parse coordinates
            if line.startswith("SECTION Coordinates") or line.startswith(
                    "Section Coordinates"):
                in_graph, in_terms, in_coords = False, False, True

            if in_coords:
                if line.startswith("DD"):
                    sp = line.split()
                    n, x, y = int(sp[1]), int(sp[2]), int(sp[3])
                    coords[n] = (x, y)

        # create the undirected graph

        self.add_nodes_from([i + 1 for i in range(n_nodes)])

        for u in self.nodes():
            # prize
            self.nodes[u]["prize"] = default_prize
            # node period attributes
            if node_period_attributes is not None:
                for key, value in node_period_attributes.items():
                    for t in self.iter_periods():
                        self.nodes[u][
                            self.node_attribute_format.format(key) +
                            self.period_suffix_format.format(t)] = value
            # terminals
            if u in terminals:
                if u == self.root_node:
                    self.nodes[u]["is_terminal"] = self.root_value
                else:
                    self.nodes[u]["is_terminal"] = self.terminal_value
            else:
                self.nodes[u]["is_terminal"] = self.no_terminal_value

        for e in edges:
            # weight/distance
            self.add_edge(e[0], e[1], weight=e[2])
            self.add_edge(e[0], e[1], distance=e[2])
            # edge attributes
            if edge_period_attributes is not None:
                for key, value in edge_period_attributes.items():
                    for t in self.iter_periods():
                        self.edges[(
                            e[0],
                            e[1])][self.edge_attribute_format.format(key) +
                                   self.period_suffix_format.format(t)] = value

        if len(coords) > 0:
            for u, (x, y) in coords.items():
                self.nodes[u]["x"] = x
                self.nodes[u]["y"] = y

    def iter_periods(self):
        """Returns a period iterator."""
        for t in range(1, self.num_periods() + 1):
            yield t

    def iter_terminals(self):
        """Terminal iterator."""
        return (node for node, value in nx.get_node_attributes(
            self, "is_terminal").items() if self.is_terminal(node))

    def terminals(self):
        """List of terminals."""
        return list(self.iter_terminals())

    def is_terminal(self, t: int) -> bool:
        return self.nodes[t]["is_terminal"] in [
            self.terminal_value, self.root_value
        ]

    def add_terminal(self, t: int) -> None:
        if t == self.root_node:
            self.nodes[t]["is_terminal"] = self.root_value
        else:
            self.nodes[t]["is_terminal"] = self.terminal_value

    def coordinates_dict(self) -> dict:
        x_coords = nx.get_node_attributes(self, "x")
        y_coords = nx.get_node_attributes(self, "y")
        return {
            n: (x_coords.get(n, None), y_coords.get(n, None))
            for n in self.nodes()
        }

    def get_coordinates(self, node: int):
        return (self.nodes[node]["x"], self.nodes[node]["y"])

    def set_coordinates(self, node, x, y):
        self.nodes[node]["x"] = x
        self.nodes[node]["y"] = y

    def write_stp(self, file_name):
        f = open(file_name, "w")
        print("0 STP File, STP Format Version 1.0", file=f)

        print("\nSECTION Graph", file=f)
        print(f"Nodes {self.order()}", file=f)
        print(f"Edges {self.size()}", file=f)
        weights = nx.get_edge_attributes(self, 'weight')
        for u, v in self.edges():
            print(f"E {u} {v} {weights[u,v]}", file=f)
        print("END", file=f)

        print("\nSECTION Terminals", file=f)
        print(f"Terminals {len(self.terminals())}", file=f)
        for t in self.terminals():
            print(f"T {t}", file=f)
        print("END", file=f)

        if len(self.coordinates_dict()) > 0:
            print("\nSECTION Coordinates", file=f)
            for n, (x, y) in self.coordinates_dict().items():
                print(f"DD {n} {x} {y}", file=f)
            print("END", file=f)

        print("\nEOF", file=f)
        f.close()

    def write_graphml(self, file_name):
        nx.write_graphml_lxml(self, file_name)

    def edge_attributes_df(self) -> pd.DataFrame:
        """Returns a pandas DataFrame where each row contains edge attributes"""
        return nx.to_pandas_edgelist(self).set_index(["source", "target"])

    def node_attributes_df(self, index_name="node") -> pd.DataFrame:
        """Returns a pandas DataFrame where each row contains node attributes"""
        df = pd.DataFrame.from_dict(dict(self.nodes(data=True)), orient='index')
        df.index.name = index_name
        return df

    def graph_attributes(self) -> dict:
        """Returns a dictionnary with the graph attributes.
        """
        return self.graph

    def refactor_attribute_periods(
        self,
        attribute_list: List[str] = ["update_factor"],
        stp_attributes_dict: dict = None,
        attribute_format: str = None,
    ):
        """Refactor multi-period attributes into one list-valued attribute.

        The list order corresponds to the values for different periods.
        """
        if stp_attributes_dict is None:
            stp_attributes_dict = self.graph_attributes()
        if attribute_format is None:
            attribute_format = self.graph_attribute_format

        updated_attributes = dict.fromkeys(attribute_list,
                                           [None] * self.num_periods())
        for attribute in attribute_list:
            attribute_value_list = []
            for t in self.iter_periods():
                attribute_name = (attribute_format.format(attribute) +
                                  self.period_suffix_format.format(t))
                attribute_value_list.append(
                    stp_attributes_dict.get(attribute_name, None))
            updated_attributes[attribute] = attribute_value_list
        return updated_attributes


if __name__ == '__main__':
    for i in range(10):
        file_name = f"stlib/datasets/B/b{i+1:02d}.stp"
        print(f"Parsing file {file_name}")
        stp = SteinerTreeProblem(stp_file=file_name,
                                 update_factors=[1.0, 0.75, 0.5, 0.25, 0.0])
        print(f"Found {stp.order()} nodes and {stp.size()} edges")
        stp.write_graphml(f"b{i+1:02d}.graphml")
        #print (f"Edges are {stp.edges()}")
