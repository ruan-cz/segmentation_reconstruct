import math
import networkx as nx
from typing import Dict, List, Tuple, Any, Set
import numpy as np


def graph_min_cut(G, seeds, convex_edges, 
    alpha=1, beta=0.05):

    graph = nx.Graph()
    edge_cost = {}
    for node in G.nodes:
        graph.add_node(node)
    for edge in convex_edges:
        u, v = edge
        graph.add_edge(u, v)
        edge_cost[(u, v)] = G.edges[u, v]['convexity']
        edge_cost[(v, u)] = G.edges[u, v]['convexity']
    
    results = {}

    max_area = max([G.nodes[node]['area'] for node in G.nodes])

    for seed in seeds:
        force_nodes = set()
        dist_map = dict(nx.shortest_path_length(graph, seed))
        for n, d in dist_map.items():
            if d <= 1:
                force_nodes.add(n)
        sorted_nodes = sorted(dist_map.items(), key=lambda x: (x[1], x[0]))
        for n, d in sorted_nodes[:1]:
            force_nodes.add(n)



        _g = nx.DiGraph()
        source = 's'
        sink = 't'
        _g.add_node(source)
        _g.add_node(sink)
        for node in G.nodes:
            _g.add_node(node)


        for edge in convex_edges:
            u, v = edge
            cost = edge_cost[(u, v)]
            _g.add_edge(u, v, capacity=cost)
            _g.add_edge(v, u, capacity=cost)
        
        for node in G.nodes:
            Si = max(0, beta - alpha * G.nodes[node]['area'] / max_area)
            Ti = max(0, alpha * G.nodes[node]['area'] / max_area - beta)
            
            
            if node in force_nodes:
                _g.add_edge(source, node, capacity=math.inf)
            else:
                if Si > 0:
                    _g.add_edge(source, node, capacity=Si)
                if Ti > 0:
                    _g.add_edge(node, sink, capacity=Ti)
        
        if not force_nodes:
            _g.add_edge(source, seed, capacity=math.inf)

        cut_value, (S_set, T_set) = nx.minimum_cut(
            _g, source, sink, capacity='capacity'
        )

        raw_source_set = S_set - {source, sink}
        comp_nodes = set()
        if seed in raw_source_set:
            sub_graph = graph.subgraph(raw_source_set)
            for comp in nx.connected_components(sub_graph):
                if seed in comp:
                    comp_nodes = comp
                    break
        else:
            comp_nodes = set()

        results[seed] = comp_nodes
    
    return results



    