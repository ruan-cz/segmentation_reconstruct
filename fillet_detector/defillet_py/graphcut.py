"""Binary face graph cut using NetworkX's min-cut implementation."""
import networkx as nx
import numpy as np


def binary_cut(unary_zero, unary_one, edges, weights):
    """最小化逐面标签代价与邻接面异标签惩罚，返回 0/1 标签数组。"""
    graph = nx.DiGraph()
    source, sink = "source", "sink"
    for index, (cost_zero, cost_one) in enumerate(zip(unary_zero, unary_one)):
        # 源侧为标签 0：割断面到汇的边，支付 cost_zero；标签 1 反之。
        graph.add_edge(source, index, capacity=float(cost_one))
        graph.add_edge(index, sink, capacity=float(cost_zero))
    for (left, right), weight in zip(edges, weights):
        weight = float(weight)
        if weight > 0:
            graph.add_edge(left, right, capacity=weight)
            graph.add_edge(right, left, capacity=weight)
    _, partition = nx.minimum_cut(graph, source, sink)
    source_side, _ = partition
    # In GCO, the source segment is label 0 and the sink segment is label 1.
    return np.array(
        [int(index not in source_side) for index in range(len(unary_zero))],
        dtype=np.int32,
    )
