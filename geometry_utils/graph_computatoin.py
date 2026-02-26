import numpy as np
from collections import defaultdict, deque
import networkx as nx

'''
    INPUT:
    - n*2 edges
    OUTPUT:
    - edge segments num
    - edge segments: list of connected edges in each segment
    - vertex orders: list of ordered vertices in each segment
    Require:
    - separate node with degree Nd>2 into Nd segments
'''
def connect_edges(edges):
    n = len(edges)
    
    # construct node adjacency
    node_adj = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        node_adj[u].append((v, i))
        node_adj[v].append((u, i))
    degree = {v: len(adj) for v, adj in node_adj.items()}

    e_visited = [False] * n
    e_segments, v_segments = [], []
      
    # edge(u, v) -> trace start from v
    def one_way_trace(start_u, start_v, start_e):
        e_seq, v_seq = [start_e], [start_u, start_v]
        e_visited[start_e] = True

        cur_v, prev_e = start_v, start_e
        while degree.get(cur_v, 0) == 2:
            nbr = node_adj[cur_v]
            next_e = nbr[0][1] if nbr[0][1] != prev_e else nbr[1][1]
            if e_visited[next_e]:
                break
            tmp_u, tmp_v = edges[next_e]
            next_v = tmp_v if tmp_u == cur_v else tmp_u

            e_visited[next_e] = True
            e_seq.append(next_e)
            v_seq.append(next_v)

            cur_v, prev_e = next_v, next_e
        return e_seq, v_seq
    
    for v in list(node_adj.keys()):
        if degree.get(v, 0) == 0:
            continue
        if degree[v] != 2:
            for nbr, e_idx in node_adj[v]:
                if not e_visited[e_idx]:
                    e_seq, v_seq = one_way_trace(v, nbr, e_idx)
                    e_segments.append(e_seq)
                    v_segments.append(v_seq)
    
    # unvisited edges (in cycles)
    for e_id in range(n):
        if e_visited[e_id]:
            continue

        u, v = edges[e_id]
        e_seq, v_seq = one_way_trace(u, v, e_id)
        e_segments.append(e_seq)
        v_segments.append(v_seq)
    
    return e_segments, v_segments


