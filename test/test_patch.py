class PatchMesh:
    def __init__(self, v, f, patch2f):
        self.v = v
        self.f = f
        self.patch2f = {}
        for i in range(len(patch2f)):
            self.patch2f[i] = list(patch2f[i])
        self.om_mesh = om.TriMesh(v, f)
        # self.region2patch = self.read_region_cluster(region_file)
        
        self.initialize() 
    
    def update(self):
        self.f2patch = np.zeros(self.f.shape[0], dtype=int)
        for p_id in self.patch2f:
            for f_id in self.patch2f[p_id]:
                self.f2patch[f_id] = p_id
        self.om_mesh = om.TriMesh(self.v, self.f)

    
    def initialize(self):
        self.v_num = self.v.shape[0]
        self.f_num = self.f.shape[0]
        self.patch_num = len(self.patch2f)

        
        # mesh geometry
        self.f_normal = igl.per_face_normals(self.v, self.f, np.array([1., 0., 0.]))
        self.f_normal = self.f_normal / np.linalg.norm(self.f_normal, axis=1).reshape(-1, 1)
        self.f_center = np.mean(self.v[self.f], axis=1)
        self.f_area = igl.doublearea(self.v, self.f) / 2.
        
        # mesh topology
        self.vv_idx = self.om_mesh.vv_indices()
        self.vf_idx = self.om_mesh.vf_indices()
        self.ef_idx = {}
        for f_h in self.om_mesh.faces():
            f_id = f_h.idx()
            for e_h in self.om_mesh.fe(f_h):
                e_idx = e_h.idx()
                if e_idx not in self.ef_idx.keys():
                    self.ef_idx[e_idx] = [f_id]
                else:
                    self.ef_idx[e_idx].append(f_id)
        
        # patch info
        self.p_boundary_e = self.get_boundary()
        self.patch2v = {}
        for p_id in self.patch2f:
            self.patch2v[p_id] = []
            for f_id in self.patch2f[p_id]:
                for v_id in self.f[f_id]:
                    if v_id not in self.patch2v[p_id]:
                        self.patch2v[p_id].append(v_id)
        self.f2patch = -np.ones(self.f_num, dtype=int)
        for p_id in range(self.patch_num):
            for f_id in self.patch2f[p_id]:
                self.f2patch[f_id] = p_id
        
        # patch adj info
        self.construct_graph()
        self.p_p_adj, self.p_p_adj_v, self.p_p_adj_e, self.p_p_adj_f, self.p_p_adj_of, self.p_p_adj_info = self.adjacency_construct()
        self.compute_graph_edge()
        
        
    
    def get_boundary(self):
        p_boundary_e = {}
        for p_id in range(self.patch_num):
            e_set = set()
                        
            # count appearances of edge
            edge = {}
            for f_id in self.patch2f[p_id]:
                f_h = self.om_mesh.face_handle(f_id)
                for e_h in self.om_mesh.fe(f_h):
                    e_idx = e_h.idx()
                    if e_idx not in edge:
                        edge[e_idx] = 1
                    else:
                        edge[e_idx] = 2

            for e_idx in edge:
                if edge[e_idx] == 1:
                    e_set.add(e_idx)
    
            p_boundary_e[p_id] = np.array(list(e_set), dtype=int)
        
        return p_boundary_e



    def adjacency_construct(self):
        # all boundary edges 
        all_edges = []
        for p_id in range(self.patch_num):
            boundary_e = self.p_boundary_e[p_id]
            all_edges.extend(boundary_e)
        all_edges = np.unique(np.array(all_edges))
        
        # init
        p_p_adj = {}
        p_p_adj_info = {}
        for p_id in range(self.patch_num):
            p_p_adj[p_id] = []
            p_p_adj_info[p_id] = {}
        
        # get info on all edges
        for e_id in all_edges:
            # continue if mesh's boundary
            if self.om_mesh.is_boundary(self.om_mesh.edge_handle(e_id)):
                continue
            
            f1, f2 = self.ef_idx[e_id]
            p1, p2 = self.f2patch[f1], self.f2patch[f2]
            
            # adjacent matrix
            if p2 not in p_p_adj[p1]:
                p_p_adj[p1].append(p2)
            if p1 not in p_p_adj[p2]:
                p_p_adj[p2].append(p1)
            
            # adjacent info
            if p1 not in p_p_adj_info[p2]:
                p_p_adj_info[p2][p1] = []
            p_p_adj_info[p2][p1].append([f1, f2, e_id])
            if p2 not in p_p_adj_info[p1]:
                p_p_adj_info[p1][p2] = []
            p_p_adj_info[p1][p2].append([f2, f1, e_id])
        
        
        # for p-p border, assemble ordered e, v, f
        p_p_adj_e, p_p_adj_v, p_p_adj_f, p_p_adj_of = {}, {}, {}, {}
        for p_id in range(self.patch_num):
            # init
            p_p_adj_e[p_id] = {}
            p_p_adj_v[p_id] = {}
            p_p_adj_f[p_id] = {}
            p_p_adj_of[p_id] = {}
            for p_id_adj in p_p_adj[p_id]:
                p_p_adj_e[p_id][p_id_adj] = []
                p_p_adj_v[p_id][p_id_adj] = []
                p_p_adj_f[p_id][p_id_adj] = []
                p_p_adj_of[p_id][p_id_adj] = []
            
            
            # push and order
            for p_id_adj in p_p_adj[p_id]:
                info = np.array(p_p_adj_info[p_id][p_id_adj])
                f1, f2, e_id = info[:, 0], info[:, 1], info[:, 2]
                
                # get edges
                edge_v_id = np.zeros((len(e_id), 2), dtype=int)
                for i in range(len(e_id)):
                    e_h = self.om_mesh.edge_handle(e_id[i])
                    he_h = self.om_mesh.halfedge_handle(e_h, 0)
                    edge_v_id[i][0] = self.om_mesh.from_vertex_handle(he_h).idx()
                    edge_v_id[i][1] = self.om_mesh.to_vertex_handle(he_h).idx()
                
                
                
                edge_order, ordered_v = connect_edge_to_line(edge_v_id)
                
                p_p_adj_e[p_id][p_id_adj] = e_id[edge_order]
                p_p_adj_v[p_id][p_id_adj] = ordered_v
                p_p_adj_f[p_id][p_id_adj] = f1[edge_order]
                p_p_adj_of[p_id][p_id_adj] = f2[edge_order]

        return p_p_adj, p_p_adj_v, p_p_adj_e, p_p_adj_f, p_p_adj_of, p_p_adj_info



    def construct_graph(self):
        graph = nx.Graph()
        for p_id in self.patch2f:
            graph.add_node(p_id)
            graph.nodes[p_id]['f'] = self.patch2f[p_id]
            graph.nodes[p_id]['v'] = self.patch2v[p_id]
        self.graph = graph
        
        
    def compute_graph_edge(self):
        for p_id in self.patch2f:
            for n_p_id in self.p_p_adj[p_id]:
                self.graph.add_edge(p_id, n_p_id)
                self.graph.edges[p_id, n_p_id]['v'] = self.p_p_adj_v[p_id][n_p_id]
                self.graph.edges[p_id, n_p_id]['e'] = self.p_p_adj_e[p_id][n_p_id]
            
    '''
        update for each neighbor nodes:
        - edges    of EDGE: list(e_id in om_mesh) not ordered
        - vertices of EDGE: list(v_id in om_mesh) ordered
    '''
    def update_graph_edge(self):
        mesh = om.TriMesh(self.v, self.f)
        
        for p in self.graph.nodes:
            edges = set()
            for f in self.graph.nodes[p]['f']:
                for e in self.om_mesh.fe(mesh.face_handle(f)):
                    edges.add(e.idx())
            self.graph.nodes[p]['e'] = edges
        
        for edge in self.graph.edges:
            p1, p2 = edge
            common_e = list(self.graph.nodes[p1]['e'].intersection(self.graph.nodes[p2]['e']))
            common_e_v = np.zeros((len(common_e), 2), dtype=int)
            for i in range(len(common_e)):
                e_h = mesh.edge_handle(common_e[i])
                he_h = mesh.halfedge_handle(e_h, 0)
                common_e_v[i][0] = mesh.from_vertex_handle(he_h).idx()
                common_e_v[i][1] = mesh.to_vertex_handle(he_h).idx()
            
            _, ordered_v = connect_edge_to_line(common_e_v)
            self.graph.edges[p1, p2]['v'] = ordered_v
    
            
