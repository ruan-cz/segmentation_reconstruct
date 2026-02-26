import numpy as np
import igl
import meshplot as mp
# import timeit
import openmesh as om
import os

from utils.utils import *
from primitive_fitting import *
from proxy import *


vsa_obj_id = 0

class VSA():
    def __init__(self, v, f, proxy_num=200, norme=1, max_iters=100):
        self.v = v
        self.f = f
        self.norme = norme
        self.proxy_num = proxy_num
        self.f_num = f.shape[0]
        self.v_num = v.shape[0]
        

        self.max_iters = max_iters
        self.iter_diff_radio_threhold = 1 * 1e-3
        self.iter_diff_threhold = 1 * 1e-2
        # self.threshold_convexity = np.cos(87.5 * np.pi / 180)
        
        self.init_modules()
        
        global vsa_obj_id
        vsa_obj_id += 1
    
    
    def init_modules(self):
        self.distance = Distance(self.v, self.f)
        self.partition = Partition(self.v, self.f, self.distance)
        self.proxy = Proxy(self.v, self.f, self.distance)
    

    # OUTPUT:
    #   min_R: proxies of faces, f_num * 1
    #   proxy_cluster: cluster of proxies, {proxy_id: []}
    def iteration(self, visualize=False):
        all_errors = []
        all_R = []
        cur_iteration = 0
        cur_error = 0.
        precedent_error = np.inf

        n_colors = colors[:self.proxy_num]
        
        cur_proxy_num = 1
        while cur_iteration < self.max_iters:
            # if abs((cur_error - precedent_error) / precedent_error) < self.iter_diff_radio_threhold:
            #     break
            # if abs(cur_error - precedent_error) < self.iter_diff_threhold:
            #     break
            if (abs((cur_error - precedent_error) / precedent_error) < self.iter_diff_radio_threhold) & (abs(cur_error - precedent_error) < self.iter_diff_threhold):
                break
            
            # if cur_iteration == 0:
            #     # cur_R = self.partition.initial_partition(self.proxy_num, self.norme)
            #     cur_R, _ = self.partition.initial_partition_furthest(self.proxy_num, self.norme)
            #     if visualize:
            #         f_colors = np.ones((self.f_num, 3), dtype=float)
            #         for i in range(self.proxy_num):
            #             f_colors[cur_R == i] = n_colors[i]
            #         mesh = mp.plot(self.v, self.f, c=f_colors, shading={"wireframe": True})
            # else:
            #     cur_R = self.partition.region_growing_proxy_initialized(cur_R, cur_proxy_center, cur_proxy_normal, self.norme)
            # cur_proxy_center, cur_proxy_normal = self.proxy.new_proxy(cur_R, self.proxy_num, self.norme)

            if cur_iteration == 0:
                # print(self.norme)
                cur_R, cur_proxy_triangle = self.partition.initial_partition(1, self.norme)
                # cur_R, cur_proxy_triangle = self.partition.initial_partition_furthest(1, self.norme)
                if visualize:
                    f_colors = np.ones((self.f_num, 3), dtype=float)
                    for i in range(self.proxy_num):
                        f_colors[cur_R == i] = n_colors[i%500]
                    mesh = mp.plot(self.v, self.f, c=f_colors, shading={"wireframe": True})
            else:
                cur_R, cur_proxy_triangle = self.partition.insert_proxy(cur_proxy_triangle, self.norme)
                cur_proxy_num += 1
            cur_proxy_center, cur_proxy_normal = self.proxy.new_proxy(cur_R, cur_proxy_num, self.norme)
            
        
            precedent_error = cur_error
            cur_error = self.distance.distance_error(cur_R, cur_proxy_center, cur_proxy_normal, self.norme)
            
            if cur_error in all_errors:
                break
            all_errors.append(cur_error)
            all_R.append(cur_R)
            cur_iteration += 1
            print("[Iteration {}/{}] Error: {}, Delta-Error: {}".format(
                cur_iteration, self.max_iters, cur_error, abs(cur_error - precedent_error)
            ))

            if visualize:        
                proxy_colors = np.array([ n_colors[cur_R[i]%50] for i in range(self.f_num) ])
                # mesh.update_object(colors=proxy_colors)
                mp.plot(self.v, self.f, c=proxy_colors, shading={"wireframe": True})

                # # write R
                # path = 'render/2/vsa/{}/iter_{}.txt'.format(vsa_obj_id, cur_iteration)
                # f = open(path, 'w')
                # for i in range(cur_R.shape[0]):
                #     f.write(str(cur_R[i]) + '\n')
                # f.close()


                
        diff_radio = abs((cur_error - precedent_error) / precedent_error)
        diff = abs(cur_error - precedent_error)
        if diff_radio < self.iter_diff_radio_threhold:
            print("Converged(diff radio): {}".format(diff_radio))
        elif diff < self.iter_diff_threhold:
            print("Converged(diff): {}".format(diff))
        elif cur_iteration == self.max_iters:
            print("Max Iteration Reached")
        else:
            print("Cycle Proxy Detected")
            
        min_R = all_R[np.argmin(all_errors)]
        if visualize:
            proxy_colors = np.array([ n_colors[min_R[i]%50] for i in range(self.f_num) ])
            # mesh.update_object(colors=proxy_colors)
            mp.plot(self.v, self.f, c=proxy_colors, shading={"wireframe": True})
            
        
        proxy_cluster = self.partition.partition_to_cluster(min_R)
        
        return min_R, proxy_cluster


    def write_proxy_cluster(self, proxy_cluster, proxy_cluster_file):
        with open(proxy_cluster_file, "w") as f:
            for key in proxy_cluster.keys():
                for val in proxy_cluster[key]:
                    f.write(str(val) + " ")
                f.write("\n")


def pre_split(v, f, visualize=False, visualize_part=False):
    mesh = om.TriMesh(v, f)
    
    f_center = np.mean(v[f], axis=1)
    f_area = igl.doublearea(v, f) / 2
    f_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
    f_normal = f_normal / np.linalg.norm(f_normal, axis=1).reshape(-1, 1)
    # threshold_convexity = np.cos(89.5 * np.pi / 180)
    # threshold_convexity = np.cos(87.5 * np.pi / 180)
    threshold_convexity = np.cos(77.5 * np.pi / 180)
    # threshold_convexity = np.cos(67.5 * np.pi / 180)
    # threshold_convexity = np.cos(57.5 * np.pi / 180)
    # threshold_convexity = np.cos(47.5 * np.pi / 180)
    threshold_fitting = 0.9

    # # show normal on mesh
    # plot = mp.plot(v, f, np.ones((f.shape[0], 3)), shading={'wireframe': True})
    # plot.add_lines(
    #     f_center,
    #     f_center + f_normal * 0.01,
    #     shading={'line_width': 1, 'line_color': 'red'},
    # )

    # convex_edge, concave_edge = get_border_edge(mesh, f_center, f_normal, threshold_convexity, get_boundary=False)
    convex_edge, concave_edge, boundary_edge = get_border_edge(mesh, f_center, f_normal, threshold_convexity, get_boundary=True)

    print(f'convex edges: {len(convex_edge)}, concave edges: {len(concave_edge)}')
    
    edge = []
    if len(convex_edge) > 0:
        edge.extend(convex_edge[:, 0])
    if len(concave_edge) > 0:
        edge.extend(concave_edge[:, 0])
    # if len(boundary_edge) > 0:
    #     edge.extend(boundary_edge[:, 0])
    
    # edges = []
    # for i in range(convex_edge.shape[0]):
    #     edges.append([convex_edge[i][1], convex_edge[i][2]])
    # for i in range(concave_edge.shape[0]):
    #     edges.append([concave_edge[i][1], concave_edge[i][2]])
    # edges = np.array(edges)
    
    # if visualize:
    #     # get all edges
    #     edges_v_id = []
    #     edges_color = []
    #     for i in range(len(convex_edge)):
    #         edges_v_id.append([convex_edge[i][1], convex_edge[i][2]])
    #         edges_color.append([0, 0, 1])
    #     for i in range(len(concave_edge)):
    #         edges_v_id.append([concave_edge[i][1], concave_edge[i][2]])
    #         edges_color.append([1, 0, 0])
    #     edges_color = np.array(edges_color)

    #     f_colors = np.ones((f.shape[0], 3))
    #     xml_path = '../meshsegment_paper/camera/140780.xml'
    #     render_mesh_vtk(
    #         v, f, xml_path=xml_path, write_img=True,
    #         output_path='render/140780/vsa_borders.png',
    #         # 'render/140750/raw_borders.png',
    #         # 'render/cad16/raw_borders.png',
    #         # 'render/2/all_patch_borders.png',
    #         # 'render/2/concave_loop.png',
    #         draw_e=True, e = edges_v_id, edge_radius=0.3, _e_color = edges_color, 
    #         draw_f=True, _f_color=f_colors,
    #     )
        
    
    splited_f = split(mesh, edge)
    print(f'{len(splited_f)} regions')
    
    clusters = []
    region_cluster = []

    for i in range(len(splited_f)):
        region_f = np.array(splited_f[i])
        

        cur_v, cur_f = get_submesh(v, f, region_f)
        
        if visualize_part:
            print(f'region {i}')
            mp.plot(cur_v, cur_f, shading={'wireframe': False})
        
        if cur_f.shape[0] == 1:
            print('planar')
            clusters.append(region_f)
            region_cluster.append(len(clusters) - 1)
            continue
        
        f_center = np.mean(cur_v[cur_f], axis=1)
        f_area = igl.doublearea(cur_v, cur_f) / 2
        f_normal = igl.per_face_normals(cur_v, cur_f, np.array([1., 0., 0.]))
        f_normal = f_normal / np.linalg.norm(f_normal, axis=1).reshape(-1, 1)
        primitive_fitting = PrimitiveFitting(cur_v, cur_f, f_normal, f_center, f_area)

        # primitive fitting
        f_id = np.arange(cur_f.shape[0])
        fit_rate, fit_planar_params = primitive_fitting.fit_planar(f_id)
        print(f"planar fit:{fit_rate}")
        if fit_rate > threshold_fitting:
            print(f"planar: {fit_planar_params}")
            clusters.append(region_f)
            region_cluster.append(len(clusters) - 1)
            continue
        fit_rate, fit_cylinder_params = primitive_fitting.fit_cylinder(f_id, visualize=True)
        print(f"cylinder fit:{fit_rate}")
        if fit_rate > threshold_fitting:
            print(f"cylinder: {fit_cylinder_params}")
            clusters.append(region_f)
            region_cluster.append(len(clusters) - 1)
            continue

        # fit_rate, fit_sphere_params = primitive_fitting.fit_sphere(f_id)
        # print(f"sphere fit:{fit_rate}")
        # if fit_rate > threshold_fitting:
        #     print(f"sphere: {fit_sphere_params}")
        #     clusters.append(region_f)
        #     region_cluster.append(len(clusters) - 1)
        #     continue

        
        # fit_rate, fit_cone_params = primitive_fitting.fit_cone(f_id)
        # print(f"cone fit:{fit_rate}")
        # if fit_rate > threshold_fitting:
        #     print(f"cone: {fit_cone_params}")
        #     clusters.append(region_f)
        #     region_cluster.append(len(clusters) - 1)
        #     continue
        
        # fit_rate, fit_extrusion_params = primitive_fitting.fit_extrusion(f_id)
        # print(f"extrusion fit:{fit_rate}")
        # if fit_rate > threshold_fitting:
        #     print(f"extrusion: extrude thourgh {fit_extrusion_params}")
        #     clusters.append(region_f)
        #     region_cluster.append(len(clusters) - 1)
        #     continue
        

        # else do vsa
        # print(len(region_f), region_f[:10], region_f[-10:])
        # # write region_f to file
        # path = 'render/2/vsa/region_{}.txt'.format(vsa_obj_id)
        # tmp_file = open(path, 'w')
        # for j in range(len(region_f)):
        #     tmp_file.write(str(region_f[j]) + '\n')
        # tmp_file.close()

        vsa = VSA(cur_v, cur_f, proxy_num=50)
        _, proxy_cluster = vsa.iteration(visualize=False)
        raw_region_id = len(clusters)
        for proxy_id in proxy_cluster.keys():
            f_id = np.array(proxy_cluster[proxy_id])
            clusters.append(region_f[f_id])
        region_cluster.append(range(raw_region_id, len(clusters)))

        # # else consider as one proxy
        # clusters.append(region_f)
        # region_cluster.append(len(clusters) - 1)

    
    face_color = np.ones((f.shape[0], 3))
    for i in range(len(clusters)):
        cur_color = colors[i%500]
        for f_id in clusters[i]:
            face_color[f_id] = cur_color
    mesh = mp.plot(v, f, c=face_color, shading={"wireframe": False})
    return clusters, region_cluster, face_color
    


# file = "example_data/ABC/00000002.obj"
# file = "example_data/example_CAD/2.obj"
# file = "example_data/example_CAD/cad16.obj"
# file = "example_results/merge/cad/cad16/iter_1_part_1.ply"
# file = "example_results/merge/cad/cad16/iter_1_part_0.ply"

# file = "example_data/selected/00140179_7d7f0dc4c3b8202c03a32241_trimesh_000.obj"
# file = "example_data/selected/00140180_7b0779cf3270ccee61e01b3e_trimesh_000.obj"
# file = "example_data/selected/00140223_5374ce893909eddbc7536b71_trimesh_000.obj"
# file = "example_data/selected/00140302_dce0fb9e808b32ded73025c1_trimesh_000.obj"
# file = 'example_data/selected/00140316_62dfc752c2876cd94647532a_trimesh_011.obj'
# file = "example_data/selected/00140400_e4a5e7862a9beda824b2e00c_trimesh_003.obj"
# file = "example_data/selected/00140553_e2c0841b6c86e3bfdcc8c477_trimesh_000.obj"
# file = "example_data/selected/00140682_572115b8e4b01ff83d93b7c0_trimesh_000.obj"
# file = 'example_data/selected/00140708_1c9a6a91c5121e671690124c_trimesh_000.obj'
file = 'example_data/selected/00140710_33073a6cb55fadd20ec0e3fb_trimesh_001.obj'
# file = 'example_data/selected/00140722_0752ced969c82bbb19ce666b_trimesh_000.obj'
# file = "example_data/selected/00140750_0da4d13f288d4cd70deecf20_trimesh_001.obj"
# file = "example_data/selected/00140756_eaf6606df2a105f356a99677_trimesh_000.obj"
# file = "example_data/selected/00140780_d56b3836d8a40e3035d42380_trimesh_000.obj"
# file = "example_data/selected/00140919_3023c3cfd7d6b1a80700e17e_trimesh_000.obj"
# file = "example_data/selected/00140923_0e818d7ff8780682567c824e_trimesh_000.obj"
# file = "example_data/selected/00140936_9a2b76f7e489ea90f5e41ef8_trimesh_007.obj"
# file = "example_data/selected/00140967_631426bb5199fa39fc8eeb1c_trimesh_008.obj"
# file = "example_data/selected/00140982_f2e481e34cff3a450799acac_trimesh_000.obj"
# file = "example_data/selected/00141000_e5a1548042d09e9e460ec51e_trimesh_000.obj"
# file = "example_data/selected/00141019_e19ef37e52d5182509d42ac9_trimesh_000.obj"
# file = "example_data/selected/00141097_57222034e4b08229a097b208_trimesh_010.obj"
# file = "example_data/selected/00141172_5722364be4b0e6559901e691_trimesh_000.obj"
# file = "example_data/selected/00141176_5ff893b2e5c70dac030147d6_trimesh_001.obj"


# file = "abc_all/meshes/00023435_385b221a0d58490b84f3edee_trimesh_000.off"
# file = "abc_all/meshes/00023471_7f45ff9e8c754def8bb4b1cb_trimesh_000.off"
# file = "abc_all/meshes/00023576_35c7024f831f44048104c331_trimesh_000.off"
# file = "abc_all/meshes/00023582_9c917172a61b472fb0e6ae3c_trimesh_002.off"
# file = "abc_all/meshes/00023792_30c31f050d2c40139e9b36ca_trimesh_001.off"
# file = "abc_all/meshes/00024036_de448e53313b4faa9dff2245_trimesh_003.off"
# file = "abc_all/meshes/00024040_de448e53313b4faa9dff2245_trimesh_007.off"
# file = "abc_all/meshes/00024044_de448e53313b4faa9dff2245_trimesh_011.off"
# file = "abc_all/meshes/00024082_de448e53313b4faa9dff2245_trimesh_049.off"
# file = "abc_all/meshes/00024089_de448e53313b4faa9dff2245_trimesh_056.off"
# file = "abc_all/meshes/00024132_a087bf84d26046349bbbd6f6_trimesh_000.off"
# file = "abc_all/meshes/00024448_274a7c9d6def4a699961a747_trimesh_002.off"
# file = "abc_all/meshes/00024792_34a17822747a4b20a8c2954b_trimesh_009.off"
# file = "abc_all/meshes/00024961_e26f4b039c6c48c9878b21fe_trimesh_000.off"
# file = "abc_all/meshes/00025063_9347be1a9dad4e70852dcce2_trimesh_000.off"
# file = "abc_all/meshes/00025183_021c990acfc648618a49bd47_trimesh_000.off"
# file = "abc_all/meshes/00025611_37f3a717a9b842cfbe5f6005_trimesh_002.off"
# file = "abc_all/meshes/00025633_2aaf78a0f41a47b1bae8c447_trimesh_003.off"
# file = "abc_all/meshes/00025710_6fe81cf35e2740b3bbde9aa6_trimesh_020.off"
# file = "abc_all/meshes/00025728_6fe81cf35e2740b3bbde9aa6_trimesh_038.off"
# file = "abc_all/meshes/00025749_c01a880ceccb4e05908f8255_trimesh_000.off"
# file = "abc_all/meshes/00026055_11394f3abb83419b88f6eed9_trimesh_000.off"
# file = "abc_all/meshes/00026133_8010d7708c06499e9e99ce1c_trimesh_008.off"

# 25611 25633 25710 25728 25749 26055 26133

# file = '00000027.obj'
# file = '00000031.obj'
# file = '00000057.obj'
# file = '00000060.obj'
# file = '00000065.obj'

# file = 'fillet_transition.obj'

obj_name = file.split('/')[-1].split('.')[0]
# obj_name = '1'
# obj_path = "example_data/example_CAD"
# obj_path = "example_data/abc_all/meshes/"
obj_path = "example_data/selected/"
# obj_path = "example_results/merge/selected"
# obj_path = "example_results/merge/selected/example_result/00140223_5374ce893909eddbc7536b71_trimesh_000_our/"
# obj_path = ''
# obj_path = 'D:/dataset/aggregate/ABC/'
# obj_path = "example_results\merge\cad\cad16"
# output_path = "example_results/vsa/CAD/"
# output_path = "example_results/vsa/ABC/"
# output_path = "example_results/merge/selected/"
output_path = "example_results/vsa/selected/"
# output_path = "example_results/vsa/abc_all/"
# output_path = "example_results/merge/selected/example_result/00140223_5374ce893909eddbc7536b71_trimesh_000_our/"
# output_path = "example_results/ablation/presplit/"
# output_path = "example_results/vsa/test/"
# output_path = "example_results/vsa/cad/"


import time
def test_example(v, f):
    min_coord, max_coord = np.min(v, axis=0), np.max(v, axis=0)
    global_diagonal_len = np.linalg.norm(max_coord - min_coord)
    print(f'global diagonal {global_diagonal_len}')
    time_start = time.time()
    proxy_cluster, region_cluster, face_color = pre_split(v, f, visualize=True, visualize_part=False)
    time_end = time.time()
    print(f'pre-split time: {time_end - time_start}s')
    # write_ply_file(v, f, face_color, os.path.join(output_path, obj_name+'_split.ply'))

    # patch2f = []
    # patch2f.append(proxy_cluster[0])
    # other_f = []
    # for i in range(1, len(proxy_cluster)):
    #     other_f.extend(proxy_cluster[i])
    # patch2f.append(other_f)
    # face_color = np.ones((f.shape[0], 3))
    # face_color[patch2f[0]] = colors[0]
    # face_color[patch2f[1]] = colors[1]
    # write_ply_file(v, f, face_color, os.path.join(output_path, obj_name+'_split.ply'))

    return proxy_cluster, region_cluster, face_color


def direct_vsa(v, f):
    proxy_num=60
    time_start = time.time()
    vsa = VSA(v, f, proxy_num=proxy_num)
    _, proxy_cluster = vsa.iteration(visualize=True)
    
    time_end = time.time()
    print(f'direct vsa time: {time_end - time_start}s')
    # clusters = []
    # for proxy_id in proxy_cluster.keys():
    #     f_id = np.array(proxy_cluster[proxy_id])
    #     clusters.append(f_id)
    clusters = [proxy_cluster[key] for key in proxy_cluster.keys()]
    # print(len(clusters))
    face_color = np.ones((f.shape[0], 3))
    for i in range(len(clusters)):
        face_color[clusters[i]] = colors[i%500]
    write_ply_file(v, f, face_color, os.path.join(output_path, obj_name+f'_direct_vsa_{proxy_num}.ply'))

# file_path = os.path.join(obj_path, obj_name+'.obj')
file_path = os.path.join(obj_path, obj_name+'.off')
output_path = os.path.join(output_path, obj_name+'_proxy_cluster.txt')
v, f = igl.read_triangle_mesh(file_path)
# print(v.shape, v.dtype, f.shape, f.dtype)
# print("faces min/max:", f.min(), f.max())
# print("any non-tri?", f.shape[1] != 3)
print(f'load mesh: {obj_name}, #v: {v.shape[0]}, #f: {f.shape[0]}')
proxy_cluster, region_cluster, face_color = test_example(v, f)
# write_proxy_cluster(proxy_cluster, output_path)
# direct_vsa(v, f)
