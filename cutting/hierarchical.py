import time

import meshplot as mp
import numpy as np

from hole_filling.hole_filling import HoleFilling
from patch_graph.patch_mesh import BorderType, InitPatchMesh, PatchMesh

def mesh_segment(v, f, iteration_time, patch2f=None, init_patch_file=None, init_region_file=None):
    print("iteration time: ", iteration_time)
    Objects = []
    
    # hole filling
    mp.plot(v, f, np.ones((f.shape[0], 3)), shading={"wireframe": False})
    do_hole_filling = input("Do hole filling? (y/n):")
    if do_hole_filling == 'y':
        time_start = time.time()
        hole_filling = HoleFilling(v, f, visualize=True)
        new_v, new_f, filled_regions = hole_filling.new_v, hole_filling.new_f, hole_filling.region_triangles
        if len(filled_regions) != 0:
            cur_id = f.shape[0]
            for region in filled_regions:
                patch2f.append(list(np.arange(cur_id, cur_id + region.shape[0])))
                cur_id += len(region)
        v, f = new_v, new_f
        time_end = time.time()
        print(f"[TIME-HOLEFILLING] {time_end - time_start:.4f} seconds")
        mp.plot(v, f, np.ones((f.shape[0], 3)), shading={"wireframe": False})
    
    
    # generate patch or add patch
    time_start = time.time()
    if patch2f is None:
        patch_mesh = InitPatchMesh(v, f, init_patch_file, init_region_file)
    else:
        patch_mesh = PatchMesh(v, f, patch2f)
    time_end = time.time()
    print(f'[TIME-PATCH]: {time_end - time_start:.4f} seconds')
    
    # get patch type
    time_start = time.time()
    border_type = BorderType(patch_mesh)
    time_end1 = time.time()
    # border_type.show_each_patch()
    border_type.show_patch()
    border_type.get_patch_type()
    time_end2 = time.time()
    border_type.get_all_borders()
    time_end = time.time()
    print(f"[TIME-BORDER]: {time_end - time_start:.4f} seconds \n \
        init border_type use {time_end1 - time_start:.4f} seconds, \n \
        get patch type use {time_end2 - time_end1:.4f} seconds, \n \
        get all borders use {time_end - time_end2:.4f} seconds")


    # combine patches
    time_start = time.time()
    border_type.combine_patches_by_fitting(visualize=True)
    time_end1 = time.time()
    border_type.get_all_borders()
    time_end = time.time()
    border_type.show_patch()
    patch2f = border_type.patch_mesh.patch2f
    if iteration_time == 0:
        if patch2f is None:
            patch2f = []
        return [{'v':v, 'f':f, 'patch': patch2f}]
    time_end = time.time()
    print(f"[TIME-COMBINE]: {time_end - time_start:.4f} seconds \n\
        combine patches use {time_end1 - time_start:.4f} seconds, \n\
        get all borders use {time_end - time_end1:.4f} seconds")

    # border_type.show_each_patch()
    # border_type.get_all_borders()

    
    time_start = time.time()
    concave_circle_cut, concave_circle_cut_list = border_type.cut_by_concave_border()
    time_end = time.time()
    print(f"[TIME]:try concave circle time: {time_end - time_start:.4f} seconds")    
    if concave_circle_cut:
        print("find concave circle cut")
        concave_circle_cut_list.reverse()
        for sub_obj in concave_circle_cut_list:
            result = mesh_segment(sub_obj['v'], sub_obj['f'], iteration_time-1, sub_obj['patch'])
            Objects.extend(result)
    else:
        # objs = border_type.cut_by_convex_concave_border()
        # for sub_obj in objs:
        #     result = mesh_segment(sub_obj['v'], sub_obj['f'], iteration_time-1, sub_obj['patch'])
        #     Objects.extend(result)
        # return objs


        concave_curve_cut, concave_curve_cut_list = border_type.cut_by_concave_curve()
        time_end = time.time()
        print(f"[TIME]:try concave curve time: {time_end - time_start:.4f} seconds")
        if concave_curve_cut:
            print("find concave curve cut")
            for sub_obj in concave_curve_cut_list:
                result = mesh_segment(sub_obj['v'], sub_obj['f'], iteration_time-1, sub_obj['patch'])
                Objects.extend(result)
        else:
            print("no concave curve")
            # consider convex cut in object
            return [{'v':v, 'f':f, 'patch': patch2f}]
    return Objects
