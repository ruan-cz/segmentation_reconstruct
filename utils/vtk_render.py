import numpy as np
import vtk
import igl
import os
import sys
from lxml import etree
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mesh_utils import *



def get_viewport(xml_path):
    x = etree.parse(xml_path)
    camera = x.find('.//VCGCamera')
    vp_px = (int(s) for s in camera.get('ViewportPx').split())
    return vp_px

def get_meshlab_camera(renderer, xml_path):
    # read
    x = etree.parse(xml_path)
    camera = x.find('.//VCGCamera')
    view_settings = x.find('.//ViewSettings')

    track_scale = float(view_settings.get('TrackScale'))

    near = float(view_settings.get('NearPlane'))
    far  = float(view_settings.get('FarPlane'))

    rot = [float(s) for s in camera.get('RotationMatrix').split()]
    R4 = np.array(rot).reshape(4, 4)
    R3 = R4[:3, :3]

    trans = [float(s) for s in camera.get('TranslationVector').split()]
    trans = np.array(trans)[:3]

    transform_mat = np.eye(4)
    transform_mat[:3, :3] = R3
    transform_mat[:3, 3] = trans
    transform_mat = np.linalg.inv(transform_mat)

    vp_px = (int(s) for s in camera.get('ViewportPx').split())
    center_px = (int(s) for s in camera.get('CenterPx').split())

    focal = float(camera.get('FocalMm'))
    pixel_size= [float(s) for s in camera.get('PixelSizeMm').split()][0]



    # transform
    pos = transform_mat[:3, 3]
    forward = transform_mat[:3, 2]
    rot = transform_mat[:3, :3]
    view_up = transform_mat[:3, 1]

    # set
    # camera = vtk.vtkCamera()
    camera = renderer.GetActiveCamera()
    camera.SetPosition(*pos)
    camera.SetFocalPoint(*(pos + forward))
    camera.SetViewUp(*view_up)


    return camera
    

def render_mesh_vtk(v, f,
    write_img=True, output_path=None, 
    draw_v=False, _v_color=None, 
    draw_f=False, _f_color=None, face_opacity=1.0,
    e=None, draw_e=False, _e_color=None, edge_radius=0.1, edge_prism_num=12,
    background=[1, 1, 1], 
    xml_path=None,
    scale=5,
    interactive=False,
    draw_p=False, _p=None, _p_color=None, _p_size=0.5, 
    
    ):
    
    # build mesh
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(v.shape[0])
    for i, p in enumerate(v):
        points.SetPoint(i, *p)

    polys = vtk.vtkCellArray()
    polys.SetNumberOfCells(f.shape[0])
    for i, tri in enumerate(f):
        cell = vtk.vtkTriangle()
        for j in range(3):
            cell.GetPointIds().SetId(j, int(tri[j]))
        polys.InsertNextCell(cell)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(polys)

    if draw_v:
        v_colors = vtk.vtkUnsignedCharArray()
        v_colors.SetName('VertexColors')
        v_colors.SetNumberOfComponents(3)
        for c in _v_color:
            c = np.clip(c * 255, 0, 255)
            v_colors.InsertNextTuple3(*map(int, c))
        mesh.GetPointData().SetScalars(v_colors)
        mesh.GetPointData().SetActiveScalars("VertexColors")

    if draw_f:
        f_colors = vtk.vtkUnsignedCharArray()
        f_colors.SetName('FaceColors')
        f_colors.SetNumberOfComponents(3)
        for c in _f_color:
            c = np.clip(c * 255, 0, 255)
            f_colors.InsertNextTuple3(*map(int, c))
        mesh.GetCellData().SetScalars(f_colors)
        mesh.GetCellData().SetActiveScalars("FaceColors")

    if draw_e:
        lines = vtk.vtkCellArray()
        for _e in e:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, int(_e[0]))
            line.GetPointIds().SetId(1, int(_e[1]))
            lines.InsertNextCell(line)
        
        edge = vtk.vtkPolyData()
        edge.SetPoints(points)
        edge.SetLines(lines)

        e_colors = vtk.vtkUnsignedCharArray()
        e_colors.SetName('EdgeColors')
        e_colors.SetNumberOfComponents(3)
        for c in _e_color:
            c = np.clip(c * 255, 0, 255)
            # print(c)
            e_colors.InsertNextTuple3(*map(int, c))
        edge.GetCellData().SetScalars(e_colors)
        
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(edge)
        tube.SetRadius(edge_radius)
        tube.SetNumberOfSides(edge_prism_num)
        tube.CappingOn()
        tube.Update()
    

    if draw_p:
        def make_sphere(center):
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*center)
            sphere.SetRadius(_p_size)
            sphere.SetThetaResolution(32)
            sphere.SetPhiResolution(32)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*_p_color)
            actor.GetProperty().SetSpecular(0.0)
            return actor
        p_actors = []
        for tmp_p in _p:
            p_actor = make_sphere(tmp_p)
            p_actors.append(p_actor)


    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_mapper.SetInputData(mesh)
    mesh_mapper.SetScalarModeToUseCellData()
    mesh_mapper.SetColorModeToDirectScalars()
    mesh_mapper.ScalarVisibilityOn()
    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)
    mesh_actor.GetProperty().SetOpacity(face_opacity)
    mesh_actor.GetProperty().SetInterpolationToFlat()



    # render
    renderer = vtk.vtkRenderer()
    renderer.AddActor(mesh_actor)
    if draw_e:
        edge_mapper = vtk.vtkPolyDataMapper()
        edge_mapper.SetInputConnection(tube.GetOutputPort())
        edge_mapper.SetScalarModeToUseCellData()
        edge_mapper.SetColorModeToDirectScalars()
        edge_mapper.ScalarVisibilityOn()
        edge_actor = vtk.vtkActor()
        edge_actor.SetMapper(edge_mapper)
        renderer.AddActor(edge_actor)
    if draw_p:
        for p_actor in p_actors:
            renderer.AddActor(p_actor)
    renderer.SetBackground(background[0], background[1], background[2])
    


    render_window = vtk.vtkRenderWindow()
    W, H = get_viewport(xml_path)
    # print(W, H)
    render_window.SetSize(W*scale, H*scale)
    render_window.AddRenderer(renderer)
    render_window.SetMultiSamples(10)
    # render_window.SetOffScreenRendering(not interactive)

    # camera setting
    # camera = get_meshlab_camera(renderer, xml_path)
    # read
    x = etree.parse(xml_path)
    camera = x.find('.//VCGCamera')
    view_settings = x.find('.//ViewSettings')
    track_scale = float(view_settings.get('TrackScale'))
    near = float(view_settings.get('NearPlane'))
    far  = float(view_settings.get('FarPlane'))
    rot = [float(s) for s in camera.get('RotationMatrix').split()]
    R4 = np.array(rot).reshape(4, 4)
    R3 = R4[:3, :3]
    trans = [float(s) for s in camera.get('TranslationVector').split()]
    trans = np.array(trans)[:3]
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = R3
    transform_mat[:3, 3] = trans
    transform_mat = np.linalg.inv(transform_mat)


    # transform
    rot = transform_mat[:3, :3]
    view_up = transform_mat[:3, 1]
    forward = transform_mat[:3, 2]

    # set
    camera = renderer.GetActiveCamera()
    camera.SetViewUp(*view_up)
    camera.SetClippingRange(near / track_scale, far / track_scale * 30)
    bounds = mesh.GetBounds()
    center = [(bounds[i*2] + bounds[i*2+1]) / 2 for i in range(3)]
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + forward / track_scale * 15))
    
    renderer.ResetCameraClippingRange()
    render_window.Render()
    
    if write_img:
        win2img = vtk.vtkWindowToImageFilter()
        win2img.SetInput(render_window)
        win2img.Update()
        png = vtk.vtkPNGWriter()
        png.SetFileName(output_path)
        png.SetInputConnection(win2img.GetOutputPort())
        png.Write()
        print(output_path)
        
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()
    interactor.Start()


def render_transparent_mesh_vtk(v, f, output_path, e=None,
    draw_v=False, _v_color=None, 
    draw_f=False, _f_color=None, face_opacity=1.0,
    draw_e=False, _e_color=None, edge_radius=0.1, edge_prism_num=12,
    background=[1, 1, 1], 
    xml_path=None,
    transparent_face=None,
    scale=5,
    interactive=False,
    draw_p=False, _p=None, _p_size=0.5
    ):
    
    # build mesh
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(v.shape[0])
    for i, p in enumerate(v):
        points.SetPoint(i, *p)

    polys_opaque = vtk.vtkCellArray()
    polys_trans = vtk.vtkCellArray()
    # polys.SetNumberOfCells(f.shape[0])
    for i, tri in enumerate(f):
        cell = vtk.vtkTriangle()
        for j in range(3):
            cell.GetPointIds().SetId(j, int(tri[j]))
        if i in transparent_face:    
            polys_trans.InsertNextCell(cell)
        else:
            polys_opaque.InsertNextCell(cell)
        

    mesh_opaque = vtk.vtkPolyData()
    mesh_opaque.SetPoints(points)
    mesh_opaque.SetPolys(polys_opaque)
    mesh_trans = vtk.vtkPolyData()
    mesh_trans.SetPoints(points)
    mesh_trans.SetPolys(polys_trans)

    # if draw_v:
    #     v_colors = vtk.vtkUnsignedCharArray()
    #     v_colors.SetName('VertexColors')
    #     v_colors.SetNumberOfComponents(3)
    #     for c in _v_color:
    #         c = np.clip(c * 255, 0, 255)
    #         v_colors.InsertNextTuple3(*map(int, c))
    #     mesh_opaque.GetPointData().SetScalars(v_colors)
    #     mesh_opaque.GetPointData().SetActiveScalars("VertexColors")

    # if draw_f:
    #     f_colors = vtk.vtkUnsignedCharArray()
    #     f_colors.SetName('FaceColors')
    #     f_colors.SetNumberOfComponents(3)
    #     for c in _f_color:
    #         c = np.clip(c * 255, 0, 255)
    #         f_colors.InsertNextTuple3(*map(int, c))
    #     mesh_opaque.GetCellData().SetScalars(f_colors)
    #     mesh_opaque.GetCellData().SetActiveScalars("FaceColors")

    if draw_e:
        lines = vtk.vtkCellArray()
        for _e in e:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, int(_e[0]))
            line.GetPointIds().SetId(1, int(_e[1]))
            lines.InsertNextCell(line)
        
        edge = vtk.vtkPolyData()
        edge.SetPoints(points)
        edge.SetLines(lines)

        e_colors = vtk.vtkUnsignedCharArray()
        e_colors.SetName('EdgeColors')
        e_colors.SetNumberOfComponents(3)
        for c in _e_color:
            c = np.clip(c * 255, 0, 255)
            e_colors.InsertNextTuple3(*map(int, c))
        edge.GetCellData().SetScalars(e_colors)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(edge)
        tube.SetRadius(edge_radius)
        tube.SetNumberOfSides(edge_prism_num)
        tube.CappingOn()
        tube.Update()
    

    if draw_p:
        def make_sphere(center, p_radius=0.1, p_color=colors[92]):
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*center)
            sphere.SetRadius(p_radius)
            sphere.SetThetaResolution(32)
            sphere.SetPhiResolution(32)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*p_color)
            actor.GetProperty().SetSpecular(0.0)
            return actor
        p_actors = []
        for tmp_p in _p:
            p_actor = make_sphere(tmp_p, p_radius=_p_size)
            p_actors.append(p_actor)


    mesh_opaque_mapper = vtk.vtkPolyDataMapper()
    mesh_opaque_mapper.SetInputData(mesh_opaque)
    mesh_opaque_mapper.SetScalarModeToUseCellData()
    mesh_opaque_mapper.SetColorModeToDirectScalars()
    mesh_opaque_mapper.ScalarVisibilityOn()
    mesh_opaque_actor = vtk.vtkActor()
    mesh_opaque_actor.SetMapper(mesh_opaque_mapper)
    mesh_opaque_actor.GetProperty().SetOpacity(1.0)
    mesh_opaque_actor.GetProperty().SetInterpolationToFlat()
    
    mesh_trans_mapper = vtk.vtkPolyDataMapper()
    mesh_trans_mapper.SetInputData(mesh_trans)
    mesh_trans_mapper.SetScalarModeToUseCellData()
    mesh_trans_mapper.SetColorModeToDirectScalars()
    mesh_trans_mapper.ScalarVisibilityOn()
    mesh_trans_actor = vtk.vtkActor()
    mesh_trans_actor.SetMapper(mesh_trans_mapper)
    mesh_trans_actor.GetProperty().SetOpacity(face_opacity)
    mesh_trans_actor.GetProperty().SetInterpolationToFlat()
    


    # render
    renderer = vtk.vtkRenderer()
    renderer.AddActor(mesh_opaque_actor)
    renderer.AddActor(mesh_trans_actor)
    if draw_e:
        edge_mapper = vtk.vtkPolyDataMapper()
        edge_mapper.SetInputConnection(tube.GetOutputPort())
        edge_mapper.SetScalarModeToUseCellData()
        edge_mapper.SetColorModeToDirectScalars()
        edge_mapper.ScalarVisibilityOn()
        edge_actor = vtk.vtkActor()
        edge_actor.SetMapper(edge_mapper)
        renderer.AddActor(edge_actor)
    if draw_p:
        for p_actor in p_actors:
            renderer.AddActor(p_actor)
    renderer.SetBackground(background[0], background[1], background[2])
    


    render_window = vtk.vtkRenderWindow()
    W, H = get_viewport(xml_path)
    print(W, H)
    render_window.SetSize(W*scale, H*scale)
    render_window.AddRenderer(renderer)
    render_window.SetMultiSamples(10)

    # camera setting
    # read
    x = etree.parse(xml_path)
    camera = x.find('.//VCGCamera')
    view_settings = x.find('.//ViewSettings')
    track_scale = float(view_settings.get('TrackScale'))
    near = float(view_settings.get('NearPlane'))
    far  = float(view_settings.get('FarPlane'))
    rot = [float(s) for s in camera.get('RotationMatrix').split()]
    R4 = np.array(rot).reshape(4, 4)
    R3 = R4[:3, :3]
    trans = [float(s) for s in camera.get('TranslationVector').split()]
    trans = np.array(trans)[:3]
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = R3
    transform_mat[:3, 3] = trans
    transform_mat = np.linalg.inv(transform_mat)


    # transform
    rot = transform_mat[:3, :3]
    view_up = transform_mat[:3, 1]
    forward = transform_mat[:3, 2]

    # set
    camera = renderer.GetActiveCamera()
    camera.SetViewUp(*view_up)
    camera.SetClippingRange(near / track_scale, far / track_scale * 20)
    bounds = mesh_opaque.GetBounds()
    center = [(bounds[i*2] + bounds[i*2+1]) / 2 for i in range(3)]    
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + forward / track_scale * 5))


    
    renderer.ResetCameraClippingRange()

    render_window.Render()

    win2img = vtk.vtkWindowToImageFilter()
    win2img.SetInput(render_window)
    win2img.Update()
    png = vtk.vtkPNGWriter()
    png.SetFileName(output_path)
    png.SetInputConnection(win2img.GetOutputPort())
    png.Write()
    print(output_path)
    

if __name__ == '__main__':
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v, f = igl.read_triangle_mesh(os.path.join(repo_root, "example_data/example_CAD/2.obj"))
    proxy2f = read_patch_cluster(os.path.join(repo_root, "example_results/vsa/2_merged_proxy_cluster.txt"))
    color500_file = os.path.join(repo_root, "utils/colors_500.txt")
    file = open(color500_file, 'r')
    lines = file.readlines()
    colors = np.zeros((len(lines),3))
    for i in range(len(lines)):
        colors[i] = np.array(lines[i].split())

    f_colors = np.zeros((f.shape[0],3))
    for i in range(len(proxy2f)):
        f_colors[proxy2f[i]] = colors[i]


    render_mesh_vtk(
        v, f, output_path='test_render_2.png',
        draw_f=True, _f_color=f_colors,
        interactive=True,
        xml_path='render/2_test.xml'
    )