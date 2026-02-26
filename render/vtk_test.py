# pip install vtk
import numpy as np
import vtk
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderWindow
from vtkmodules.vtkInteractionStyle import *
import igl
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *

file = open('/Users/ruan/Desktop/MeshSegment/global/colors_500.txt', 'r')
lines = file.readlines()
colors = np.zeros((len(lines),3))
for i in range(len(lines)):
    colors[i] = np.array(lines[i].split())
# color: transform from [0, 1] to int [0, 255]
colors = (colors * 255).astype(np.uint8)
colors.clip(0, 255, out=colors)



obj_file = '/Users/ruan/Desktop/MeshSegment/example_data/example_CAD/2.obj'
patch_file = '/Users/ruan/Desktop/MeshSegment/example_results/vsa/2_merged_proxy_cluster.txt'

v, f = igl.read_triangle_mesh(obj_file)
proxy2f = read_patch_cluster(patch_file)

# col_v = np.ones((v.shape[0], 3))
col_f = np.zeros((f.shape[0], 3), dtype=np.uint8)
for i in range(len(proxy2f)):
    col_f[proxy2f[i]] = colors[i]



# ------------------------------------------------------------
# 1. Cube geometry: 8 vertices, 12 triangles, 12 edges
# ------------------------------------------------------------
# v = np.array([
#     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
#     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
# ], float)

# f = np.array([
#     [0, 1, 2], [0, 2, 3],   # bottom
#     [4, 5, 6], [4, 6, 7],   # top
#     [0, 1, 5], [0, 5, 4],   # front
#     [2, 3, 7], [2, 7, 6],   # back
#     [1, 2, 6], [1, 6, 5],   # right
#     [3, 0, 4], [3, 4, 7]    # left
# ], int)

# E = np.array([
#     [0,1],[1,2],[2,3],[3,0],
#     [4,5],[5,6],[6,7],[7,4],
#     [0,4],[1,5],[2,6],[3,7]
# ], int)

# ------------------------------------------------------------
# 2. Define colors
# ------------------------------------------------------------
# col_v = np.array([
#     [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
#     [255, 0, 255], [0, 255, 255], [255, 128, 0], [255, 255, 255]
# ], np.uint8)

# col_f = np.array([
#     [255, 0, 0], [255, 0, 0],
#     [0, 255, 0], [0, 255, 0],
#     [0, 0, 255], [0, 0, 255],
#     [255, 255, 0], [255, 255, 0],
#     [255, 0, 255], [255, 0, 255],
#     [0, 255, 255], [0, 255, 255]
# ], np.uint8)

# col_e = np.array([
#     [255, 255, 255] if i < 4 else
#     [255, 0, 255] if i < 8 else
#     [0, 0, 0] for i in range(len(E))
# ], np.uint8)

# ------------------------------------------------------------
# 3. Build VTK PolyData for cube faces
# ------------------------------------------------------------
points = vtk.vtkPoints()
points.SetNumberOfPoints(len(v))
for i, p in enumerate(v):
    points.SetPoint(i, *p)

polys = vtk.vtkCellArray()
for tri in f:
    cell = vtk.vtkTriangle()
    for j in range(3):
        cell.GetPointIds().SetId(j, int(tri[j]))
    polys.InsertNextCell(cell)

mesh_pd = vtk.vtkPolyData()
mesh_pd.SetPoints(points)
mesh_pd.SetPolys(polys)

# vertex colors
# vcols = vtk.vtkUnsignedCharArray()
# vcols.SetName("VertexColors")
# vcols.SetNumberOfComponents(3)
# for c in col_v:
#     vcols.InsertNextTuple3(*map(int, c))
# mesh_pd.GetPointData().SetScalars(vcols)

# face colors
fcols = vtk.vtkUnsignedCharArray()
fcols.SetName("FaceColors")
fcols.SetNumberOfComponents(3)
for c in col_f:
    fcols.InsertNextTuple3(*map(int, c))
mesh_pd.GetCellData().SetScalars(fcols)
mesh_pd.GetCellData().SetActiveScalars("FaceColors")

# ------------------------------------------------------------
# 4. Build colored edges as tubes
# ------------------------------------------------------------
# lines = vtk.vtkCellArray()
# for e in E:
#     line = vtk.vtkLine()
#     line.GetPointIds().SetId(0, int(e[0]))
#     line.GetPointIds().SetId(1, int(e[1]))
#     lines.InsertNextCell(line)

# edge_pd = vtk.vtkPolyData()
# edge_pd.SetPoints(points)
# edge_pd.SetLines(lines)

# ecols = vtk.vtkUnsignedCharArray()
# ecols.SetName("EdgeColors")
# ecols.SetNumberOfComponents(3)
# for c in col_e:
#     ecols.InsertNextTuple3(*map(int, c))
# edge_pd.GetCellData().SetScalars(ecols)

# tube = vtk.vtkTubeFilter()
# tube.SetInputData(edge_pd)
# tube.SetRadius(0.02)
# tube.SetNumberOfSides(12)
# tube.CappingOn()
# tube.Update()

# ------------------------------------------------------------
# 5. Write .vtp files
# ------------------------------------------------------------
# w = vtk.vtkXMLPolyDataWriter()
# w.SetFileName("cube_faces.vtp")
# w.SetInputData(mesh_pd)
# w.Write()

# w2 = vtk.vtkXMLPolyDataWriter()
# w2.SetFileName("cube_edges.vtp")
# w2.SetInputData(tube.GetOutput())
# w2.Write()
# print("Wrote cube_faces.vtp and cube_edges.vtp")

# ------------------------------------------------------------
# 6. Render to PNG
# ------------------------------------------------------------
mesh_mapper = vtk.vtkPolyDataMapper()
mesh_mapper.SetInputData(mesh_pd)
mesh_mapper.SetScalarModeToUseCellFieldData()
mesh_mapper.SelectColorArray("FaceColors")
mesh_mapper.SetColorModeToDirectScalars()
mesh_mapper.ScalarVisibilityOn()

mesh_actor = vtk.vtkActor()
mesh_actor.SetMapper(mesh_mapper)

property = mesh_actor.GetProperty()
property.LightingOff()

# edge_mapper = vtk.vtkPolyDataMapper()
# edge_mapper.SetInputConnection(tube.GetOutputPort())
# edge_mapper.SetScalarModeToUseCellData()
# edge_mapper.ScalarVisibilityOn()

# edge_actor = vtk.vtkActor()
# edge_actor.SetMapper(edge_mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(mesh_actor)
# renderer.AddActor(edge_actor)
renderer.SetBackground(1,1,1)

renwin = vtk.vtkRenderWindow()
renwin.SetOffScreenRendering(1)
renwin.AddRenderer(renderer)

W, H = 800, 600
scale = 5
renwin.SetSize(W * scale, H * scale)



renderer.ResetCamera()
renderer.GetActiveCamera().Azimuth(55)
renderer.GetActiveCamera().Elevation(120)
renderer.ResetCameraClippingRange()
renwin.Render()

w2i = vtk.vtkWindowToImageFilter()
w2i.SetInput(renwin)
# w2i.SetScale(scale)
w2i.SetInputBufferTypeToRGB()
w2i.ReadFrontBufferOff()
w2i.Update()


# resample = vtk.vtkImageResample()
# resample.SetInputConnection(w2i.GetOutputPort())
# resample.SetAxisMagnificationFactor(0, 1.0/scale)
# resample.SetAxisMagnificationFactor(1, 1.0/scale)
# resample.SetAxisMagnificationFactor(2, 1.0)
# resample.InterpolateOn()
# resample.SetInterpolationModeToNearestNeighbor()
# resample.SetOutputDimensionality(W, H, 1)
# resample.Update()

png = vtk.vtkPNGWriter()
png.SetFileName("cube_render.png")
# png.SetInputConnection(resample.GetOutputPort())
png.SetInputConnection(w2i.GetOutputPort())
png.Write()

print("Wrote cube_render.png")
