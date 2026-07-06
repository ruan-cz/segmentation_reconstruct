# Pipeline

A comprehensive CAD mesh segmentation framework that combines mesh decomposition, geometric primitive fitting, and hole filling techniques for CAD model processing and analysis.

## Overview

This project implements a complete pipeline for CAD mesh segmentation. The framework integrates multiple techniques to:

1. **Segment** complex 3D meshes into meaningful regions
2. **Fit** geometric primitives to mesh patches
3. **Fill** holes and gaps in segmented meshes
4. **Visualize** results through rendering and analysis tools


## Core Modules

### 1. **Surface Segmentation** - `vsa/`
Segment mesh into meaningful patches based on surface properties.

**Key Classes:**
- `VSA`: Main segmentation engine
- `Partition`: Manages patch-based decomposition
- `Proxy`: Represents geometric proxies for each patch
- `Distance`: Computes distance metrics between surfaces

**Functions:**
- Lloyd-like iteration for proxy optimization
- Patch boundary detection
- Geometric error computation

### 2. **Geometric Primitive Fitting** - `fitting_geometric_primitives/`
Fit canonical geometric shapes to mesh patches.

**Supported Primitives:**
- Planes
- Cylinders
- Cones
- Spheres
- Torus

**Key Files:**
- `primitive_fitting.py`: Main fitting algorithms
- `geometry_primitive.py`: Primitive shape definitions
- `primitives_2d.py`: 2D shape analysis
- Jupyter notebooks for interactive fitting and testing

### 3. **Mesh Cutting** - `cut/`
Advanced mesh decomposition using various cutting techniques.

**Cutting Methods:**

#### Graph Cut (`GraphCut`)
- Separates mesh along concave boundaries
- Uses graph connectivity analysis
- Removes edges marked as borders (border_type = 1)

#### Concave Curve Cutting (`ConcaveCurveCut`)
- Aggregates concave edges from patch graph
- Connects fragmented borders into continuous lines
- Interactive surface selection
- Delegates to appropriate cutting method

#### Plane-based Cutting (`PlaneCut`)
- Cuts mesh using plane equations
- Computes edge-plane intersections
- Triangulates cut faces
- Splits mesh into two parts relative to plane

#### Cylinder-based Cutting (`CylinderCut`)
- Cuts mesh using cylinder surfaces
- Parameters: center, axis direction, radius
- Handles perpendicular distance calculation
- Applies threshold-based face classification

#### Tracing Cuts (`PlaneTracingCut`, `CylinderTracingCut`)
- Traces cutting path through mesh faces
- Maintains continuity along cutting surface
- Supports vertex-to-vertex and edge-based transitions
- Generates smooth boundaries



### 4. **Hole Filling** - `hole_filling/`
Reconstructs missing surface regions and fills discontinuities.

### 5. **Geometry Utilities** - `geometry_utils/`
Low-level geometric operations and computations.


### 6. **Triangulation** - `triangulate/`
Mesh generation and refinement.


### 7. **Cutting Calculations** - `cut_calculate/`
Numerical computations for cutting operations.


### 8. **Rendering & Visualization** - `render/`
VTK-based 3D visualization and rendering.


### 9. **Data Processing** - `data_process/`
Input/output and data format handling.


### 10. **Utilities** - `utils/`
Common utility functions used across modules.


### 11. **Global Configuration** - `global/`
Project-wide settings and constants.



## Dependencies

### Core Libraries
```
numpy           - Numerical computations
networkx        - Graph analysis
openmesh        - Mesh data structures
igl             - Geometry processing
meshplot        - Interactive visualization
scipy           - Scientific computing
scikit-learn    - Machine learning utilities
matplotlib      - 2D plotting
```

### Optional
```
vtk             - Advanced rendering
distinctipy     - Color generation
```

