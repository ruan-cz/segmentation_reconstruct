"""OpenSCAD serialization for the supported CSG IR."""

import numpy as np

from . import candidates as candidate_module


def _format_vector(values):
    return "[" + ", ".join(f"{float(value):.9g}" for value in values) + "]"


def _format_matrix(values):
    return "[" + ", ".join(_format_vector(row) for row in values) + "]"


def _axis_rotation(axis):
    axis = candidate_module._normalize(axis)
    reference = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(reference, axis), -1.0, 1.0))
    angle = np.rad2deg(np.arccos(dot))
    rotation_axis = np.cross(reference, axis)
    if np.linalg.norm(rotation_axis) <= candidate_module.EPSILON:
        rotation_axis = np.array([1.0, 0.0, 0.0])
    return angle, candidate_module._normalize(rotation_axis)


def _clip_shape(shape, clip_bounds, padding):
    """Wrap a mother surface in an AABB intersection for partial features."""
    if clip_bounds is None:
        return shape
    clip_bounds = np.asarray(clip_bounds, dtype=float)
    if clip_bounds.shape != (2, 3) or not np.all(np.isfinite(clip_bounds)):
        return shape
    center = (clip_bounds[0] + clip_bounds[1]) / 2.0
    size = np.maximum(clip_bounds[1] - clip_bounds[0], 1e-9)
    return (
        f"{padding}intersection() {{\n"
        f"{shape}\n"
        f"{padding}  translate({_format_vector(center)}) "
        f"cube({_format_vector(size)}, center=true);\n"
        f"{padding}}}"
    )


def csg_to_openscad(node, indent=0):
    """Serialize the supported CSG IR recursively as OpenSCAD source."""
    padding = " " * indent
    operation = node["op"]
    if operation == "EMPTY":
        return f"{padding}// empty CSG"
    if operation in {"UNION", "DIFFERENCE", "INTERSECTION"}:
        name = {
            "UNION": "union",
            "DIFFERENCE": "difference",
            "INTERSECTION": "intersection",
        }[operation]
        body = "\n".join(
            csg_to_openscad(child, indent + 2) for child in node["children"]
        )
        return f"{padding}{name}() {{\n{body}\n{padding}}}"
    parameters = node["parameters"]
    if operation == "INNER_TORUS":
        # The concave round is the material inside a short cylinder after
        # removing the torus tube that defines its visible surface.
        cylinder = {
            "op": "CYLINDER",
            "parameters": {
                "axis_point": parameters["center"],
                "axis": parameters["axis"],
                "extent": parameters["extent"],
                "radius": parameters["major_radius"],
            },
        }
        torus = {
            "op": "TORUS",
            "parameters": {
                "center": parameters["center"],
                "axis": parameters["axis"],
                "major_radius": parameters["major_radius"],
                "minor_radius": parameters["minor_radius"],
            },
        }
        return csg_to_openscad(
            {"op": "DIFFERENCE", "children": [cylinder, torus]}, indent=indent
        )
    if operation == "CUBE":
        center = _format_vector(parameters["center"])
        angles = _format_vector(parameters["euler_xyz_degrees"])
        size = _format_vector(parameters["size"])
        base_padding = padding + "  " if parameters.get("clip_bounds") is not None else padding
        base = f"{base_padding}translate({center}) rotate({angles}) cube({size}, center=true);"
        return _clip_shape(base, parameters.get("clip_bounds"), padding)
    if operation == "SPHERE":
        center = _format_vector(parameters["center"])
        base_padding = padding + "  " if parameters.get("clip_bounds") is not None else padding
        base = f"{base_padding}translate({center}) sphere(r={float(parameters['radius']):.9g});"
        return _clip_shape(base, parameters.get("clip_bounds"), padding)
    if operation in {"CYLINDER", "CONE"}:
        axis = np.asarray(parameters["axis"])
        extent = np.asarray(parameters["extent"])
        origin = np.asarray(parameters.get("axis_point", parameters.get("apex")))
        start = origin + axis * extent[0]
        height = float(extent[1] - extent[0])
        angle, rotation_axis = _axis_rotation(axis)
        if operation == "CYLINDER":
            shape = (
                f"cylinder(h={height:.9g}, "
                f"r={float(parameters['radius']):.9g}, center=false);"
            )
        else:
            radius0 = float(max(0.0, extent[0]) * np.tan(parameters["angle"]))
            radius1 = float(max(0.0, extent[1]) * np.tan(parameters["angle"]))
            shape = f"cylinder(h={height:.9g}, r1={radius0:.9g}, r2={radius1:.9g}, center=false);"
        base_padding = padding + "  " if parameters.get("clip_bounds") is not None else padding
        base = (
            f"{base_padding}translate({_format_vector(start)}) "
            f"rotate(a={angle:.9g}, v={_format_vector(rotation_axis)}) {shape}"
        )
        return _clip_shape(base, parameters.get("clip_bounds"), padding)
    if operation == "TORUS":
        angle, rotation_axis = _axis_rotation(parameters["axis"])
        center = _format_vector(parameters["center"])
        major = float(parameters["major_radius"])
        minor = float(parameters["minor_radius"])
        base_padding = padding + "  " if parameters.get("clip_bounds") is not None else padding
        base = (
            f"{base_padding}translate({center}) "
            f"rotate(a={angle:.9g}, v={_format_vector(rotation_axis)}) "
            f"rotate_extrude() translate([{major:.9g}, 0, 0]) circle(r={minor:.9g});"
        )
        return _clip_shape(base, parameters.get("clip_bounds"), padding)
    if operation in {"EXTRUSION", "SPLINE_EXTRUSION"}:
        axis = np.asarray(parameters["axis"], dtype=float)
        origin = np.asarray(parameters["axis_point"], dtype=float)
        extent = np.asarray(parameters["extent"], dtype=float)
        basis = np.asarray(parameters["basis"], dtype=float)
        start = origin + extent[0] * axis
        transform = np.eye(4)
        transform[:3, :2] = basis
        transform[:3, 2] = axis
        transform[:3, 3] = start
        profile = parameters.get("polygon", parameters.get("control_points_2d"))
        polygon = "[" + ", ".join(
            _format_vector(point) for point in profile
        ) + "]"
        height = float(extent[1] - extent[0])
        return (
            f"{padding}multmatrix({_format_matrix(transform)}) "
            f"linear_extrude(height={height:.9g}) polygon(points={polygon});"
        )
    raise ValueError(f"unsupported CSG operation: {operation}")


class OpenSCADExporter:
    """Serialize supported CSG IR nodes to OpenSCAD source."""

    @staticmethod
    def generate(csg_ir, indent=0):
        return csg_to_openscad(csg_ir, indent=indent)


__all__ = ["OpenSCADExporter", "csg_to_openscad"]
