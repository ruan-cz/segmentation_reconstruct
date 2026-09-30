"""Build fitted surface patches and their shared-boundary graph.

The segmented PLY stores a patch label in each face's RGB color.  This module
turns those labels into local ``trimesh`` patch meshes, fits primitive types,
and records adjacency in the original mesh coordinate/index space.
"""

from pathlib import Path

import numpy as np
import trimesh

from primitive_fitting import PrimitiveFitting


def _load_triangle_mesh(mesh_or_path):
    """Load one triangle mesh without changing its indices."""
    if isinstance(mesh_or_path, trimesh.Trimesh):
        mesh = mesh_or_path
    else:
        path = Path(mesh_or_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh = trimesh.load_mesh(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError("mesh scene contains no geometry")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("expected a trimesh.Trimesh or a path to one")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("segmented PLY must contain triangular faces")
    return mesh


def _face_labels(mesh):
    """Return the RGB segmentation label for every mesh face."""
    colors = np.asarray(mesh.visual.face_colors)
    if colors.ndim != 2 or len(colors) != len(mesh.faces) or colors.shape[1] < 3:
        raise ValueError("segmented PLY must provide one RGB color per face")
    return np.ascontiguousarray(colors[:, :3], dtype=np.uint8)


def _extract_segment_patches(mesh, min_faces=1, merge_vertices=False):
    labels = _face_labels(mesh)
    patches = []

    for label in np.unique(labels, axis=0):
        face_ids = np.flatnonzero(np.all(labels == label, axis=1))
        if len(face_ids) < min_faces:
            continue

        global_faces = np.asarray(mesh.faces[face_ids], dtype=np.int64)
        vertex_ids, inverse = np.unique(global_faces.reshape(-1), return_inverse=True)
        patch = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices[vertex_ids], dtype=float),
            faces=inverse.reshape((-1, 3)),
            process=merge_vertices,
        )
        patch.metadata.update(
            {
                "patch_index": len(patches),
                "segment_label": tuple(int(value) for value in label),
                "source_face_ids": face_ids.copy(),
                "face_id": face_ids.copy(),
            }
        )
        patches.append(patch)

    return patches


def load_segment_patches(ply_path, min_faces=1, merge_vertices=False):
    """Read face-color segments from a PLY file as an ordered patch list."""
    if min_faces < 1:
        raise ValueError("min_faces must be at least 1")
    mesh = _load_triangle_mesh(ply_path)
    return _extract_segment_patches(mesh, min_faces, merge_vertices)


def _fit_residual_errors(points, fits):
    """Compute normalized RMS residuals of each analytic surface fit.

    Thin annular patches (screw threads) reach ``fit_rate == 1.0`` for almost
    every primitive because the loose absolute inlier thresholds cannot
    discriminate them.  The residual error still can: a helical/conical
    flank deviates measurably from its best plane, while a true cone or
    cylinder ring fits with a nearly zero residual.
    """
    points = np.asarray(points, dtype=float)
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-12)
    errors = {}

    def params_of(name, size):
        parameters = np.asarray(
            fits.get(name, {}).get("params", []), dtype=float
        ).reshape(-1)
        return parameters if parameters.size >= size else None

    plane = params_of("plane", 4)
    if plane is not None and np.linalg.norm(plane[:3]) > 1e-12:
        residual = (points @ plane[:3] + plane[3]) / np.linalg.norm(plane[:3])
        errors["plane"] = float(np.sqrt(np.mean(residual**2)) / scale)
    cylinder = params_of("cylinder", 7)
    if cylinder is not None and np.linalg.norm(cylinder[3:6]) > 1e-12:
        axis = cylinder[3:6] / np.linalg.norm(cylinder[3:6])
        radial = np.linalg.norm(
            np.cross(points - cylinder[:3], axis), axis=1
        )
        residual = radial - abs(float(cylinder[6]))
        errors["cylinder"] = float(np.sqrt(np.mean(residual**2)) / scale)
    sphere = params_of("sphere", 4)
    if sphere is not None:
        residual = np.linalg.norm(points - sphere[:3], axis=1) - abs(float(sphere[3]))
        errors["sphere"] = float(np.sqrt(np.mean(residual**2)) / scale)
    cone = params_of("cone", 7)
    if cone is not None and np.linalg.norm(cone[3:6]) > 1e-12:
        axis = cone[3:6] / np.linalg.norm(cone[3:6])
        vectors = points - cone[:3]
        axial = vectors @ axis
        radial = np.linalg.norm(vectors - axial[:, None] * axis, axis=1)
        residual = radial - axial * np.tan(cone[6])
        errors["cone"] = float(np.sqrt(np.mean(residual**2)) / scale)
    return errors


def _select_best_fit(fits, error_ratio=0.5):
    """Pick the patch type: highest fit rate, ties broken by residual error.

    Without the residual tie-break, degenerate thin rings (where every
    primitive reports ``fit_rate == 1.0``) are always classified as planes
    by insertion order, hiding conical thread flanks from the curved
    candidate generators.  A different type only takes over when its
    residual is dramatically smaller (``error_ratio``) so clean planes keep
    their type.
    """
    best_type = max(fits, key=lambda name: fits[name]["fit_rate"])
    errors = {
        name: fit["rms_error"]
        for name, fit in fits.items()
        if np.isfinite(fit.get("rms_error", np.nan))
    }
    best_rate = fits[best_type]["fit_rate"]
    reference = errors.get(best_type)
    if reference is None:
        return best_type
    tied = [
        name
        for name in errors
        if fits[name]["fit_rate"] >= best_rate - 1e-9
    ]
    if not tied:
        return best_type
    winner = min(tied, key=lambda name: errors[name])
    if winner != best_type and errors[winner] < error_ratio * reference:
        return winner
    return best_type


def fit_patch_primitives(
    patch,
    include_cone=True,
    include_extrusion=True,
    include_spline_extrusion=True,
):
    """Fit all enabled primitive types and return the best patch hypothesis."""
    empty_result = {
        "type": "invalid",
        "fit_rate": 0.0,
        "params": np.array([], dtype=float),
        "fits": {},
    }
    if len(patch.faces) == 0 or len(patch.vertices) < 3:
        return empty_result

    face_areas = np.asarray(patch.area_faces, dtype=float).reshape(-1)
    if len(face_areas) == 0 or not np.any(face_areas > 1e-12):
        return empty_result

    fitter = PrimitiveFitting(
        np.asarray(patch.vertices, dtype=float),
        np.asarray(patch.faces, dtype=np.int64),
    )
    face_ids = np.arange(len(patch.faces), dtype=np.int64)
    methods = {
        "plane": fitter.fit_planar,
        "cylinder": fitter.fit_cylinder,
        "sphere": fitter.fit_sphere,
    }
    if include_cone:
        methods["cone"] = fitter.fit_cone
    if include_extrusion:
        methods["extrusion"] = fitter.fit_extrusion
    if include_spline_extrusion:
        methods["spline_extrusion"] = fitter.fit_spline_extrusion

    fits = {}
    for primitive_type, method in methods.items():
        try:
            fit_rate, parameters = method(face_ids, is_v_id=False)
            fit_rate = float(np.asarray(fit_rate).squeeze())
            parameters = np.asarray(parameters, dtype=float).reshape(-1)
            if not np.isfinite(fit_rate) or not np.all(np.isfinite(parameters)):
                raise ValueError("primitive fitter returned a non-finite result")
            fits[primitive_type] = {
                "fit_rate": fit_rate,
                "params": parameters,
            }
        except Exception as error:
            fits[primitive_type] = {
                "fit_rate": 0.0,
                "params": np.array([], dtype=float),
                "error": repr(error),
            }

    for name, error in _fit_residual_errors(patch.vertices, fits).items():
        fits[name]["rms_error"] = error
    best_type = _select_best_fit(fits)
    return {
        "type": best_type,
        "fit_rate": fits[best_type]["fit_rate"],
        "params": fits[best_type]["params"],
        "fits": fits,
    }


def identify_patches(patches, **fit_options):
    """Fit every patch while preserving list and original-face indices."""
    results = []
    for patch_index, patch in enumerate(patches):
        result = fit_patch_primitives(patch, **fit_options)
        result.update(
            {
                "patch_index": patch_index,
                "segment_label": patch.metadata.get("segment_label"),
                "face_id": np.asarray(
                    patch.metadata.get("face_id", []), dtype=np.int64
                ),
                "mesh": patch,
            }
        )
        results.append(result)
    return results


def _enforce_model_fit_bounds(mesh, results, maximum_sphere_radius_ratio=1.0):
    """Reject sphere fits whose radius is implausible for the input model."""
    model_scale = max(float(np.linalg.norm(np.ptp(mesh.vertices, axis=0))), 1e-12)
    maximum_radius = float(maximum_sphere_radius_ratio) * model_scale
    for result in results:
        sphere_fit = result.get("fits", {}).get("sphere")
        if not sphere_fit:
            continue
        parameters = np.asarray(sphere_fit.get("params", []), dtype=float).reshape(-1)
        if parameters.size < 4:
            continue
        radius = abs(float(parameters[3]))
        if np.isfinite(radius) and radius <= maximum_radius:
            continue
        sphere_fit["fit_rate"] = 0.0
        sphere_fit["rejected_reason"] = "radius_exceeds_model_scale"
        sphere_fit["maximum_radius"] = maximum_radius
        if str(result.get("type", "")).lower() != "sphere":
            continue
        valid_fits = {
            name: fit
            for name, fit in result.get("fits", {}).items()
            if np.asarray(fit.get("params", [])).size > 0
        }
        if not valid_fits:
            result.update(type="invalid", fit_rate=0.0, params=np.array([], dtype=float))
            continue
        best_type = _select_best_fit(valid_fits)
        result.update(
            type=best_type,
            fit_rate=float(valid_fits[best_type]["fit_rate"]),
            params=np.asarray(valid_fits[best_type]["params"], dtype=float),
        )
    return results


def _patch_records(mesh, patches):
    records = []
    for patch_index, patch in enumerate(patches):
        face_ids = np.asarray(patch.metadata.get("face_id", []), dtype=np.int64)
        if np.any(face_ids < 0) or np.any(face_ids >= len(mesh.faces)):
            raise ValueError(f"patch {patch_index} contains invalid face_id values")
        records.append(
            {
                "patch_index": patch_index,
                "segment_label": patch.metadata.get("segment_label"),
                "face_id": face_ids.copy(),
                "v": np.asarray(patch.vertices, dtype=float),
                "f": np.asarray(patch.faces, dtype=np.int64),
                "parameter": None,
                "area": float(mesh.area_faces[face_ids].sum()),
                "boundary_edges": [],
                "neighboring_patches": [],
            }
        )
    return records


def _face_patch_map(face_count, patches):
    face_to_patch = np.full(face_count, -1, dtype=np.int64)
    for patch_index, patch in enumerate(patches):
        face_ids = np.asarray(patch.metadata.get("face_id", []), dtype=np.int64)
        if np.any(face_to_patch[face_ids] >= 0):
            raise ValueError("patch face_id sets must not overlap")
        face_to_patch[face_ids] = patch_index
    return face_to_patch


def _mesh_edge_faces(mesh):
    incident_faces = {}
    for face_id, face in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            incident_faces.setdefault(edge, []).append(face_id)
    return incident_faces


def _patch_normal_on_edge(mesh, incident_faces, face_to_patch, patch_index):
    face_ids = [
        face_id
        for face_id in incident_faces
        if face_to_patch[face_id] == patch_index
    ]
    normal = np.asarray(mesh.face_normals[face_ids], dtype=float).mean(axis=0)
    length = np.linalg.norm(normal)
    return normal / length if length > 1e-12 else None


def _build_patch_adjacency(mesh, patches):
    records = _patch_records(mesh, patches)
    face_to_patch = _face_patch_map(len(mesh.faces), patches)
    graph = {"nodes": records, "edges": []}

    for edge, incident_faces in _mesh_edge_faces(mesh).items():
        patch_ids = sorted(
            {
                int(face_to_patch[face_id])
                for face_id in incident_faces
                if face_to_patch[face_id] >= 0
            }
        )
        if len(patch_ids) != 2:
            continue

        patch0, patch1 = patch_ids
        boundary_vertices = np.asarray(mesh.vertices[list(edge)], dtype=float)
        length = float(np.linalg.norm(boundary_vertices[1] - boundary_vertices[0]))
        normal0 = _patch_normal_on_edge(
            mesh, incident_faces, face_to_patch, patch0
        )
        normal1 = _patch_normal_on_edge(
            mesh, incident_faces, face_to_patch, patch1
        )
        dihedral_angle = 0.0
        if normal0 is not None and normal1 is not None:
            # Plane angle is orientation independent and lies in [0, pi / 2].
            cosine = np.clip(abs(np.dot(normal0, normal1)), 0.0, 1.0)
            dihedral_angle = float(np.arccos(cosine))

        boundary = {
            "v": np.asarray(edge, dtype=np.int64),
            "boundary_v": boundary_vertices,
            "length": length,
            "dihedral_angle": dihedral_angle,
        }
        records[patch0]["boundary_edges"].append(
            {**boundary, "neighbor": patch1}
        )
        records[patch1]["boundary_edges"].append(
            {**boundary, "neighbor": patch0}
        )
        records[patch0]["neighboring_patches"].append(patch1)
        records[patch1]["neighboring_patches"].append(patch0)
        graph["edges"].append(
            {"patch0": patch0, "patch1": patch1, **boundary}
        )

    for record in records:
        record["neighboring_patches"] = sorted(
            set(record["neighboring_patches"])
        )
    return records, graph


def build_patch_adjacency(ply_path, patches):
    """Build patch records and graph edges from shared original-mesh edges."""
    mesh = _load_triangle_mesh(ply_path)
    return _build_patch_adjacency(mesh, patches)


def prepare_patch_graph(
    ply_path,
    min_faces=1,
    merge_vertices=False,
    maximum_sphere_radius_ratio=1.0,
    **fit_options,
):
    """Run segmentation, primitive fitting, and graph construction.

    Returns ``(mesh, patches, results, patch_graph)``.  Each item in
    ``results`` contains the fitting result plus the graph node attributes.
    """
    if min_faces < 1:
        raise ValueError("min_faces must be at least 1")
    mesh = _load_triangle_mesh(ply_path)
    patches = _extract_segment_patches(mesh, min_faces, merge_vertices)
    results = identify_patches(patches, **fit_options)
    _enforce_model_fit_bounds(
        mesh,
        results,
        maximum_sphere_radius_ratio=maximum_sphere_radius_ratio,
    )
    records, patch_graph = _build_patch_adjacency(mesh, patches)

    for result, record in zip(results, records):
        record["parameter"] = np.asarray(result["params"], dtype=float)
        result.update(record)

    return mesh, patches, results, patch_graph


def load_colormap(path="global/colors_500.txt"):
    """Load an RGB text colormap used for notebook visualization."""
    colors = np.asarray(np.loadtxt(path, dtype=float), dtype=float)
    if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) == 0:
        raise ValueError("colormap must be a non-empty N x 3 array")
    return colors


def patch_face_colors(face_count, results, colors=None):
    """Create one display color per original face."""
    colors = load_colormap() if colors is None else np.asarray(colors, dtype=float)
    if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) == 0:
        raise ValueError("colors must be a non-empty N x 3 array")

    face_colors = np.zeros((face_count, 3), dtype=float)
    for patch_index, result in enumerate(results):
        face_ids = np.asarray(result.get("face_id", []), dtype=np.int64)
        face_colors[face_ids] = colors[patch_index % len(colors)]
    return face_colors


def plot_patch_graph(mesh, results, colors=None):
    """Display fitted patches with ``meshplot`` when requested by a notebook."""
    try:
        import meshplot as mesh_plot
    except ImportError as error:
        raise ImportError("plot_patch_graph requires the optional meshplot package") from error

    face_colors = patch_face_colors(len(mesh.faces), results, colors)
    return mesh_plot.plot(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        c=face_colors,
    )
