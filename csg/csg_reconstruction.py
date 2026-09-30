"""Stable public API for patch-graph based CSG reconstruction.

Implementation is organized by responsibility under :mod:`csg_core`:

* ``candidates``: primitive fitting and candidate generation;
* ``boolean``: ADD/SUBTRACT classification, INTERSECTION inference, and search;
* ``validation``: containment and geometry validation;
* ``openscad``: OpenSCAD serialization.

The original function-based entry points remain available from this module, so
existing notebooks and scripts do not need import changes.  Class-based entry
points are also exported for more structured pipelines.
"""

from .csg_core import boolean, candidates, openscad, validation


BooleanCSGReconstructor = boolean.BooleanCSGReconstructor
build_csg_ir = boolean.build_csg_ir
classify_boolean_operations = boolean.classify_boolean_operations
generate_boolean_hypotheses = boolean.generate_boolean_hypotheses
infer_intersection_groups = boolean.infer_intersection_groups
reconstruct_csg_tree = boolean.reconstruct_csg_tree

PrimitiveCandidate = candidates.PrimitiveCandidate
PrimitiveCandidateGenerator = candidates.PrimitiveCandidateGenerator
PrimitiveFitter = candidates.PrimitiveFitter
fit_cone = candidates.fit_cone
fit_cylinder = candidates.fit_cylinder
fit_sphere = candidates.fit_sphere
fit_torus = candidates.fit_torus
classify_transition_patches = candidates.classify_transition_patches
generate_transition_curved_candidates = candidates.generate_transition_curved_candidates
generate_planar_notch_cutters = candidates.generate_planar_notch_cutters
generate_pocket_floor_cutters = candidates.generate_pocket_floor_cutters
generate_cube_candidates = candidates.generate_cube_candidates
generate_curved_candidates = candidates.generate_curved_candidates
generate_extrusion_candidates = candidates.generate_extrusion_candidates
generate_spline_extrusion_candidates = candidates.generate_spline_extrusion_candidates
generate_primitive_candidates = candidates.generate_primitive_candidates
primitive_bounds = candidates.primitive_bounds
primitive_contains = candidates.primitive_contains

CSGValidator = validation.CSGValidator
csg_contains = validation.csg_contains
validate_csg = validation.validate_csg

OpenSCADExporter = openscad.OpenSCADExporter
csg_to_openscad = openscad.csg_to_openscad

__all__ = [
    "BooleanCSGReconstructor",
    "CSGValidator",
    "OpenSCADExporter",
    "PrimitiveCandidate",
    "PrimitiveCandidateGenerator",
    "PrimitiveFitter",
    "build_csg_ir",
    "classify_boolean_operations",
    "csg_contains",
    "csg_to_openscad",
    "fit_cone",
    "fit_cylinder",
    "fit_sphere",
    "fit_torus",
    "classify_transition_patches",
    "generate_transition_curved_candidates",
    "generate_planar_notch_cutters",
    "generate_pocket_floor_cutters",
    "generate_extrusion_candidates",
    "generate_spline_extrusion_candidates",
    "generate_boolean_hypotheses",
    "infer_intersection_groups",
    "generate_cube_candidates",
    "generate_curved_candidates",
    "generate_primitive_candidates",
    "primitive_bounds",
    "primitive_contains",
    "reconstruct_csg_tree",
    "validate_csg",
]
