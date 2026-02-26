







#### bugs

+ cylinder fitting: wrongly fit extrusion to cylinder: fillet_transition_model
+ border between two patches consists of multi segments
+ cut operation
    + patch assignment in 
      + closed_cylinder_cutting
+ hole filling
    + planar: accelerate advance front
+ final surface segment
    + pipeline: vsa -> merge on flat and fitted primitive -> cluster on flat but not fitted neighbor patches
    + ordered dihedral computing
+ cylinder cutting: unknown bugs after triangulation



#### accelerate

+ sub models derive patch info

  + patch type
  + border_v and border type

  from base model

+ accelerate merging




$$
A_{ij} = \begin{cases}
L_{ij} \times (1 + \alpha) & \text{if edge is Convex} \\
L_{ij} \times \epsilon & \text{if edge is Concave} \\
0 & \text{if not adjacent}
\end{cases}
$$
