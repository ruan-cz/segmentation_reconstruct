# Function Ccontained

## vertex classify
+ given a cutting surface, classify the vertices into 3 types
    + on surface
    + inside
    + outside

## edge classify
+ given a cutting surface, classify the edges into 3 types
    + on surface
    + inside
    + outside

## edge intersection
+ given a cutting surface, return the intersection points of the edges having intersection with the surface

## triangularization: for shapes constructed by dense points
+ given a parameterized cutting surface, or a surface containing sampled points
    + find the surface patch where projection located
    + compute new sampled points on the project surface
+ given sampled points and desired connections, triangularize all the concerning faces in projection surface
