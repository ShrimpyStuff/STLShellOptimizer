import trimesh
import numpy as np

# Load STL
mesh = trimesh.load("input.stl")

# Number of random points you want on the surface
num_points = 500  # adjust as needed

# Sample points uniformly across the surface
points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

# points is an array of shape (num_points, 3)
print("Sampled points on surface:", points.shape)

# Optional: save points for inspection
np.savetxt("surface_points.csv", points, delimiter=",")