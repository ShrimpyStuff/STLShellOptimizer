import math

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dolfinx import *

def create_shell(stl_path):
    mesh = trimesh.load_mesh(stl_path)

    center = mesh.bounds.mean(axis=0)
    translation = -center
    mesh.apply_translation(translation)

    # angle = math.pi / 2
    # axis = [0, 0, 1]
    # transform = trimesh.transformations.rotation_matrix(angle, axis)
    # mesh.apply_transform(transform)
    
    vertices = mesh.vertices
    faces = mesh.faces
    pts = []
    edges = set()
    boundary_nodes = set()

    for face in faces:
        v1, v2, v3 = vertices[face]
        pts.extend([v1, v2, v3])
        edges.update([(tuple(v1), tuple(v2)), (tuple(v2), tuple(v3)), (tuple(v3), tuple(v1))])
    edge_count = {}
    for edge in edges:
        edge_count[edge] = edge_count.get(edge, 0) + 1
    for edge, count in edge_count.items():
        if count == 1:
            boundary_nodes.update(edge)
    
    pts_array = np.array(pts)
    # Get points from all three axes (x=0, y=0, z=0) plus top and bottom planes
    tolerance = 1e-6
    aligned_pts = pts_array[
        np.isclose(pts_array[:, 0], 0, atol=tolerance) | # X
        np.isclose(pts_array[:, 1], 0, atol=tolerance) | # Y
        np.isclose(pts_array[:, 2], 0, atol=tolerance) | # Z
        np.isclose(pts_array[:, 2], pts_array[:, 2].max(), atol=tolerance) | # Top
        np.isclose(pts_array[:, 2], pts_array[:, 2].min(), atol=tolerance) # Bottom
    ]

    aligned_pts_tuples = set(map(tuple, aligned_pts))
    aligned_edges = set()
    for edge in edges: 
        if edge[0] in aligned_pts_tuples and edge[1] in aligned_pts_tuples:
            aligned_edges.add(edge)

    return aligned_pts, list(aligned_edges), list(boundary_nodes)

def calculate_compliance(pts, edges):
    V = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))

if __name__ == "__main__":
    stl_path = "input1.stl"
    pts, edges, boundary_nodes = create_shell(stl_path)
    print(f"Generated {len(pts)} points, {len(edges)} edges, and {len(boundary_nodes)} boundary nodes.")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    pts_array = np.array(pts)
    ax.scatter(pts_array[:, 0], pts_array[:, 1], pts_array[:, 2], s=1, c='red')
    
    # Draw edges connecting the points
    for edge in edges:
        pt1, pt2 = edge
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'b-', alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("STL Mesh with Generated Points")
    plt.show()