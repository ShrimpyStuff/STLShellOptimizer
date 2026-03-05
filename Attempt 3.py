import math

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dolfinx import fem
import ufl

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
    
    aligned_vertex_normals = mesh.vertex_normals[np.isin(vertices, aligned_pts, axis=0).any(axis=1)]

    return aligned_pts, list(aligned_edges), list(boundary_nodes), aligned_vertex_normals

def add_thickness(pts, edges, thickness=2.0):
    t = 2.0  # mm

    outer_points = points + (t / 2.0) * vertex_normals
    inner_points = points - (t / 2.0) * vertex_normals


def calculate_compliance(pts, edges, boundary_nodes):
    from dolfinx import mesh as dmesh
    from dolfinx.io import gmshio
    
    # Convert points and edges to dolfinx mesh
    pts_array = np.array(pts)
    
    # Create cells (line elements for 1D structure, or triangles if 3D)
    # For a shell structure with edges, we'll use line segments
    cells = np.array([(0, 1)], dtype=np.int32)
    
    # Build cell connectivity from edges
    point_dict = {tuple(pt): idx for idx, pt in enumerate(pts_array)}
    cells_list = []
    for edge in edges:
        pt1_idx = point_dict.get(edge[0])
        pt2_idx = point_dict.get(edge[1])
        if pt1_idx is not None and pt2_idx is not None:
            cells_list.append([pt1_idx, pt2_idx])
    
    if len(cells_list) > 0:
        cells = np.array(cells_list, dtype=np.int32)
        # Create mesh with line elements (cell type 1)
        domain = dmesh.create_mesh(dmesh.umap_type=1, cells=cells, x=pts_array, cell_type=dmesh.CellType.line)
    else:
        print("Warning: No valid cells created from edges")
        return None
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        E = 1.0
        nu = 0.3
        mu = E / (2*(1+nu))
        lmbda = E*nu/((1+nu)*(1-2*nu))
        return 2*mu*epsilon(u) + lmbda*ufl.tr(epsilon(u))*ufl.Identity(u.ufl_function_space().mesh.geometry.dim)
    
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    u = fem.Function(V)
    
    # Set boundary conditions on boundary nodes
    boundary_node_indices = []
    for boundary_node in boundary_nodes:
        if boundary_node in point_dict:
            boundary_node_indices.append(point_dict[boundary_node])
    
    if len(boundary_node_indices) > 0:
        bc = fem.dirichletbc(np.zeros(domain.geometry.dim), boundary_node_indices, V)
        bcs = [bc]
    else:
        bcs = []
    
    a = ufl.inner(sigma(u), epsilon(u)) * ufl.dx
    compliance = fem.assemble_scalar(fem.form(a))
    
    return compliance

if __name__ == "__main__":
    stl_path = "input1.stl"
    pts, edges, boundary_nodes, vertex_normals = create_shell(stl_path)
    print(f"Generated {len(pts)} points, {len(edges)} edges, and {len(boundary_nodes)} boundary nodes.")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    pts_array = np.array(pts)
    ax.scatter(pts_array[:, 0], pts_array[:, 1], pts_array[:, 2], s=1, c='red')
    
    # Draw edges connecting the points
    for edge in edges:
        pt1, pt2 = edge
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'b-', alpha=0.6)
    
    calculate_compliance(pts, edges)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("STL Mesh with Generated Points")
    plt.show()