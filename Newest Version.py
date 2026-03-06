import numpy as np
import trimesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dolfinx import fem

class GeodesicShell:
    def __init__(self, stl_path, tolerance=3.5e-3):
        # Load STL
        mesh = trimesh.load_mesh(stl_path)
        self.mesh = mesh

        # Center mesh at origin
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)

        # Original vertices and faces
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.vertex_normals = mesh.vertex_normals

        # Simplify points along X, Y, Z axes + top/bottom layers
        pts_array = self._select_aligned_points(tolerance)
        # pts_array = self.vertices  # For now, use all vertices
        self.points, self.point_idx_map = self._unique_vertices(pts_array)

        # Build edges between selected points
        self.edges = self._build_edges()
        
        # Identify boundary nodes (edges that appear only once)
        self.boundary_nodes = self._find_boundary_nodes()

    def _select_aligned_points(self, tol):
        # Select points aligned with axes and top/bottom planes
        pts_array = self.vertices
        aligned_pts = pts_array[
            np.isclose(pts_array[:, 0], 0, atol=tol) |  # X=0
            np.isclose(pts_array[:, 1], 0, atol=tol) |  # Y=0
            np.isclose(pts_array[:, 2], 0, atol=tol) |  # Z=0
            np.isclose(pts_array[:, 2], pts_array[:, 2].max(), atol=tol) |  # Top
            np.isclose(pts_array[:, 2], pts_array[:, 2].min() + 0.01, atol=tol) # Bottom
        ]
        return aligned_pts

    def _unique_vertices(self, pts_array):
        # Deduplicate vertices
        unique_pts, inverse_indices = np.unique(pts_array, axis=0, return_inverse=True)
        # Map original tuple -> index
        point_idx_map = {tuple(p): i for i, p in enumerate(unique_pts)}
        return unique_pts, point_idx_map

    def _build_edges(self):
        # Build edges from faces, keep only edges between selected points
        edges_set = set()
        point_tuples = set(map(tuple, self.points))
        for face in self.faces:
            verts = self.vertices[face]
            face_edges = [
                (tuple(verts[0]), tuple(verts[1])),
                (tuple(verts[1]), tuple(verts[2])),
                (tuple(verts[2]), tuple(verts[0]))
            ]
            for a, b in face_edges:
                if a in point_tuples and b in point_tuples:
                    # store sorted tuples to avoid duplicates
                    edges_set.add(tuple(sorted([a, b])))
        # Convert to index-based edges
        edges_idx = [(self.point_idx_map[a], self.point_idx_map[b]) for a, b in edges_set]
        return edges_idx

    def _find_boundary_nodes(self):
        # Count edges per vertex to find boundary vertices
        edge_count = {}
        for a, b in self.edges:
            edge_count[a] = edge_count.get(a, 0) + 1
            edge_count[b] = edge_count.get(b, 0) + 1
        # Boundary vertices = those connected to only 1 edge
        boundary_nodes = [v for v, count in edge_count.items() if count == 1]
        return boundary_nodes

    # -----------------------------
    # Helper functions for geodesic manipulation
    # -----------------------------
    def subdivide_edge(self, edge_idx):
        """
        Subdivide a single edge by inserting a midpoint
        edge_idx: index in self.edges
        Returns new point index, and the two new edges created
        """
        a_idx, b_idx = self.edges[edge_idx]
        a = self.points[a_idx]
        b = self.points[b_idx]
        midpoint = (a + b) / 2
        new_idx = len(self.points)
        self.points = np.vstack([self.points, midpoint])
        self.edges.pop(edge_idx)
        # Replace old edge with two new edges
        self.edges.append((a_idx, new_idx))
        self.edges.append((new_idx, b_idx))
        return new_idx, (a_idx, new_idx), (new_idx, b_idx)

    def subdivide_edge_to_vertex(self, edge_idx, target_idx, num_subdiv=3, full_edge=False):
        """
        Create a curve from edge midpoint to another vertex (like a geodesic strut)
        """
        a_idx, b_idx = self.edges[edge_idx]
        a = self.points[a_idx]
        b = self.points[b_idx]
        if full_edge:
            a, b = self.find_closest_corners(edge_idx)
        target = self.points[target_idx]

        # Start with midpoint of edge
        midpoint = (a + b) / 2
        prev_idx = len(self.points)
        self.points = np.vstack([self.points, midpoint])
        new_edge_list = []

        # Create points along straight line to target
        for i in range(1, num_subdiv + 1):
            t = i / (num_subdiv + 1)
            new_point = midpoint * (1 - t) + target * t
            new_idx = len(self.points)
            self.points = np.vstack([self.points, new_point])
            # Connect previous point to this one
            new_edge_list.append((prev_idx, new_idx))
            prev_idx = new_idx

        # Connect last new point to target vertex
        new_edge_list.append((prev_idx, target_idx))

        # Remove original edge if desired
        self.edges.pop(edge_idx)

        # Add new edges
        self.edges.extend(new_edge_list)

        self.project_points_to_surface()  # Ensure new points lie on surface

        return [idx for idx, _ in new_edge_list]
    
    def split_edge_multiple_to_vertex(self, edge_idx, target_idx, num_splits=3, num_subdiv=3, full_edge=False):
        """
        Split an edge into multiple segments and create struts to the target vertex,
        each strut subdivided into num_subdiv intermediate points
        """
        a_idx, b_idx = self.edges[edge_idx]
        a = self.points[a_idx]
        b = self.points[b_idx]
        if full_edge:
            a, b = self.find_closest_corners(edge_idx)
        target = self.points[target_idx]

        # Create split points along the edge (evenly spaced)
        split_points = []
        split_indices = []
        for i in range(1, num_splits + 1):
            t = i / (num_splits + 1)
            split_point = a * (1 - t) + b * t
            new_idx = len(self.points)
            self.points = np.vstack([self.points, split_point])
            split_points.append(split_point)
            split_indices.append(new_idx)

        new_edge_list = []
        prev_idx = a_idx

        # Reconnect the edge through split points
        for split_idx in split_indices:
            new_edge_list.append((prev_idx, split_idx))
            prev_idx = split_idx

        # Connect last split point to b
        new_edge_list.append((prev_idx, b_idx))

        # Create struts from each split point to target with subdivisions
        for split_idx in split_indices:
            split_pt = self.points[split_idx]
            prev_idx = split_idx

            # Create num_subdiv intermediate points along the strut
            for i in range(1, num_subdiv + 1):
                t = i / (num_subdiv + 1)
                strut_point = split_pt * (1 - t) + target * t
                new_idx = len(self.points)
                self.points = np.vstack([self.points, strut_point])
                new_edge_list.append((prev_idx, new_idx))
                prev_idx = new_idx

            # Connect last intermediate point to target
            new_edge_list.append((prev_idx, target_idx))

        # Remove original edge
        self.edges.pop(edge_idx)

        # Add new edges
        self.edges.extend(new_edge_list)

        self.project_points_to_surface()  # Ensure new points lie on surface

        return [idx for idx, _ in new_edge_list]

    def project_points_to_surface(self):
        """
        Project all points onto the STL surface using nearest point
        """
        closest_points, _, _ = trimesh.proximity.closest_point(self.mesh, self.points)
        self.points = closest_points

    def find_closest_corners(self, edge_idx):
        """
        Find the closest corners (vertex with valence 2) to the endpoints of the edge.
        """
        a_idx, b_idx = self.edges[edge_idx]
        def find_closest_corner(start_idx):
            visited = set()
            current_idx = start_idx
            while True:
                visited.add(current_idx)
                # Get connected edges
                connected_edges = [e for e in self.edges if current_idx in e]
                if len(connected_edges) == 2:
                    return current_idx  # Found a corner
                # Move to the next vertex along an edge
                next_idx = None
                for e in connected_edges:
                    if e[0] == current_idx and e[1] not in visited:
                        next_idx = e[1]
                        break
                    elif e[1] == current_idx and e[0] not in visited:
                        next_idx = e[0]
                        break
                if next_idx is None:
                    return current_idx  # No more vertices to visit, return last one
                current_idx = next_idx

        corner_a = find_closest_corner(a_idx)
        corner_b = find_closest_corner(b_idx)
        return self.points[corner_a], self.points[corner_b]


    def add_thickness(self, thickness=2.0):
        """
        Add uniform thickness along vertex normals.
        Returns outer_points, inner_points
        """
        normals = self.vertex_normals[:len(self.points)]
        outer = self.points + 0.5 * thickness * normals
        inner = self.points - 0.5 * thickness * normals
        return outer, inner
    

    def add_edge(self, a_idx, b_idx):
        """
        Add an edge between two existing vertices
        """
        self.edges.append((a_idx, b_idx))
        return len(self.edges) - 1  # Return index of new edge

def plot_geodesic_shell(shell, show_points=True, show_edges=True, show_boundary=True):
    """
    shell: GeodesicShell instance
    show_points: plot vertices
    show_edges: plot struts / edges
    show_boundary: highlight boundary nodes
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    points = shell.points
    edges = shell.edges
    boundary_nodes = shell.boundary_nodes

    # Plot points
    # if show_points:
    #     ax.scatter(points[:, 0], points[:, 1], points[:, 2],
    #                color='red', s=20, alpha=0.8, label='Vertices')

    # Plot edges
    if show_edges:
        for a_idx, b_idx in edges:
            a = points[a_idx]
            b = points[b_idx]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color='blue', alpha=0.6)

    # Highlight boundary nodes
    if show_boundary:
        boundary_pts = points[boundary_nodes]
        ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1], boundary_pts[:, 2],
                   color='green', s=40, label='Boundary Nodes')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Geodesic Shell Visualization')
    ax.legend()
    ax.view_init(elev=30, azim=45)
    plt.show()

if __name__ == "__main__":
    shell = GeodesicShell("Tibia_Cut.stl")

    # edge1 = shell.add_edge(0, 1)  # Example of adding an edge between two vertices
    # shell.subdivide_edge(edge1)
    # shell.project_points_to_surface()

    print(len(shell.points), "points")
    print(len(shell.edges), "edges")
    print(len(shell.boundary_nodes), "boundary nodes")

    plot_geodesic_shell(shell)