import numpy as np
import trimesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dolfinx import fem

class GeodesicShell:
    def __init__(self, stl_path, tolerance=2e-3):
        # Load STL
        mesh = trimesh.load_mesh(stl_path)
        self.mesh = mesh
        self.tolerance = tolerance

        # Center mesh at origin
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)

        # Original vertices and faces
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.vertex_normals = mesh.vertex_normals

        # Simplify points along X, Y, Z axes + top/bottom layers
        pts_array = self._select_aligned_points()
        # pts_array = self.vertices  # For now, use all vertices
        self.points, self.point_idx_map = self._unique_vertices(pts_array)

        # Build edges between selected points
        self.edges = self._build_edges()

        # Explicitly connect the farthest side points of top and bottom loops
        self._connect_top_bottom_farthest_side_points()
        
        # Identify boundary nodes (edges that appear only once)
        self.boundary_nodes = self._find_boundary_nodes()

        self.full_edges = self._full_edges()

        self.add_close_edges()

        self.boundary_nodes = self._find_boundary_nodes()

    def _select_aligned_points(self):
        # Select points at edges and middle of X and Y directions, plus Z planes
        pts_array = self.vertices
        
        # Get X range (min, middle, max)
        x_min = pts_array[:, 0].min() + 0.01  # Avoid exact min to prevent flat face
        x_max = pts_array[:, 0].max() - 0.01  # Avoid exact max to prevent flat face
        x_mid = (x_min + x_max) / 2
        
        # Get Y range (min, middle, max)
        y_min = pts_array[:, 1].min() + 0.01  # Avoid exact min to prevent flat face
        y_max = pts_array[:, 1].max() - 0.01  # Avoid exact max to prevent flat face
        y_mid = (y_min + y_max) / 2
        
        # Get Z range
        z_min = pts_array[:, 2].min()
        z_max = pts_array[:, 2].max()
        
        aligned_pts = pts_array[
            np.isclose(pts_array[:, 0], x_min, atol=self.tolerance) |  # X min
            np.isclose(pts_array[:, 0], x_mid, atol=self.tolerance) |  # X middle
            np.isclose(pts_array[:, 0], x_max, atol=self.tolerance) |  # X max
            np.isclose(pts_array[:, 1], y_min, atol=self.tolerance) |  # Y min
            np.isclose(pts_array[:, 1], y_mid, atol=self.tolerance) |  # Y middle
            np.isclose(pts_array[:, 1], y_max, atol=self.tolerance) |  # Y max
            np.isclose(pts_array[:, 2], 0, atol=self.tolerance) |  # Z=0
            np.isclose(pts_array[:, 2], z_max - 0.01, atol=self.tolerance) |  # Top
            np.isclose(pts_array[:, 2], z_min + 0.01, atol=self.tolerance) # Bottom
        ]
        return aligned_pts

    def _unique_vertices(self, pts_array):
        # Deduplicate vertices
        unique_pts, inverse_indices = np.unique(pts_array, axis=0, return_inverse=True)
        # Map original tuple -> index
        point_idx_map = {tuple(p): i for i, p in enumerate(unique_pts)}
        return unique_pts, point_idx_map

    def _build_edges(self):
        # Manually build edges for each aligned plane/axis to create clean loops    
        edges_set = set()
        tol = 1e-5
        
        # Helper function to create loop edges from ordered points
        def create_loop_edges(indices):
            """Connect consecutive points and close the loop"""
            if len(indices) < 2:
                return
            for i in range(len(indices)):
                next_i = (i + 1) % len(indices)  # Wrap around to close the loop
                edge = tuple(sorted([indices[i], indices[next_i]]))
                edges_set.add(edge)
        
        # Helper function to sort points into a loop based on angle or parametric position
        def sort_points_into_loop(pts_subset, plane_axis):
            """
            Sort points to form a continuous loop
            plane_axis: 0 for YZ plane (X=0), 1 for XZ plane (Y=0), 2 for XY plane (Z=const)
            """
            if len(pts_subset) < 2:
                return pts_subset
            
            # Get the two axes perpendicular to the plane
            if plane_axis == 0:  # X=0, use Y and Z
                ax1, ax2 = 1, 2
            elif plane_axis == 1:  # Y=0, use X and Z
                ax1, ax2 = 0, 2
            else:  # Z=const, use X and Y
                ax1, ax2 = 0, 1
            
            # Calculate center of the point cloud
            center = np.mean(pts_subset[:, [ax1, ax2]], axis=0)
            
            # Calculate angles from center
            angles = np.arctan2(
                pts_subset[:, ax2] - center[1],
                pts_subset[:, ax1] - center[0]
            )
            
            # Sort by angle to form a loop
            sorted_indices = np.argsort(angles)
            return pts_subset[sorted_indices]
        
        # Helper function to process a plane at a specific value
        def process_plane(axis, value, plane_axis):
            """Process a single plane and create loop edges"""
            mask = np.isclose(self.points[:, axis], value, atol=self.tolerance)
            if np.sum(mask) >= 2:
                plane_points = self.points[mask]
                sorted_pts = sort_points_into_loop(plane_points, plane_axis=plane_axis)
                # Get original indices
                indices = [i for i, mask_val in enumerate(mask) if mask_val]
                # Re-map to sorted order
                sorted_indices = []
                for sorted_pt in sorted_pts:
                    for idx in indices:
                        if np.allclose(self.points[idx], sorted_pt, atol=tol):
                            sorted_indices.append(idx)
                            break
                create_loop_edges(sorted_indices)
        
        # Get X range (min, middle, max)
        x_min = self.points[:, 0].min() + 0.01  # Avoid exact min to prevent flat face
        x_max = self.points[:, 0].max() - 0.01  # Avoid exact max to prevent flat face
        x_mid = (x_min + x_max) / 2
        
        # Get Y range (min, middle, max)
        y_min = self.points[:, 1].min() + 0.01  # Avoid exact min to prevent flat face
        y_max = self.points[:, 1].max() - 0.01  # Avoid exact max to prevent flat face
        y_mid = (y_min + y_max) / 2
        
        # Get Z range
        z_min = self.points[:, 2].min() + 0.01  # Avoid exact min to prevent flat face
        z_max = self.points[:, 2].max() - 0.01  # Avoid exact max to prevent flat face
        
        # Process X planes (min, middle, max)
        process_plane(0, x_mid, plane_axis=0)
        
        # Process Y planes (min, middle, max)
        process_plane(1, y_min, plane_axis=1)
        process_plane(1, y_mid, plane_axis=1)
        process_plane(1, y_max, plane_axis=1)
        
        # Process Z planes (0, top, bottom)
        process_plane(2, 0, plane_axis=2)  # Z=0
        process_plane(2, z_max, plane_axis=2)  # Top
        process_plane(2, z_min, plane_axis=2)  # Bottom
        
        return list(edges_set)
    
    def _full_edges(self):
            """
            Build a set of all macro edges (struts between major corners) of the curved shape.
            Returns a set of sorted coordinate tuples to avoid (A, B) and (B, A) duplicates.
            """
            full_edges = set()
            for edge_idx in range(len(self.edges)):
                corner_a, corner_b = self.find_closest_corners(edge_idx)
                
                # Sort the coordinate tuples lexicographically so (A, B) == (B, A)
                edge_pair = tuple(sorted((tuple(corner_a), tuple(corner_b))))
                
                # Prevent adding zero-length edges just in case
                if edge_pair[0] != edge_pair[1]:
                    full_edges.add(edge_pair)
                    
            return full_edges

    def _find_boundary_nodes(self):
        # Count edges per vertex to find boundary vertices
        edge_count = {}
        for a, b in self.edges:
            edge_count[a] = edge_count.get(a, 0) + 1
            edge_count[b] = edge_count.get(b, 0) + 1
        # Boundary vertices = those connected to only 1 edge
        boundary_nodes = [v for v, count in edge_count.items() if count == 1]
        return boundary_nodes

    def _connect_top_bottom_farthest_side_points(self, num_segments=12):
        """
        Add a segmented path between the most sideward points of the top and
        bottom loops, projecting intermediate points to the mesh surface.
        """
        if len(self.points) < 2:
            return

        z_vals = self.points[:, 2]
        z_top = np.max(z_vals)
        z_bottom = np.min(z_vals)

        # Use a slightly relaxed tolerance to robustly capture loop samples.
        z_tol = max(self.tolerance * 2, 1e-5)
        top_indices = np.where(np.isclose(z_vals, z_top, atol=z_tol))[0]
        bottom_indices = np.where(np.isclose(z_vals, z_bottom, atol=z_tol))[0]

        if len(top_indices) == 0 or len(bottom_indices) == 0:
            return

        # Sideward score is distance from the loop centerline in X.
        x_mid = 0.5 * (self.points[:, 0].min() + self.points[:, 0].max())

        top_x_dist = np.abs(self.points[top_indices, 0] - x_mid)
        bot_x_dist = np.abs(self.points[bottom_indices, 0] - x_mid)

        top_side_idx = int(top_indices[np.argmax(top_x_dist)])
        bottom_side_idx = int(bottom_indices[np.argmax(bot_x_dist)])

        if top_side_idx == bottom_side_idx:
            return

        # Build a polyline from top -> bottom and project each intermediate
        # point to the surface to follow the geometry rather than a straight chord.
        start_pt = self.points[top_side_idx]
        end_pt = self.points[bottom_side_idx]
        chain_indices = [top_side_idx]
        merge_tol = max(self.tolerance * 0.75, 1e-5)

        for step in range(1, num_segments):
            t = step / num_segments
            interp_pt = start_pt * (1.0 - t) + end_pt * t
            projected_pt, _, _ = trimesh.proximity.closest_point(
                self.mesh,
                interp_pt.reshape(1, 3)
            )
            projected_pt = projected_pt[0]

            # Reuse a nearby existing point when possible to avoid duplicates.
            dists = np.linalg.norm(self.points - projected_pt, axis=1)
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] <= merge_tol:
                chain_indices.append(nearest_idx)
            else:
                new_idx = len(self.points)
                self.points = np.vstack([self.points, projected_pt])
                chain_indices.append(new_idx)

        chain_indices.append(bottom_side_idx)

        edge_set = {tuple(sorted(e)) for e in self.edges}
        for i in range(len(chain_indices) - 1):
            a_idx = chain_indices[i]
            b_idx = chain_indices[i + 1]
            if a_idx == b_idx:
                continue
            edge = tuple(sorted((a_idx, b_idx)))
            if edge not in edge_set:
                self.edges.append(edge)
                edge_set.add(edge)

    # -----------------------------
    # Helper functions for geodesic manipulation
    # -----------------------------

    def add_close_edges(self):
        """
        Add edges between boundary nodes and their closest unconnected neighbor.
        Uses a normalized set to strictly prevent duplicate or overlapping edges.
        """
        # Normalize all existing edges to (min_idx, max_idx) for O(1) safe lookup
        existing_edges = {tuple(sorted(e)) for e in self.edges}
        new_edges = []
        
        for i in self.boundary_nodes:
            min_dist = float('inf')
            closest_j = None
            
            for j in range(len(self.points)):
                if i == j:
                    continue
                
                # Normalize the potential edge to check if it already exists
                potential_edge = tuple(sorted([i, j]))
                if potential_edge in existing_edges:
                    continue
                    
                dist = np.linalg.norm(self.points[i] - self.points[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_j = j
            
            if closest_j is not None:
                new_edge = tuple(sorted([i, closest_j]))
                new_edges.append(new_edge)
                # Add to existing_edges immediately so other boundary nodes 
                # don't accidentally create the exact same edge
                existing_edges.add(new_edge) 
                
        self.edges.extend(new_edges)

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
        
        # Check for preexisting point close to the midpoint
        tolerance = 1e-4
        closest_existing_idx = None
        min_dist_to_existing = float('inf')
        
        for idx, point in enumerate(self.points):
            dist_to_midpoint = np.linalg.norm(point - midpoint)
            if dist_to_midpoint < tolerance and dist_to_midpoint < min_dist_to_existing:
                min_dist_to_existing = dist_to_midpoint
                closest_existing_idx = idx
        
        # Use existing point if found, otherwise create new one
        if closest_existing_idx is not None:
            prev_idx = closest_existing_idx
        else:
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
        Walk along a subdivided line to find the major junctions/corners at the ends.
        A corner is defined as any vertex with a valence NOT equal to 2 (e.g., junctions 
        with 3+ connections, or dead-end boundaries with 1 connection).
        """
        a_idx, b_idx = self.edges[edge_idx]

        def walk_to_corner(start_idx, coming_from_idx):
            current_idx = start_idx
            prev_idx = coming_from_idx
            
            # Safety counter to prevent infinite loops in closed circular loops
            max_steps = len(self.points) 
            steps = 0
            
            while steps < max_steps:
                steps += 1
                
                # Find all neighbors of the current vertex
                neighbors = []
                for e in self.edges:
                    if current_idx == e[0]:
                        neighbors.append(e[1])
                    elif current_idx == e[1]:
                        neighbors.append(e[0])
                
                # If valence != 2, we have found a structural corner or boundary
                if len(neighbors) != 2:
                    return current_idx
                    
                # Otherwise, it's a subdivision point (valence == 2). 
                # Keep walking to the neighbor that isn't the one we just came from.
                next_idx = neighbors[0] if neighbors[1] == prev_idx else neighbors[1]
                
                prev_idx = current_idx
                current_idx = next_idx
                
            return current_idx # Fallback if loop finishes (e.g., a perfect circle)

        corner_a = walk_to_corner(a_idx, b_idx)
        corner_b = walk_to_corner(b_idx, a_idx)
        
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
    # plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    shell = GeodesicShell("Tibia_Section.stl")

    # edge1 = shell.add_edge(0, 1)  # Example of adding an edge between two vertices
    # shell.subdivide_edge(edge1)
    # shell.project_points_to_surface()

    print(len(shell.points), "points")
    print(len(shell.edges), "edges")
    print(len(shell.boundary_nodes), "boundary nodes")

    print(len(shell.full_edges), "full edges")

    plot_geodesic_shell(shell)