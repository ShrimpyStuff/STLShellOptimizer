import numpy as np
import trimesh
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dolfinx import fem

class GeodesicShell:
    def __init__(
        self,
        stl_path,
        tolerance=2e-3,
        num_layers=8,
        points_per_layer=12,
        edge_subdivisions=5,
        triangle_span=2,
        layer_stagger_fraction=0.0,
        connectivity_stagger_fraction=0.5,
    ):
        # Load STL
        mesh = trimesh.load_mesh(stl_path)
        self.mesh = mesh
        self.tolerance = tolerance
        
        # Prism parameters
        self.num_layers = num_layers
        self.points_per_layer = points_per_layer
        self.edge_subdivisions = edge_subdivisions
        self.triangle_span = triangle_span
        self.layer_stagger_fraction = layer_stagger_fraction
        self.connectivity_stagger_fraction = connectivity_stagger_fraction

        # Center mesh at origin
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)

        # Original vertices and faces
        self.vertices = mesh.vertices
        self.mesh_faces = mesh.faces
        self.vertex_normals = mesh.vertex_normals

        # Generate outer boundary of trapezoidal prism
        self.points = self._generate_prism_boundary(
            num_layers=num_layers, points_per_layer=points_per_layer
        )
        
        # Build triangular faces for prism shell
        self.faces = self._build_prism_faces()
        
        # Extract edges from faces
        self.edges = self._extract_edges_from_faces()
        
        # Project the entire prism onto the mesh surface with edge subdivision
        self._project_to_surface(num_segments=edge_subdivisions)
        
        # Identify boundary nodes (edges that appear only once)
        self.boundary_nodes = self._find_boundary_nodes()
        self.full_edges = self._full_edges()

    def _generate_prism_boundary(self, num_layers=8, points_per_layer=12):
        """
        Generate only the outer boundary points of a prism.
        Creates a loop of points at each Z layer, forming the outer shell.
        num_layers: number of horizontal layers (Z direction)
        points_per_layer: number of points around each layer's perimeter
        """
        bounds = self.mesh.bounds
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]
        
        # Add small inset to avoid exact boundary
        inset = 0.01
        x_min += inset
        x_max -= inset
        y_min += inset
        y_max -= inset
        z_min += inset
        z_max -= inset
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Calculate radius for the perimeter loop
        x_radius = (x_max - x_min) / 2
        y_radius = (y_max - y_min) / 2
        
        points = []
        
        # Generate points layer by layer (Z direction)
        for iz in range(num_layers):
            z = z_min + (z_max - z_min) * iz / (num_layers - 1) if num_layers > 1 else (z_min + z_max) / 2
            
            # Create a loop of points around the perimeter at this Z level
            for ip in range(points_per_layer):
                angle = 2 * np.pi * ip / points_per_layer
                x = x_center + x_radius * np.cos(angle)
                y = y_center + y_radius * np.sin(angle)
                points.append([x, y, z])
        
        return np.array(points)

    def _project_to_surface(self, num_segments=5):
        """
        Project all prism points onto the mesh surface.
        Subdivide edges with intermediate points for better surface conformance.
        Triangular faces remain intact for later geodesic subdivision.
        """
        # First, project all existing points
        projected_points, _, _ = trimesh.proximity.closest_point(self.mesh, self.points)
        self.points = projected_points
        
        # Then subdivide edges and project intermediate points
        new_edges = []
        new_points_list = []
        
        for edge in self.edges:
            a_idx, b_idx = edge
            a = self.points[a_idx]
            b = self.points[b_idx]
            
            # Create subdivided path along the edge
            segment_indices = [a_idx]
            
            for i in range(1, num_segments):
                t = i / num_segments
                interp_pt = a * (1 - t) + b * t
                # Project to surface
                proj_pt, _, _ = trimesh.proximity.closest_point(self.mesh, interp_pt.reshape(1, 3))
                new_points_list.append(proj_pt[0])
                segment_indices.append(len(self.points) + len(new_points_list) - 1)
            
            segment_indices.append(b_idx)
            
            # Create edges between consecutive points in the subdivided path
            for i in range(len(segment_indices) - 1):
                new_edges.append((segment_indices[i], segment_indices[i + 1]))
        
        # Add new points and replace edges
        if new_points_list:
            self.points = np.vstack([self.points, np.array(new_points_list)])
        self.edges = new_edges

    def _build_prism_faces(self):
        """
        Build a geodesic-like strip between adjacent layers.
        triangle_span defines the base width (in ring-index steps), and the
        opposite vertex is placed at the midpoint index on the other layer.
        """
        faces = []
        
        # Calculate points per layer
        ppl = self.points_per_layer
        num_layers = self.num_layers
        # Enforce an even span so a midpoint index exists.
        span = max(2, int(self.triangle_span))
        if span % 2 != 0:
            span += 1
        half = span // 2
        
        # Create triangular faces between adjacent layers
        # Stagger connectivity by band while keeping ring geometry aligned.
        for layer in range(num_layers - 1):
            lower_base = layer * ppl
            upper_base = (layer + 1) * ppl
            band_shift = int(round(layer * self.connectivity_stagger_fraction * span)) % ppl
            
            for i in range(ppl):
                j = (i + band_shift) % ppl

                # Alternate orientation by index to avoid overlapping triangles.
                if i % 2 == 0:
                    # Base on lower layer, apex at midpoint on upper layer.
                    lower_a = lower_base + j
                    lower_b = lower_base + ((j + span) % ppl)
                    upper_mid = upper_base + ((j + half) % ppl)
                    faces.append((lower_a, lower_b, upper_mid))
                else:
                    # Base on upper layer, apex at midpoint on lower layer.
                    upper_a = upper_base + j
                    upper_b = upper_base + ((j + span) % ppl)
                    lower_mid = lower_base + ((j + half) % ppl)
                    faces.append((upper_a, lower_mid, upper_b))
        
        return faces
    
    def _extract_edges_from_faces(self):
        """
        Extract unique edges directly from triangular faces.
        """
        edges_set = set()
        for face in self.faces:
            edges_set.add(tuple(sorted([face[0], face[1]])))
            edges_set.add(tuple(sorted([face[1], face[2]])))
            edges_set.add(tuple(sorted([face[2], face[0]])))
        
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

    def export_ml_dataset(self, output_path, include_dense_adjacency=False):
        """
        Export shell geometry and metadata for machine-learning workflows.

        Supported outputs:
        - .npz: multi-array dataset (recommended)
        - .npy:  point matrix only (N, 3)

        Returns the absolute path of the written file.
        """
        ext = os.path.splitext(output_path)[1].lower()
        if ext not in {".npz", ".npy"}:
            raise ValueError("output_path must end with .npz or .npy")

        points = np.asarray(self.points, dtype=np.float32)
        faces = np.asarray(self.faces, dtype=np.int32)
        edges = np.asarray(self.edges, dtype=np.int32)
        boundary_nodes = np.asarray(self.boundary_nodes, dtype=np.int32)

        # Sparse graph representation is generally better for ML than dense NxN.
        edge_index = edges.T if edges.size else np.empty((2, 0), dtype=np.int32)

        if ext == ".npy":
            np.save(output_path, points)
            return os.path.abspath(output_path)

        save_payload = {
            "points": points,
            "faces": faces,
            "edges": edges,
            "edge_index": edge_index,
            "boundary_nodes": boundary_nodes,
            "num_layers": np.int32(self.num_layers),
            "points_per_layer": np.int32(self.points_per_layer),
            "edge_subdivisions": np.int32(self.edge_subdivisions),
            "triangle_span": np.int32(self.triangle_span),
            "tolerance": np.float32(self.tolerance),
            "connectivity_stagger_fraction": np.float32(self.connectivity_stagger_fraction),
        }

        if include_dense_adjacency:
            n = points.shape[0]
            adjacency = np.zeros((n, n), dtype=np.uint8)
            if edges.size:
                adjacency[edges[:, 0], edges[:, 1]] = 1
                adjacency[edges[:, 1], edges[:, 0]] = 1
            save_payload["adjacency"] = adjacency

        np.savez_compressed(output_path, **save_payload)
        return os.path.abspath(output_path)

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
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # Create geodesic shell with outer trapezoidal prism using triangular faces
    # num_layers: number of horizontal layers (Z direction)
    # points_per_layer: number of points around each layer's perimeter
    # edge_subdivisions: number of segments per edge for surface conformance
    # triangle_span controls base width around each ring (2 means apex at the middle index).
    # Keep rings aligned; stagger is handled in face connectivity per layer band.
    shell = GeodesicShell("Tibia_Cut.stl", 
                          num_layers=4,
                          points_per_layer=16,
                          edge_subdivisions=16,
                          triangle_span=2,
                          layer_stagger_fraction=0.0,
                          connectivity_stagger_fraction=0.5)

    # edge1 = shell.add_edge(0, 1)  # Example of adding an edge between two vertices
    # shell.subdivide_edge(edge1)
    # shell.project_points_to_surface()

    print(len(shell.points), "points")
    print(len(shell.faces), "triangular faces")
    print(len(shell.edges), "edges")
    print(len(shell.boundary_nodes), "boundary nodes")

    print(len(shell.full_edges), "full edges")

    ml_file = shell.export_ml_dataset("shell_ml_data.npz")
    print("Saved ML dataset:", ml_file)

    plot_geodesic_shell(shell)