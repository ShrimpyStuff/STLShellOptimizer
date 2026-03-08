#!/usr/bin/env python3
"""
Simple script to detect and visualize the planar cut in a mesh.
This helps identify where boundary points exist that should be excluded from prism generation.
"""

import numpy as np
import trimesh
import sys

def detect_cut(stl_path):
    """Detect planar cut in the mesh"""
    print(f"Loading mesh: {stl_path}")
    mesh = trimesh.load_mesh(stl_path)
    
    # Center mesh at origin (same as GeodesicShell does)
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    
    print(f"\nMesh info:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Is watertight: {mesh.is_watertight}")
    
    print(f"\nMesh bounds (centered):")
    print(f"  X: [{mesh.bounds[0][0]:.4f}, {mesh.bounds[1][0]:.4f}]")
    print(f"  Y: [{mesh.bounds[0][1]:.4f}, {mesh.bounds[1][1]:.4f}]")
    print(f"  Z: [{mesh.bounds[0][2]:.4f}, {mesh.bounds[1][2]:.4f}]")
    
    if mesh.is_watertight:
        print("\n✓ Mesh is watertight - no planar cut detected")
        return None, None
    
    print("\nDetecting boundary edges...")
    
    # Find boundary edges using numpy's unique function for efficiency
    edges_sorted = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    
    if len(boundary_edges) == 0:
        print("No boundary edges found")
        return None, None
    
    print(f"  Found {len(boundary_edges)} boundary edges")
    
    # Get boundary vertices
    boundary_vertices = np.unique(boundary_edges.flatten())
    boundary_coords = mesh.vertices[boundary_vertices]
    
    print(f"  Found {len(boundary_vertices)} boundary vertices")
    
    print(f"\nBoundary region bounds:")
    print(f"  X: [{boundary_coords[:, 0].min():.4f}, {boundary_coords[:, 0].max():.4f}]")
    print(f"  Y: [{boundary_coords[:, 1].min():.4f}, {boundary_coords[:, 1].max():.4f}]")
    print(f"  Z: [{boundary_coords[:, 2].min():.4f}, {boundary_coords[:, 2].max():.4f}]")
    
    # Check which axis is most planar (smallest std deviation indicates planar cut)
    stds = [
        boundary_coords[:, 0].std(),  # X
        boundary_coords[:, 1].std(),  # Y  
        boundary_coords[:, 2].std()   # Z
    ]
    
    print(f"\nBoundary coordinate variation (std dev - smaller = more planar):")
    print(f"  X: {stds[0]:.6f}")
    print(f"  Y: {stds[1]:.6f}")
    print(f"  Z: {stds[2]:.6f}")
    
    min_std_axis = np.argmin(stds)
    axis_names = ['X', 'Y', 'Z']
    axis_chars = ['x', 'y', 'z']
    
    axis = axis_chars[min_std_axis]
    position = boundary_coords[:, min_std_axis].mean()
    cut_min = boundary_coords[:, min_std_axis].min()
    cut_max = boundary_coords[:, min_std_axis].max()
    
    print(f"\n{'='*60}")
    print(f"PLANAR CUT DETECTED")
    print(f"{'='*60}")
    print(f"  Axis: {axis_names[min_std_axis]}")
    print(f"  Average Position: {position:.4f}")
    print(f"  Range: [{cut_min:.4f}, {cut_max:.4f}]")
    print(f"  Std deviation: {stds[min_std_axis]:.6f}")
    print(f"\nTo exclude this region from prism generation, use:")
    print(f"  cut_plane_axis='{axis}'")
    print(f"  cut_plane_position={position:.4f}")
    print(f"  cut_exclusion_tolerance=0.02  # adjust as needed")
    print(f"{'='*60}")
    
    return axis, position

if __name__ == "__main__":
    stl_file = "Tibia_No_Fill.stl" if len(sys.argv) < 2 else sys.argv[1]
    detect_cut(stl_file)
