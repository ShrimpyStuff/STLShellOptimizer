#!/usr/bin/env python3
"""
Visualize the mesh boundary (planar cut) and the generated prism points.
This helps identify if prism points are near the cut region and might bridge the gap.
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import GeodesicShell class
import importlib.util
spec = importlib.util.spec_from_file_location("newest", "Newest Version.py")
newest_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(newest_module)
GeodesicShell = newest_module.GeodesicShell

def visualize_cut_and_prism(stl_path="Tibia.3mf"):
    """
    Visualize the mesh with its boundary edges (cut) and the generated prism points.
    """
    # Load and center mesh
    mesh = trimesh.load_mesh(stl_path)
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    
    # Find boundary edges
    print("Detecting boundary edges...")
    edges_sorted = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    
    if len(boundary_edges) > 0:
        boundary_vertices = np.unique(boundary_edges.flatten())
        boundary_coords = mesh.vertices[boundary_vertices]
        print(f"Found {len(boundary_vertices)} boundary vertices")
        
        # Detect cut plane
        stds = [
            boundary_coords[:, 0].std(),
            boundary_coords[:, 1].std(),
            boundary_coords[:, 2].std()
        ]
        min_std_axis = np.argmin(stds)
        axis_names = ['X', 'Y', 'Z']
        cut_axis = axis_names[min_std_axis]
        cut_position = boundary_coords[:, min_std_axis].mean()
        
        print(f"\nCut plane: {cut_axis}-axis at {cut_position:.4f}")
        print(f"Cut range: [{boundary_coords[:, min_std_axis].min():.4f}, {boundary_coords[:, min_std_axis].max():.4f}]")
    else:
        boundary_coords = None
        print("No boundary edges found")
    
    # Generate prism shell
    print("\nGenerating prism shell...")
    shell = GeodesicShell(stl_path, 
                          num_layers=4,
                          points_per_layer=16,
                          edge_subdivisions=16,
                          triangle_span=2)
    
    prism_points = shell.points
    print(f"Generated {len(prism_points)} prism points")
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    
    # View 1: 3D overview
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(prism_points[:, 0], prism_points[:, 1], prism_points[:, 2],
               c='blue', s=20, alpha=0.6, label='Prism points')
    
    if boundary_coords is not None:
        ax1.scatter(boundary_coords[:, 0], boundary_coords[:, 1], boundary_coords[:, 2],
                   c='red', s=10, alpha=0.8, label='Boundary (cut)')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Overview')
    ax1.legend()
    
    # View 2: XY projection (looking down Z)
    ax2 = fig.add_subplot(132)
    ax2.scatter(prism_points[:, 0], prism_points[:, 1],
               c='blue', s=20, alpha=0.6, label='Prism points')
    
    if boundary_coords is not None:
        ax2.scatter(boundary_coords[:, 0], boundary_coords[:, 1],
                   c='red', s=10, alpha=0.8, label='Boundary (cut)')
        if cut_axis == 'X':
            ax2.axvline(cut_position, color='red', linestyle='--', alpha=0.5)
        elif cut_axis == 'Y':
            ax2.axhline(cut_position, color='red', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY View (Top)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # View 3: XZ or YZ projection depending on cut axis
    ax3 = fig.add_subplot(133)
    if boundary_coords is not None and cut_axis == 'X':
        ax3.scatter(prism_points[:, 0], prism_points[:, 2],
                   c='blue', s=20, alpha=0.6, label='Prism points')
        ax3.scatter(boundary_coords[:, 0], boundary_coords[:, 2],
                   c='red', s=10, alpha=0.8, label='Boundary (cut)')
        ax3.axvline(cut_position, color='red', linestyle='--', alpha=0.5, label=f'Cut at X={cut_position:.4f}')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('XZ View (Side)')
    elif boundary_coords is not None and cut_axis == 'Y':
        ax3.scatter(prism_points[:, 1], prism_points[:, 2],
                   c='blue', s=20, alpha=0.6, label='Prism points')
        ax3.scatter(boundary_coords[:, 1], boundary_coords[:, 2],
                   c='red', s=10, alpha=0.8, label='Boundary (cut)')
        ax3.axvline(cut_position, color='red', linestyle='--', alpha=0.5, label=f'Cut at Y={cut_position:.4f}')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        ax3.set_title('YZ View (Side)')
    else:
        ax3.scatter(prism_points[:, 0], prism_points[:, 2],
                   c='blue', s=20, alpha=0.6, label='Prism points')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('XZ View (Side)')
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.savefig('cut_and_points_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: cut_and_points_visualization.png")
    # plt.show()  # Disabled for headless mode
    
    # Analyze proximity of prism points to boundary
    if boundary_coords is not None:
        print("\nAnalyzing prism point proximity to cut boundary...")
        axis_idx = min_std_axis
        
        # Find prism points close to the cut plane
        close_points = []
        for i, pt in enumerate(prism_points):
            if abs(pt[axis_idx] - cut_position) < 0.01:  # Within 0.01 units
                close_points.append((i, pt, abs(pt[axis_idx] - cut_position)))
        
        if close_points:
            print(f"\nFound {len(close_points)} prism points within 0.01 of cut plane:")
            for i, pt, dist in close_points[:10]:  # Show first 10
                print(f"  Point {i}: {pt} (distance: {dist:.6f})")
            if len(close_points) > 10:
                print(f"  ... and {len(close_points) - 10} more")
        else:
            print("No prism points found very close to cut plane")
        
        # Check if edges span across the cut
        print("\nChecking if edges bridge across the cut...")
        bridging_edges = []
        for edge in shell.edges[:100]:  # Check first 100 edges
            a_idx, b_idx = edge
            a = prism_points[a_idx]
            b = prism_points[b_idx]
            
            # Check if edge crosses the cut plane
            a_coord = a[axis_idx]
            b_coord = b[axis_idx]
            
            # If points are on opposite sides of cut plane, edge bridges it
            if (a_coord < cut_position < b_coord) or (b_coord < cut_position < a_coord):
                bridging_edges.append((a_idx, b_idx, a_coord, b_coord))
        
        if bridging_edges:
            print(f"WARNING: Found {len(bridging_edges)} edges that bridge across the cut plane!")
            for a_idx, b_idx, a_coord, b_coord in bridging_edges[:5]:
                print(f"  Edge ({a_idx}, {b_idx}): {axis_names[axis_idx]} spans [{a_coord:.4f}, {b_coord:.4f}]")
        else:
            print("No edges found bridging across the cut plane")

if __name__ == "__main__":
    visualize_cut_and_prism()
