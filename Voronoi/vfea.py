from pathlib import Path

import numpy as np
from Pynite import FEModel3D
import matplotlib.pyplot as plt


UNIT_SQUARE_CORNERS = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ],
    dtype=float,
)


def _point_key(point, decimals=12):
    return tuple(np.round(np.asarray(point, dtype=float), decimals=decimals))


def load_clipped_geometry(npz_path=None):
    geometry_path = Path(npz_path) if npz_path is not None else Path(__file__).with_name("clipped_geometry.npz")
    if not geometry_path.exists():
        raise FileNotFoundError(f"Clipped geometry file not found: {geometry_path}")

    data = np.load(geometry_path)
    vertices_key = "vertices" if "vertices" in data.files else "points"
    vertices = np.asarray(data[vertices_key], dtype=float)
    line_indices = np.asarray(data["line_indices"], dtype=int) if "line_indices" in data.files else np.empty((0, 2), dtype=int)

    return geometry_path, vertices, line_indices


def add_vertex_nodes(model, vertices, scale=100.0, decimals=12):
    """Add vertex nodes to the model, scaling coordinates by the given factor."""
    node_names_by_point = {}
    vertex_node_names = []

    for index, vertex in enumerate(vertices):
        key = _point_key(vertex, decimals=decimals)
        if key in node_names_by_point:
            vertex_node_names.append(node_names_by_point[key])
            continue

        node_name = f"N{len(node_names_by_point) + 1}"
        # Scale coordinates for numerical stability
        x_scaled = float(vertex[0]) * scale
        y_scaled = float(vertex[1]) * scale
        model.add_node(node_name, x_scaled, y_scaled, 0.0)
        node_names_by_point[key] = node_name
        vertex_node_names.append(node_name)

    return node_names_by_point, vertex_node_names


def add_boundary_supports(model, vertices, node_names_by_point, scale=10.0, boundary_tolerance=0.05, decimals=12):
    """Apply fixed supports to all vertex nodes on the boundary [0,1] x [0,1]."""
    support_node_names = {}
    
    for vertex_idx, vertex in enumerate(vertices):
        x, y = float(vertex[0]), float(vertex[1])
        key = _point_key(vertex, decimals=decimals)
        node_name = node_names_by_point.get(key)
        
        if node_name is None:
            continue
        
        # Check if node is on or near boundary (within tolerance)
        on_x_boundary = (x < boundary_tolerance) or (x > 1.0 - boundary_tolerance)
        on_y_boundary = (y < boundary_tolerance) or (y > 1.0 - boundary_tolerance)
        
        if on_x_boundary or on_y_boundary:
            # Apply fixed support to boundary nodes
            model.def_support(node_name, support_DX=True, support_DY=True, support_DZ=True,
                             support_RX=True, support_RY=True, support_RZ=True)
            support_node_names[key] = node_name
    
    # If no boundary nodes found, fall back to original corner support approach
    if not support_node_names:
        for index, corner in enumerate(UNIT_SQUARE_CORNERS, start=1):
            key = _point_key(corner, decimals=decimals)
            node_name = node_names_by_point.get(key)
            
            if node_name is None:
                node_name = f"S{index}"
                x_scaled = float(corner[0]) * scale
                y_scaled = float(corner[1]) * scale
                model.add_node(node_name, x_scaled, y_scaled, 0.0)
                node_names_by_point[key] = node_name

            model.def_support(node_name, support_DX=True, support_DY=True, support_DZ=True,
                             support_RX=True, support_RY=True, support_RZ=True)
            support_node_names[key] = node_name
    
    return support_node_names


def add_members_from_edges(model, vertices, line_indices, node_names_by_point, material_name='Bone', section_name='W12x26', decimals=12):
    """Add frame members connecting vertices along edges."""
    member_names = []

    for member_index, edge in enumerate(line_indices, start=1):
        i_idx, j_idx = edge
        i_vertex = vertices[i_idx]
        j_vertex = vertices[j_idx]

        i_key = _point_key(i_vertex, decimals=decimals)
        j_key = _point_key(j_vertex, decimals=decimals)

        i_node = node_names_by_point.get(i_key)
        j_node = node_names_by_point.get(j_key)

        if i_node is None or j_node is None:
            continue

        member_name = f"M{member_index}"
        model.add_member(member_name, i_node, j_node, material_name, section_name)
        member_names.append(member_name)

    return member_names


def apply_uniform_loading(model, vertex_node_names, force_magnitude=-1.0, direction='FY'):
    """Apply uniform point loads to all vertex nodes."""
    for node_name in vertex_node_names:
        model.add_node_load(node_name, direction, force_magnitude)
    
    # Create a load combo that includes 'Case 1'
    model.add_load_combo('Combo 1', {'Case 1': 1.0})


def extract_displacement_extrema(model, vertex_node_names, load_case='Combo 1'):
    """Calculate total displacement magnitude and find min/max."""
    displacements = {}

    for node_name in vertex_node_names:
        node = model.nodes[node_name]
        # Extract displacement from load case dictionary
        dx = node.DX.get(load_case, 0.0) if isinstance(node.DX, dict) else node.DX
        dy = node.DY.get(load_case, 0.0) if isinstance(node.DY, dict) else node.DY
        dz = node.DZ.get(load_case, 0.0) if isinstance(node.DZ, dict) else node.DZ

        # Handle NaN values
        dx = 0.0 if (dx is None or (isinstance(dx, float) and np.isnan(dx))) else float(dx)
        dy = 0.0 if (dy is None or (isinstance(dy, float) and np.isnan(dy))) else float(dy)
        dz = 0.0 if (dz is None or (isinstance(dz, float) and np.isnan(dz))) else float(dz)

        total_disp = np.sqrt(dx**2 + dy**2 + dz**2)
        displacements[node_name] = {
            'total': total_disp,
            'dx': dx,
            'dy': dy,
            'dz': dz,
        }

    if not displacements:
        return None, None, displacements

    # Find min and max
    min_node = min(displacements.items(), key=lambda x: x[1]['total'])
    max_node = max(displacements.items(), key=lambda x: x[1]['total'])

    return min_node, max_node, displacements


def visualize_structure_with_displacements(vertices, line_indices, 
                                            vertex_node_names, displacements, scale=10.0, model=None):
    """
    Visualize the FEA structure with nodes colored and sized by displacement magnitude.
    
    Parameters:
    -----------
    vertices : ndarray
        Unscaled vertex coordinates
    line_indices : ndarray
        Member connectivity (vertex index pairs)
    vertex_node_names : list
        List of all vertex node names
    displacements : dict
        Dictionary with displacement data keyed by node name
    scale : float
        Scale factor applied to coordinates
    model : FEModel3D, optional
        FEA model to extract corner nodes from
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract displacement magnitudes for coloring and sizing
    displacement_magnitudes = [displacements.get(node_name, {}).get('total', 0.0) 
                               for node_name in vertex_node_names]
    
    if not displacement_magnitudes or all(d == 0.0 for d in displacement_magnitudes):
        print("Warning: No displacement data available for visualization")
        max_displacement = 1.0
    else:
        max_displacement = max(displacement_magnitudes)
    
    # Create colormap
    norm = Normalize(vmin=0, vmax=max_displacement if max_displacement > 0 else 1.0)
    cmap = cm.get_cmap('hot')
    
    # Draw members (lines connecting nodes)
    for edge_idx, edge in enumerate(line_indices):
        i_idx, j_idx = edge
        i_vertex = vertices[i_idx]
        j_vertex = vertices[j_idx]
        
        # Scale coordinates
        x_coords = [i_vertex[0] * scale, j_vertex[0] * scale]
        y_coords = [i_vertex[1] * scale, j_vertex[1] * scale]
        
        ax.plot(x_coords, y_coords, 'k-', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Draw corner frame if model is provided
    if model is not None:
        print(f"    [Debug] Drawing corner frame with scale={scale}")
        corner_coords = [
            ([0.0, 0.0], [1.0, 0.0]),  # Bottom
            ([1.0, 0.0], [1.0, 1.0]),  # Right
            ([1.0, 1.0], [0.0, 1.0]),  # Top
            ([0.0, 1.0], [0.0, 0.0]),  # Left
        ]
        
        for idx, (p1, p2) in enumerate(corner_coords):
            x_coords = [p1[0] * scale, p2[0] * scale]
            y_coords = [p1[1] * scale, p2[1] * scale]
            label_text = 'Corner Box' if idx == 0 else ""
            ax.plot(x_coords, y_coords, 'r--', linewidth=2.5, alpha=0.85, zorder=2, label=label_text)
        
        # Draw corner nodes as red stars
        for corner_coord in [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]:
            x_scaled = corner_coord[0] * scale
            y_scaled = corner_coord[1] * scale
            ax.plot(x_scaled, y_scaled, 'r*', markersize=20, zorder=5, markeredgecolor='darkred', markeredgewidth=1.5)
        print(f"    [Debug] Corner frame drawn successfully")
    
    # Draw nodes as circles with size and color based on displacement
    for node_idx, (vertex, node_name) in enumerate(zip(vertices, vertex_node_names)):
        x_scaled = vertex[0] * scale
        y_scaled = vertex[1] * scale
        
        # Get displacement for this node
        disp = displacements.get(node_name, {}).get('total', 0.0)
        
        # Size nodes: scale from 50 to 500 based on displacement
        # Zero displacement gets small size, max displacement gets large size
        min_size = 50
        max_size = 500
        if max_displacement > 0:
            node_size = min_size + (disp / max_displacement) * (max_size - min_size)
        else:
            node_size = min_size
        
        # Color based on displacement magnitude
        color = cmap(norm(disp))
        
        # Plot node as a circle
        circle = Circle((x_scaled, y_scaled), radius=node_size/500, 
                       color=color, ec='black', linewidth=1.5, zorder=3, alpha=0.8)
        ax.add_patch(circle)
        
        # Add node label for high-displacement nodes
        if disp > max_displacement * 0.5:  # Label high displacement nodes
            ax.text(x_scaled, y_scaled, node_name, ha='center', va='center', 
                   fontsize=7, fontweight='bold', color='white', zorder=4)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Displacement Magnitude (in)')
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Coordinate (scaled)', fontsize=11)
    ax.set_ylabel('Y Coordinate (scaled)', fontsize=11)
    ax.set_title('FEA Structure: Node Displacement Visualization\n(Marker size and color represent displacement magnitude)', 
                 fontsize=13, fontweight='bold')
    
    # Add legend showing displacement extrema
    stats_text = f'Max Disp: {max_displacement:.6e} in\n# Nodes: {len(vertex_node_names)}\n# Members: {len(line_indices)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax


def get_member_stresses_and_displacements(model, vertex_node_names, load_case='Combo 1'):
    """Extract stress and deflection data from members."""
    member_stress_data = {}
    
    for member_name, member in model.members.items():
        # Extract axial stress (average along member)
        try:
            axial_stress = 0.0
            # Get member nodes
            i_node = member.i_node
            j_node = member.j_node
            
            # Get displacements at ends
            if isinstance(i_node.DY, dict):
                i_dy = i_node.DY.get(load_case, 0.0)
            else:
                i_dy = i_node.DY
            
            if isinstance(j_node.DY, dict):
                j_dy = j_node.DY.get(load_case, 0.0)
            else:
                j_dy = j_node.DY
            
            # Calculate approximate stress from deflection
            # (This is a simplified metric - higher deflection = higher stress)
            avg_deflection = abs(float(i_dy)) + abs(float(j_dy))
            axial_stress = avg_deflection
            
            member_stress_data[member_name] = {
                'stress': axial_stress,
                'i_deflection': float(i_dy),
                'j_deflection': float(j_dy),
                'avg_deflection': avg_deflection / 2.0
            }
        except Exception as e:
            member_stress_data[member_name] = {'stress': 0.0, 'i_deflection': 0.0, 'j_deflection': 0.0, 'avg_deflection': 0.0}
    
    return member_stress_data


def calculate_section_area_from_stress(base_area, stress_level, scale_factor=2.0):
    """
    Calculate new section area based on stress level.
    Higher stress -> larger area
    Lower stress -> smaller area
    Bounded to avoid extremely thin or thick sections.
    """
    # Stress level is normalized (0 to 1)
    # Scale from 0.3x to 3.0x the base area
    min_scale = 0.3
    max_scale = 3.0
    area_scale = min_scale + (stress_level * (max_scale - min_scale))
    new_area = base_area * area_scale
    return new_area


def optimize_member_thicknesses(result, num_iterations=5, base_side_length=0.2, damping_factor=0.5):
    """
    Iteratively optimize member thicknesses based on deflection/stress.
    Saves iteration data to CSV and images for first and last iterations.
    
    Parameters:
    -----------
    result : dict
        FEA model result dictionary
    num_iterations : int
        Number of optimization iterations
    base_side_length : float
        Initial member cross-section side length (for square)
    damping_factor : float
        Damping factor (0-1) to smooth convergence and prevent oscillation.
        Lower values = smoother convergence, slower improvement.
        Higher values = faster changes, higher risk of oscillation.
    """
    import csv
    from datetime import datetime
    
    model = result["model"]
    vertices = result["vertices"]
    line_indices = result["line_indices"]
    vertex_node_names = result["vertex_node_names"]
    member_names = result["member_names"]
    scale = result["scale"]
    
    # Track iteration history
    iteration_history = []
    
    # Base section properties
    base_A = base_side_length ** 2
    base_I = (base_side_length ** 4) / 12
    base_J = (base_side_length ** 4) / 108
    
    # Create individual sections for each member
    print(f"Creating individual sections for {len(member_names)} members...")
    member_sections = {}
    for member_idx, member_name in enumerate(member_names):
        section_name = f"Section_{member_name}"
        model.add_section(section_name, base_A, base_I, base_I, base_J)
        
        # Update member to use its own section
        member = model.members[member_name]
        member.section = model.sections[section_name]
        
        # Track section properties for each member
        member_sections[member_name] = {
            'A': base_A,
            'Iy': base_I,
            'Iz': base_I,
            'J': base_J,
            'side_length': base_side_length,
            'section_name': section_name
        }
    
    print(f"\n{'='*70}")
    print(f"STARTING THICKNESS OPTIMIZATION LOOP")
    print(f"{'='*70}")
    print(f"Iterations: {num_iterations}")
    print(f"Base section: {base_side_length}\" × {base_side_length}\" square")
    print(f"Damping factor: {damping_factor}\n")
    
    for iteration in range(num_iterations):
        print(f"\n--- ITERATION {iteration + 1}/{num_iterations} ---")
        
        # Analyze current model
        model.analyze(check_stability=False)
        
        # Extract displacements
        min_node, max_node, displacements = extract_displacement_extrema(model, vertex_node_names)
        
        # Get member stresses
        member_stresses = get_member_stresses_and_displacements(model, vertex_node_names)
        
        # Calculate max stress for normalization
        max_stress = max([data['stress'] for data in member_stresses.values()]) if member_stresses else 1.0
        if max_stress == 0.0:
            max_stress = 1.0
        
        # Calculate metrics for this iteration
        max_displacement = max([d.get('total', 0.0) for d in displacements.values()]) if displacements else 0.0
        total_weight = sum([member_sections[m]['A'] * 50.0 for m in member_names])  # Rough estimate
        avg_stress = np.mean([data['stress'] for data in member_stresses.values()]) if member_stresses else 0.0
        
        iteration_data = {
            'Iteration': iteration + 1,
            'Max_Displacement_in': max_displacement,
            'Avg_Stress': avg_stress,
            'Total_Weight_estimate': total_weight,
            'Max_Stress': max_stress
        }
        
        print(f"  Max Displacement: {max_displacement:.6e} in")
        print(f"  Max Stress Level: {max_stress:.6e}")
        print(f"  Avg Stress Level: {avg_stress:.6e}")
        print(f"  Total Weight (est): {total_weight:.4f}")
        
        # Save visualization for first iteration
        if iteration == 0:
            print(f"  Saving first iteration visualization...")
            fig, ax = visualize_structure_with_displacements(
                vertices, line_indices, vertex_node_names, displacements, scale=scale, model=model
            )
            plt.savefig("optimization_iteration_001.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        iteration_history.append(iteration_data)
        
        # Update section areas based on stresses (except on last iteration)
        if iteration < num_iterations - 1:
            print(f"  Updating member sections (damping={damping_factor})...")
            
            for member_name in member_names:
                stress_data = member_stresses.get(member_name, {'stress': 0.0})
                stress_level = stress_data['stress'] / max_stress if max_stress > 0 else 0.0
                
                # Calculate target area based on stress
                target_A = calculate_section_area_from_stress(base_A, stress_level)
                target_side = np.sqrt(target_A)
                
                # Apply damping to smooth convergence
                current_A = member_sections[member_name]['A']
                current_side = member_sections[member_name]['side_length']
                
                # Blend between current and target using damping factor
                new_A = current_A + damping_factor * (target_A - current_A)
                new_side = current_side + damping_factor * (target_side - current_side)
                
                # Ensure positive dimensions
                new_A = max(new_A, 0.01)
                new_side = max(new_side, 0.1)
                
                # Update section properties for this member
                section_name = member_sections[member_name]['section_name']
                model.sections[section_name].A = new_A
                model.sections[section_name].Iy = (new_side ** 4) / 12
                model.sections[section_name].Iz = (new_side ** 4) / 12
                model.sections[section_name].J = (new_side ** 4) / 108
                
                # Update tracking
                member_sections[member_name]['A'] = new_A
                member_sections[member_name]['Iy'] = (new_side ** 4) / 12
                member_sections[member_name]['Iz'] = (new_side ** 4) / 12
                member_sections[member_name]['J'] = (new_side ** 4) / 108
                member_sections[member_name]['side_length'] = new_side
    
    # Save visualization for last iteration
    print(f"\n  Saving last iteration visualization...")
    model.analyze(check_stability=False)
    min_node, max_node, displacements = extract_displacement_extrema(model, vertex_node_names)
    fig, ax = visualize_structure_with_displacements(
        vertices, line_indices, vertex_node_names, displacements, scale=scale, model=model
    )
    plt.savefig(f"optimization_iteration_{num_iterations:03d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Save iteration history to CSV
    csv_filename = "optimization_history.csv"
    print(f"\nSaving optimization history to {csv_filename}...")
    
    if iteration_history:
        fieldnames = list(iteration_history[0].keys())
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(iteration_history)
        
        print(f"Optimization history saved: {csv_filename}")
        print(f"Total iterations recorded: {len(iteration_history)}")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION SUMMARY")
        print(f"{'='*70}")
        print(f"Initial Max Displacement: {iteration_history[0]['Max_Displacement_in']:.6e} in")
        print(f"Final Max Displacement:   {iteration_history[-1]['Max_Displacement_in']:.6e} in")
        print(f"Change: {((iteration_history[-1]['Max_Displacement_in'] - iteration_history[0]['Max_Displacement_in']) / iteration_history[0]['Max_Displacement_in'] * 100):.2f}%" if iteration_history[0]['Max_Displacement_in'] > 0 else "N/A")
        print(f"Initial Total Weight:     {iteration_history[0]['Total_Weight_estimate']:.4f}")
        print(f"Final Total Weight:       {iteration_history[-1]['Total_Weight_estimate']:.4f}")
        print(f"{'='*70}\n")
    
    return iteration_history, member_sections


def build_model_from_clipped_geometry(npz_path=None, scale=10.0, decimals=12):
    geometry_path, vertices, line_indices = load_clipped_geometry(npz_path=npz_path)

    model = FEModel3D()
    node_names_by_point, vertex_node_names = add_vertex_nodes(model, vertices, scale=scale, decimals=decimals)
    support_node_names = add_boundary_supports(model, vertices, node_names_by_point, scale=scale, decimals=decimals)

    # Define material properties for bone
    # Young's modulus for cortical bone (~20 GPa = 2.9e6 psi)
    E = 2.9e6  # psi
    # Shear modulus (approximated from Poisson's ratio)
    G = 1.1e6  # psi
    nu = 0.3   # Poisson's ratio
    rho = 0.077  # lb/in^3 (cortical bone density)

    model.add_material('Bone', E, G, nu, rho)

    # Define section properties for a square bar
    # 0.2" x 0.2" square cross-section for reasonable deflections
    A = 0.2 * 0.2  # in^2
    # For square cross-section: I = (b^4)/12
    I = (0.2**4) / 12  # in^4
    Iy = I
    Iz = I
    # Torsional constant for square: J ≈ (b^4)/108
    J = (0.2**4) / 108  # in^4

    model.add_section('SquareBar', A, Iy, Iz, J)

    # Add members connecting vertices along the edges
    member_names = add_members_from_edges(model, vertices, line_indices, node_names_by_point, 
                                         material_name='Bone', section_name='SquareBar')

    # Add corner connections (box perimeter without diagonals)
    corner_nodes, corner_member_names = add_corner_connections(model, node_names_by_point, scale=scale, include_diagonals=False)
    
    # Add corner nodes to the analysis list
    corner_node_names = list(corner_nodes.values())
    all_vertex_node_names = list(vertex_node_names) + corner_node_names
    
    # Combine all member names
    all_member_names = list(member_names) + corner_member_names

    return {
        "model": model,
        "geometry_path": geometry_path,
        "vertices": vertices,
        "line_indices": line_indices,
        "vertex_node_names": all_vertex_node_names,
        "interior_node_names": vertex_node_names,
        "corner_node_names": corner_node_names,
        "support_node_names": support_node_names,
        "member_names": all_member_names,
        "interior_member_names": member_names,
        "corner_member_names": corner_member_names,
        "corner_nodes": corner_nodes,
        "scale": scale,
    }


def create_cylinder_mesh(p1, p2, radius, num_segments=16):
    """
    Create a triangular mesh for a cylinder between two points.
    
    Parameters:
    -----------
    p1, p2 : array-like
        Start and end points of the cylinder (3D)
    radius : float
        Radius of the cylinder
    num_segments : int
        Number of segments around the circumference
    
    Returns:
    --------
    vertices : ndarray
        Nx3 array of vertex coordinates
    faces : ndarray
        Mx3 array of triangle indices
    """
    # Ensure 3D points
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if p1.shape[0] == 2:
        p1 = np.append(p1, 0.0)
    if p2.shape[0] == 2:
        p2 = np.append(p2, 0.0)
    
    # Cylinder axis
    axis = p2 - p1
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-10:
        return np.array([]), np.array([])
    
    axis_unit = axis / axis_length

    # Trim ends slightly to avoid coplanar cap overlap at shared joints.
    end_trim = min(radius * 0.2, axis_length * 0.1)
    if axis_length <= 2.0 * end_trim:
        return np.array([]), np.array([])
    p1 = p1 + axis_unit * end_trim
    p2 = p2 - axis_unit * end_trim
    
    # Create perpendicular vectors
    if abs(axis_unit[2]) < 0.9:
        perp1 = np.array([-axis_unit[1], axis_unit[0], 0.0])
    else:
        perp1 = np.array([1.0, 0.0, 0.0])
    
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis_unit, perp1)
    perp2 /= np.linalg.norm(perp2)
    
    # Create circle points
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    circle_p1 = np.array([p1 + radius * (np.cos(a) * perp1 + np.sin(a) * perp2) for a in angles])
    circle_p2 = np.array([p2 + radius * (np.cos(a) * perp1 + np.sin(a) * perp2) for a in angles])
    
    # Combine vertices
    vertices = np.vstack([circle_p1, circle_p2, [p1], [p2]])
    
    # Create faces
    faces = []
    idx_p1_circle = np.arange(num_segments)
    idx_p2_circle = np.arange(num_segments, 2 * num_segments)
    idx_p1_center = 2 * num_segments
    idx_p2_center = 2 * num_segments + 1
    
    # Side faces
    for i in range(num_segments):
        i_next = (i + 1) % num_segments
        # Two triangles per segment
        faces.append([idx_p1_circle[i], idx_p1_circle[i_next], idx_p2_circle[i]])
        faces.append([idx_p1_circle[i_next], idx_p2_circle[i_next], idx_p2_circle[i]])
    
    # Cap faces (p1 end)
    for i in range(num_segments):
        i_next = (i + 1) % num_segments
        faces.append([idx_p1_center, idx_p1_circle[i_next], idx_p1_circle[i]])
    
    # Cap faces (p2 end)
    for i in range(num_segments):
        i_next = (i + 1) % num_segments
        faces.append([idx_p2_center, idx_p2_circle[i], idx_p2_circle[i_next]])
    
    return vertices, np.array(faces, dtype=np.uint32)


def _member_geometry_key(p1, p2, decimals=8):
    """Return an orientation-invariant key for a member's geometry."""
    p1r = tuple(np.round(np.asarray(p1, dtype=float), decimals=decimals))
    p2r = tuple(np.round(np.asarray(p2, dtype=float), decimals=decimals))
    return tuple(sorted((p1r, p2r)))


def generate_optimized_structure_stl(result, member_sections, scale=10.0, filename="optimized_structure.stl"):
    """
    Generate an STL file from the optimized frame structure with varying member thicknesses.
    
    Parameters:
    -----------
    result : dict
        Result dictionary from build_model_from_clipped_geometry
    member_sections : dict
        Dictionary with member section properties (from optimization)
    scale : float
        Scale factor applied to coordinates
    filename : str
        Output STL filename
    """
    vertices_list = []
    faces_list = []
    vertex_offset = 0
    
    model = result["model"]
    vertices = result["vertices"]
    member_names = result["member_names"]
    
    print(f"\nGenerating STL from optimized structure...")
    print(f"Processing {len(member_names)} members with varying thicknesses...")
    
    # Skip coincident/reversed duplicate members to avoid overlapping cylinders.
    seen_member_geometry = set()
    duplicate_member_count = 0

    # Generate geometry for each member
    for member_idx, member_name in enumerate(member_names):
        member = model.members[member_name]
        
        # Get member endpoints
        i_node = member.i_node
        j_node = member.j_node
        
        p1 = np.array([i_node.X, i_node.Y, i_node.Z])
        p2 = np.array([j_node.X, j_node.Y, j_node.Z])

        geom_key = _member_geometry_key(p1, p2)
        if geom_key in seen_member_geometry:
            duplicate_member_count += 1
            continue
        seen_member_geometry.add(geom_key)
        
        # Get member thickness (convert from square side length to radius)
        section_data = member_sections.get(member_name, {'side_length': 0.2})
        side_length = section_data['side_length']
        
        # For a square section, approximate radius as half the diagonal / 2
        # This gives a reasonable cylinder radius proportional to the square area
        radius = (side_length / 2.0) * 0.7  # Slightly smaller than half-width
        radius = max(radius, 0.01)  # Minimum radius
        
        # Create cylinder mesh
        cylinder_verts, cylinder_faces = create_cylinder_mesh(p1, p2, radius)
        
        if len(cylinder_verts) > 0:
            # Add faces with offset
            faces_list.append(cylinder_faces + vertex_offset)
            vertices_list.append(cylinder_verts)
            vertex_offset += len(cylinder_verts)

    if duplicate_member_count > 0:
        print(f"Skipped {duplicate_member_count} duplicate/reversed members during STL export")
    
    if not vertices_list:
        print("No geometry generated!")
        return
    
    # Combine all vertices and faces
    all_vertices = np.vstack(vertices_list)
    all_faces = np.vstack(faces_list)
    
    print(f"Total vertices: {len(all_vertices)}")
    print(f"Total faces: {len(all_faces)}")
    
    # Write STL file
    write_stl_file(filename, all_vertices, all_faces)
    print(f"STL file saved: {filename}")


def write_stl_file(filename, vertices, faces):
    """
    Write vertices and faces to an ASCII STL file.
    
    Parameters:
    -----------
    filename : str
        Output filename
    vertices : ndarray
        Nx3 array of vertex coordinates
    faces : ndarray
        Mx3 array of triangle indices
    """
    with open(filename, 'w') as f:
        f.write("solid optimized_structure\n")
        
        for face in faces:
            # Get the three vertices of the triangle
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Calculate normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = np.array([0, 0, 1])
            
            # Write facet
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid optimized_structure\n")


def add_corner_connections(model, node_names_by_point, scale=10.0, include_diagonals=False):
    """
    Connect the four corner nodes of the unit square with edge members.
    
    Parameters:
    -----------
    model : FEModel3D
        The FEA model
    node_names_by_point : dict
        Mapping from point key to node name
    scale : float
        Scale factor applied to coordinates
    include_diagonals : bool
        Whether to include diagonal bracing (X pattern)
    
    Returns:
    --------
    corner_nodes : dict
        Dictionary of corner nodes
    corner_member_names : list
        List of newly created corner member names
    """
    corners = [
        ([0.0, 0.0], "SW"),
        ([1.0, 0.0], "SE"),
        ([1.0, 1.0], "NE"),
        ([0.0, 1.0], "NW"),
    ]
    
    corner_nodes = {}
    corner_member_names = []
    
    # Find or create corner nodes
    for corner_coord, label in corners:
        key = _point_key(corner_coord)
        if key in node_names_by_point:
            corner_nodes[label] = node_names_by_point[key]
        else:
            # Create corner node if it doesn't exist
            node_name = f"Corner_{label}"
            x_scaled = corner_coord[0] * scale
            y_scaled = corner_coord[1] * scale
            model.add_node(node_name, x_scaled, y_scaled, 0.0)
            corner_nodes[label] = node_name
            node_names_by_point[key] = node_name
    
    # Connect corners: edges only (box perimeter)
    corner_connections = [
        ("SW", "SE", "Edge_Bottom"),
        ("SE", "NE", "Edge_Right"),
        ("NE", "NW", "Edge_Top"),
        ("NW", "SW", "Edge_Left"),
    ]
    
    # Optionally add diagonals
    if include_diagonals:
        corner_connections.extend([
            ("SW", "NE", "Diag_1"),
            ("SE", "NW", "Diag_2"),
        ])
    
    print(f"Adding {len(corner_connections)} corner connections...")
    
    for node1_label, node2_label, member_label in corner_connections:
        node1 = corner_nodes[node1_label]
        node2 = corner_nodes[node2_label]
        
        try:
            # Add member with a small cross-section for corners
            model.add_member(member_label, node1, node2, 'Bone', 'SquareBar')
            corner_member_names.append(member_label)
            print(f"  Added: {member_label} ({node1} - {node2})")
        except Exception as e:
            print(f"  Warning: Could not add {member_label}: {e}")
    
    return corner_nodes, corner_member_names


if __name__ == "__main__":
    result = build_model_from_clipped_geometry()
    model = result["model"]

    print(f"Loaded geometry: {result['geometry_path']}")
    print(f"Vertex nodes added: {len(result['vertex_node_names'])}")
    print(f"Members created: {len(result['member_names'])}")
    print(f"Line segments available: {len(result['line_indices'])}")
    print(f"Geometry scale factor: {result['scale']}")
    print(f"Scaled domain: [0, {result['scale']}] × [0, {result['scale']}]")
    print("Corner supports:")
    for corner, node_name in result["support_node_names"].items():
        print(f"  {node_name}: {tuple(c * result['scale'] for c in corner)}")

    # Apply uniform loading (scaled to be reasonable relative to member stiffness)
    FORCE_PER_NODE = 100.0  # units per node
    print(f"\nApplying uniform downward force ({FORCE_PER_NODE} units per node) to all vertex nodes...")
    apply_uniform_loading(model, result['vertex_node_names'], force_magnitude=-FORCE_PER_NODE, direction='FY')

    # Analyze the model (check_stability=False suppresses singular matrix warnings)
    if len(result['member_names']) > 0:
        print("Running analysis...")
        model.analyze(check_stability=False)

        # Extract displacement extrema
        min_node, max_node, displacements = extract_displacement_extrema(model, result['vertex_node_names'])

        if min_node and max_node:
            min_name, min_data = min_node
            max_name, max_data = max_node

            # Compute total applied load and reaction forces
            force_per_node = 100.0
            total_applied_load = len(result['vertex_node_names']) * force_per_node
            total_reaction = 0.0
            for corner_key, node_name in result["support_node_names"].items():
                node = model.nodes[node_name]
                rxn_fy = node.RxnFY.get('Combo 1', 0.0) if isinstance(node.RxnFY, dict) else node.RxnFY
                if not (isinstance(rxn_fy, float) and np.isnan(rxn_fy)):
                    total_reaction += abs(rxn_fy)

            print(f"\n{'='*70}")
            print(f"LOAD AND REACTION SUMMARY")
            print(f"{'='*70}")
            print(f"Total Applied Load: {total_applied_load:.2f} units")
            print(f"Total Support Reactions: {total_reaction:.2f} units")

            print(f"\n{'='*70}")
            print(f"DISPLACEMENT ANALYSIS RESULTS")
            print(f"{'='*70}")
            print(f"\nMinimum Displacement Node: {min_name}")
            print(f"  Total Magnitude: {min_data['total']:.6e} in")
            print(f"  DX: {min_data['dx']:.6e} in")
            print(f"  DY: {min_data['dy']:.6e} in")
            print(f"  DZ: {min_data['dz']:.6e} in")

            print(f"\nMaximum Displacement Node: {max_name}")
            print(f"  Total Magnitude: {max_data['total']:.6e} in")
            print(f"  DX: {max_data['dx']:.6e} in")
            print(f"  DY: {max_data['dy']:.6e} in")
            print(f"  DZ: {max_data['dz']:.6e} in")

            print(f"\nDisplacement Range: {min_data['total']:.6e} - {max_data['total']:.6e} in")
            print(f"{'='*70}\n")

            # Visualize structure with displacements
            print("Generating displacement visualization...")
            fig, ax = visualize_structure_with_displacements(
                result['vertices'], result['line_indices'], 
                result['vertex_node_names'], displacements, 
                scale=result['scale'], model=model
            )
            # Save the figure instead of showing it
            visualization_path = "displacement_visualization.png"
            plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {visualization_path}")
            plt.close(fig)

            # Plot first member
            first_member = result['member_names'][0]
            print(f"Analyzing first member: {first_member}")
            model.members[first_member].plot_shear('Fy')
            model.members[first_member].plot_moment('Mz')
            model.members[first_member].plot_deflection('dy')
            
            # Run optimization loop
            print("\n" + "="*70)
            optimization_history, member_sections = optimize_member_thicknesses(result, num_iterations=100, base_side_length=0.2, damping_factor=0.3)
            print("="*70)
            
            # Generate STL from optimized structure
            print("\n" + "="*70)
            generate_optimized_structure_stl(result, member_sections, scale=result['scale'], filename="optimized_structure.stl")
            print("="*70)
        else:
            print("\nNo displacement data available.")
    else:
        print("\nNo members were created.")