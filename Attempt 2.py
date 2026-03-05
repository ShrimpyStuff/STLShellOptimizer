import trimesh
import numpy as np
import networkx as nx
import random
import concurrent.futures
from functools import partial
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import os

# =========================
# 1. PERIMETER-ONLY GRID
# =========================
def create_shell_grid(mesh_path, spacing=4.0):
    """Creates a grid ONLY on the surface/shell of the STL."""
    if mesh_path and os.path.exists(mesh_path):
        mesh = trimesh.load(mesh_path)
    else:
        print("Backup: Generating 15x15x40 shell...")
        mesh = trimesh.creation.box(extents=[15, 15, 40])

    bounds = mesh.bounds
    s = spacing / 1000.0
    
    # Slice the mesh vertically
    z_range = np.arange(bounds[0][2]/1000, (bounds[1][2]/1000) + (s/2), s)
    pts_list = []
    
    for z in z_range:
        # Create a 2D cross-section at height Z
        section = mesh.section(plane_origin=[0, 0, z*1000], plane_normal=[0, 0, 1])
        if section:
            # We snap the vertices of the slice to the grid spacing
            # This ensures vertical alignment for diamonds
            snapped = np.round((section.vertices / 1000.0) / s) * s
            pts_list.extend(snapped)

    if not pts_list:
        print("Error: No surface points found. Check STL scale.")
        return None, None, None

    pts = np.unique(np.array(pts_list), axis=0)
    tree = cKDTree(pts)
    
    # Neighbors: 1.66 * s allows for vertical, horizontal, AND 45-degree diagonals
    edges = np.array(list(tree.query_pairs(s * 1.66)))
    
    # Boundary nodes (Top and Bottom)
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    anchors = np.where((pts[:, 2] <= z_min + 0.0001) | (pts[:, 2] >= z_max - 0.0001))[0]
    
    return pts, edges, anchors

# =========================
# 2. THE SHELL-STABILITY SOLVER
# =========================
def solve_shell(mask, pts, edges, boundary_nodes, E=3e9, A=1e-6):
    active_mask = mask.copy()
    active_mask[boundary_nodes] = 1 # Force the top and bottom to stay
    
    active_idx = np.where(active_mask == 1)[0]
    n_act = len(active_idx)
    
    # Create Graph to check for "Floating Islands" and "Lonely Sticks"
    G = nx.Graph()
    G.add_nodes_from(active_idx)
    for i, j in edges:
        if active_mask[i] and active_mask[j]:
            G.add_edge(i, j)

    # 1. CONNECTIVITY GUARD: Structure must be one solid piece
    if not nx.is_connected(G.subgraph(active_idx)):
        # If the shell is broken into floating pieces, give a huge penalty
        return 1e18, n_act

    # 2. MAXWELL RIGIDITY: Every node needs 3+ bars to form a stable diamond
    lonely_nodes = sum(1 for n in active_idx if G.degree(n) < 3)

    # 3. FEA ASSEMBLY
    map_act = -np.ones(len(pts), dtype=int)
    map_act[active_idx] = np.arange(n_act)
    K = lil_matrix((n_act*3, n_act*3))
    
    for i, j in edges:
        if active_mask[i] and active_mask[j]:
            p1, p2 = pts[i], pts[j]
            L = np.linalg.norm(p2 - p1)
            l = (p2 - p1) / L
            ke = (E * A / L) * np.block([[np.outer(l,l), -np.outer(l,l)], [-np.outer(l,l), np.outer(l,l)]])
            u, v = map_act[i], map_act[j]
            dofs = [u*3, u*3+1, u*3+2, v*3, v*3+1, v*3+2]
            for a in range(6):
                for b in range(6): K[dofs[a], dofs[b]] += ke[a,b]

    # Load and Constraints
    z_max = pts[:, 2].max()
    t_nodes = [n for n in boundary_nodes if np.isclose(pts[n, 2], z_max)]
    b_nodes = [n for n in boundary_nodes if not np.isclose(pts[n, 2], z_max)]
    
    F = np.zeros(n_act*3)
    for n in t_nodes: F[map_act[n]*3 + 2] = -5.0 # Load
    
    fixed = [map_act[n]*3 + d for n in b_nodes for d in range(3)]
    free = np.setdiff1d(np.arange(n_act*3), fixed)

    try:
        K_ff = K.tocsr()[free, :][:, free] + eye(len(free)) * 1e-5
        u_f = spsolve(K_ff, F[free])
        compliance = abs(np.dot(F[free], u_f))
        return compliance + (lonely_nodes * 0.8), n_act
    except:
        return 1e18, n_act

# =========================
# 3. MAIN EVOLUTION
# =========================
if __name__ == "__main__":
    STL_FILE = "inputa.stl" 
    GRID_SPACING = 2.5 # mm
    
    pts, edges, anchors = create_shell_grid(STL_FILE, spacing=GRID_SPACING)
    if pts is None: exit()

    n_nodes = len(pts)
    population = [np.random.choice([0, 1], size=n_nodes, p=[0.2, 0.8]) for _ in range(30)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for gen in range(40):
            fn = partial(solve_shell, pts=pts, edges=edges, boundary_nodes=anchors)
            results = list(executor.map(fn, population))
            
            fits = [(r[0] + r[1]*1e-3) for r in results]
            idx = np.argsort(fits)
            population = [population[i] for i in idx]
            
            print(f"Gen {gen:02d} | Nodes: {int(np.sum(population[0]))}/{n_nodes}")

            # Reproduction
            next_gen = population[:4]
            while len(next_gen) < 30:
                p1, p2 = random.sample(population[:10], 2)
                child = np.where(np.random.rand(n_nodes) > 0.5, p1, p2)
                if random.random() < 0.2:
                    child[random.randint(0, n_nodes-1)] = 1 - child[random.randint(0, n_nodes-1)]
                next_gen.append(child)
            population = next_gen

    # EXPORT
    best = population[0]
    best[anchors] = 1
    scene = trimesh.Scene()
    for i in np.where(best == 1)[0]:
        scene.add_geometry(trimesh.creation.uv_sphere(radius=0.4, count=[6, 6]).apply_translation(pts[i]*1000))
    for i, j in edges:
        if best[i] and best[j]:
            p1, p2 = pts[i]*1000, pts[j]*1000
            vec = p2 - p1
            cyl = trimesh.creation.cylinder(radius=0.3, height=np.linalg.norm(vec), sections=6)
            mat = trimesh.geometry.align_vectors([0,0,1], vec)
            mat[:3, 3] = (p1+p2)/2
            scene.add_geometry(cyl.apply_transform(mat))
    scene.to_geometry().export("shell_only_lattice.stl")