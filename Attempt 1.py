import trimesh
import numpy as np
import networkx as nx
import random
import concurrent.futures
import warnings
from functools import partial
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import spsolve, MatrixRankWarning
import matplotlib.pyplot as plt

# =========================
# 1. THE CORE FEA ENGINE
# =========================
def solve_truss(mask, pts, edges, b_nodes, t_nodes, E, A):
    active_idx = np.where(mask == 1)[0]
    n_act = len(active_idx)
    if n_act < 4: return None, None, None
    
    # Map global node index to local solver index
    map_act = -np.ones(len(pts), dtype=int)
    map_act[active_idx] = np.arange(n_act)
    
    K = lil_matrix((n_act*3, n_act*3))
    valid_edge_indices = []
    
    for idx, (i, j) in enumerate(edges):
        if mask[i] and mask[j]:
            p1, p2 = pts[i], pts[j]
            L_vec = p2 - p1
            L = np.linalg.norm(L_vec)
            if L < 1e-9: continue
            
            l = L_vec / L
            gamma = np.array([l[0], l[1], l[2]])
            ke_sub = np.outer(gamma, gamma)
            ke = (E * A / L) * np.block([[ke_sub, -ke_sub], [-ke_sub, ke_sub]])
            
            u_idx, v_idx = map_act[i], map_act[j]
            dofs = [u_idx*3, u_idx*3+1, u_idx*3+2, v_idx*3, v_idx*3+1, v_idx*3+2]
            for a in range(6):
                for b in range(6):
                    K[dofs[a], dofs[b]] += ke[a,b]
            valid_edge_indices.append(idx)

    F = np.zeros(n_act*3)
    for n in t_nodes:
        if map_act[n] != -1: F[map_act[n]*3 + 2] = -1.0 
    
    fixed = []
    for n in b_nodes:
        if map_act[n] != -1: fixed.extend([n*3, n*3+1, n*3+2])
    
    free = np.setdiff1d(np.arange(n_act*3), fixed)
    if len(free) == 0: return None, None, None

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=MatrixRankWarning)
        try:
            K_csr = K.tocsr()
            K_ff = K_csr[free, :][:, free] + eye(len(free)) * 1e-4 
            u_f = spsolve(K_ff, F[free])
            u = np.zeros(n_act*3)
            u[free] = u_f
            
            bar_forces = np.zeros(len(edges))
            for idx in valid_edge_indices:
                i, j = edges[idx]
                L = np.linalg.norm(pts[j]-pts[i])
                l = (pts[j]-pts[i])/L
                u_i = u[map_act[i]*3 : map_act[i]*3+3]
                u_j = u[map_act[j]*3 : map_act[j]*3+3]
                bar_forces[idx] = abs((E*A/L) * np.dot((u_j - u_i), l))
                
            compliance = np.dot(F[free], u_f)
            return compliance, bar_forces, active_idx
        except:
            return None, None, None

# =========================
# 2. FITNESS & MUTATION
# =========================
def fitness_worker(mask, pts, edges, b_nodes, t_nodes, E, A, vol_pen):
    comp, forces, active = solve_truss(mask, pts, edges, b_nodes, t_nodes, E, A)
    if comp is None or comp <= 0: return 1e15, np.zeros(len(edges))
    
    G = nx.Graph()
    # Ensure all active nodes are added even if they have no edges
    G.add_nodes_from(active)
    G.add_edges_from([(i,j) for idx, (i,j) in enumerate(edges) if mask[i] and mask[j]])
    
    # Rigidity check: Only look at nodes that are actually in the graph
    instability = sum(1 for n in active if G.degree(n) < 3)
    
    score = abs(comp) + (len(active) * vol_pen) + (instability * 0.5)
    return score, forces

def mutate_by_stress(child, forces, edges, b_nodes, t_nodes):
    active_forces = forces[forces > 0]
    if len(active_forces) < 5: return child
    
    threshold = np.percentile(active_forces, 30)
    lazy_edges = np.where((forces < threshold) & (forces > 0))[0]
    
    if len(lazy_edges) > 0:
        for _ in range(min(10, len(lazy_edges))):
            idx = random.choice(lazy_edges)
            node = random.choice(edges[idx])
            if node not in b_nodes and node not in t_nodes:
                child[node] = 0
    return child

# =========================
# 3. STL EXPORTER
# =========================
def export_truss_to_stl(pts, edg, mask, filename="optimized_shell.stl", r_val=0.0005):
    active_mask = mask.astype(bool)
    scene = trimesh.Scene()
    pts_mm = pts * 1000.0
    r_mm = r_val * 1000.0
    
    active_nodes = np.where(active_mask)[0]
    for idx in active_nodes:
        sphere = trimesh.creation.uv_sphere(radius=r_mm)
        sphere.apply_translation(pts_mm[idx])
        scene.add_geometry(sphere)

    count = 0
    for i, j in edg:
        if active_mask[i] and active_mask[j]:
            p1, p2 = pts_mm[i], pts_mm[j]
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length < 1e-6: continue
            cyl = trimesh.creation.cylinder(radius=r_mm, height=length)
            matrix = trimesh.geometry.align_vectors([0, 0, 1], vec)
            matrix[:3, 3] = (p1 + p2) / 2.0
            cyl.apply_transform(matrix)
            scene.add_geometry(cyl)
            count += 1
    
    if count > 0:
        scene.to_geometry().export(filename)
        print(f"Exported {filename}")

# =========================
# 4. MAIN EXECUTION
# =========================
if __name__ == "__main__":
    mesh = trimesh.creation.box(extents=[15, 15, 40])
    pts_raw, _ = trimesh.sample.sample_surface(mesh, 800)
    pts = pts_raw / 1000.0
    
    tri = Delaunay(pts)
    edges_set = set()
    for s in tri.simplices:
        for i in range(3):
            for j in range(i+1,3): edges_set.add(tuple(sorted([s[i], s[j]])))
    edges = np.array(list(edges_set))
    dist = np.linalg.norm(pts[edges[:,0]] - pts[edges[:,1]], axis=1)
    edges = edges[dist < (np.mean(dist) * 2.5)]
    
    n_nodes = len(pts)
    z_min, z_max = pts[:,2].min(), pts[:,2].max()
    bot_n = np.where(pts[:,2] <= z_min + 0.001)[0]
    top_n = np.where(pts[:,2] >= z_max - 0.001)[0]
    
    POP_SIZE, N_GEN = 40, 75
    E_mod, A_area, VOL_PEN = 3e9, 1e-6, 8e-4
    population = [np.ones(n_nodes, dtype=int) for _ in range(POP_SIZE)]
    best_forces = np.zeros(len(edges))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for gen in range(N_GEN):
            fn = partial(fitness_worker, pts=pts, edges=edges, b_nodes=bot_n, t_nodes=top_n, E=E_mod, A=A_area, vol_pen=VOL_PEN)
            results = list(executor.map(fn, population))
            
            fits = [r[0] for r in results]
            idx = np.argsort(fits)
            population = [population[i] for i in idx]
            best_forces = results[idx[0]][1]
            
            print(f"Gen {gen:02d} | Best Fit: {fits[idx[0]]:.4e} | Nodes: {int(np.sum(population[0]))}/{n_nodes}")
            
            next_gen = population[:4]
            while len(next_gen) < POP_SIZE:
                p1, p2 = random.sample(population[:15], 2)
                child = np.where(np.random.rand(n_nodes) > 0.5, p1, p2)
                child = mutate_by_stress(child, best_forces, edges, bot_n, top_n)
                next_gen.append(child)
            population = next_gen

    export_truss_to_stl(pts, edges, population[0])