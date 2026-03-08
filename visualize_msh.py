#!/usr/bin/env python3
"""
Visualize a Gmsh .msh file in 3D.

- If triangle surface elements exist, plots them directly.
- If tetrahedral elements exist, computes and plots boundary faces.
- Highlights disconnected nodes (not referenced by plotted faces).

Usage:
    python3 visualize_msh.py shell_tetrahedral.msh
    python3 visualize_msh.py shell_tetrahedral.msh --show-all-nodes
"""

import argparse
from collections import Counter
from itertools import combinations

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _build_tag_to_index(node_tags):
    return {int(tag): i for i, tag in enumerate(node_tags)}


def _parse_elements(tag_to_index):
    """Return triangle and tetra connectivity arrays using 0-based node indexing."""
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements()

    triangles = []
    tetras = []

    for etype, node_tags_flat in zip(elem_types, elem_node_tags):
        # etype 2: 3-node triangle
        # etype 4: 4-node tetrahedron
        # higher-order elements are handled by taking corner nodes only
        if etype == 2:
            arr = np.asarray(node_tags_flat, dtype=np.int64).reshape(-1, 3)
            triangles.append(arr)
        elif etype == 4:
            arr = np.asarray(node_tags_flat, dtype=np.int64).reshape(-1, 4)
            tetras.append(arr)
        elif etype in {9, 11}:  # 6-node triangle, 10-node tetra
            npe = 6 if etype == 9 else 10
            arr = np.asarray(node_tags_flat, dtype=np.int64).reshape(-1, npe)
            if etype == 9:
                triangles.append(arr[:, :3])
            else:
                tetras.append(arr[:, :4])

    tri = np.vstack(triangles) if triangles else np.empty((0, 3), dtype=np.int64)
    tet = np.vstack(tetras) if tetras else np.empty((0, 4), dtype=np.int64)

    # Map Gmsh node tags to compact 0-based indices in node array order.
    if len(tri):
        tri = np.vectorize(tag_to_index.__getitem__)(tri)
    if len(tet):
        tet = np.vectorize(tag_to_index.__getitem__)(tet)

    return tri.astype(np.int32), tet.astype(np.int32)


def _tet_boundary_faces(tets):
    """Return boundary triangle faces from tetra connectivity."""
    if len(tets) == 0:
        return np.empty((0, 3), dtype=np.int32)

    face_count = Counter()
    face_owner = {}

    for tet in tets:
        a, b, c, d = tet
        faces = [
            (a, b, c),
            (a, b, d),
            (a, c, d),
            (b, c, d),
        ]
        for f in faces:
            key = tuple(sorted(f))
            face_count[key] += 1
            face_owner[key] = f

    boundary = [face_owner[k] for k, c in face_count.items() if c == 1]
    return np.asarray(boundary, dtype=np.int32)


def _set_equal_axes(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def visualize_msh(path, alpha=0.55, show_all_nodes=False):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    try:
        gmsh.open(path)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        tag_to_index = _build_tag_to_index(node_tags)

        tri, tet = _parse_elements(tag_to_index)

    finally:
        gmsh.finalize()

    if len(points) == 0:
        raise RuntimeError(f"No nodes found in {path}")

    # Prefer true triangle surfaces; otherwise extract tet boundary.
    surface = tri if len(tri) else _tet_boundary_faces(tet)

    used = np.unique(surface.ravel()) if len(surface) else np.array([], dtype=np.int32)
    all_idx = np.arange(len(points), dtype=np.int32)
    disconnected = np.setdiff1d(all_idx, used, assume_unique=False)

    print(f"Loaded: {path}")
    print(f"Nodes: {len(points)}")
    print(f"Triangles: {len(tri)}")
    print(f"Tetrahedra: {len(tet)}")
    print(f"Plotted surface faces: {len(surface)}")
    print(f"Disconnected nodes: {len(disconnected)}")

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    if len(surface):
        poly = Poly3DCollection(
            points[surface],
            linewidths=0.2,
            edgecolors=(0.2, 0.35, 0.4, 0.25),
            facecolors=(0.35, 0.7, 0.9, alpha),
        )
        ax.add_collection3d(poly)

    if show_all_nodes:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c="k", alpha=0.25, label="All nodes")

    if len(disconnected):
        p = points[disconnected]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=20, c="red", alpha=0.95, label="Disconnected")

    _set_equal_axes(ax, points)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("MSH Visualization")
    if show_all_nodes or len(disconnected):
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a .msh file")
    parser.add_argument("msh_path", nargs="?", default="lattice.msh", help="Path to .msh file")
    parser.add_argument("--alpha", type=float, default=0.55, help="Surface transparency (0..1)")
    parser.add_argument("--show-all-nodes", action="store_true", help="Overlay all mesh nodes")
    args = parser.parse_args()

    visualize_msh(args.msh_path, alpha=args.alpha, show_all_nodes=args.show_all_nodes)
