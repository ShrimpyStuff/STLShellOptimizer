from Pynite import FEModel3D
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


def extract_vertices_and_lines(lines, decimals=12):
    """Build unique vertex list and line connectivity from edited segments."""
    vertex_index = {}
    vertices = []
    line_indices = []
    line_segments = []

    for segment in lines:
        p1 = np.asarray(segment[0], dtype=float)
        p2 = np.asarray(segment[1], dtype=float)

        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue

        key1 = tuple(np.round(p1, decimals=decimals))
        key2 = tuple(np.round(p2, decimals=decimals))

        if key1 not in vertex_index:
            vertex_index[key1] = len(vertices)
            vertices.append([float(p1[0]), float(p1[1])])

        if key2 not in vertex_index:
            vertex_index[key2] = len(vertices)
            vertices.append([float(p2[0]), float(p2[1])])

        i1 = vertex_index[key1]
        i2 = vertex_index[key2]
        line_indices.append([i1, i2])
        line_segments.append([[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]])

    return (
        np.asarray(vertices, dtype=float),
        np.asarray(line_indices, dtype=int),
        np.asarray(line_segments, dtype=float),
    )


def plot_line_segments(line_segments, color='red', linewidth=1.0):
    """Plot a list of line segments."""
    for p1, p2 in line_segments:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth)


def plot_unit_square(color='black', linewidth=1.5, linestyle='--'):
    """Plot the [0, 1] x [0, 1] bounding box."""
    square_x = [0.0, 1.0, 1.0, 0.0, 0.0]
    square_y = [0.0, 0.0, 1.0, 1.0, 0.0]
    plt.plot(square_x, square_y, color=color, linewidth=linewidth, linestyle=linestyle, label='Unit Square')


def clip_segment_to_unit_square(p1, p2, eps=1e-12):
    """Clip a segment to the unit square [0, 1] x [0, 1]."""
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    dx = x2 - x1
    dy = y2 - y1
    t0 = 0.0
    t1 = 1.0

    def update(p, q, t0, t1):
        if np.isclose(p, 0.0, atol=eps):
            if q < -eps:
                return None
            return t0, t1

        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)

        return t0, t1

    for p, q in ((-dx, x1), (dx, 1 - x1), (-dy, y1), (dy, 1 - y1)):
        result = update(p, q, t0, t1)
        if result is None:
            return None
        t0, t1 = result

    clipped_p1 = np.array([x1 + t0 * dx, y1 + t0 * dy], dtype=float)
    clipped_p2 = np.array([x1 + t1 * dx, y1 + t1 * dy], dtype=float)
    clipped_p1 = np.clip(clipped_p1, 0.0, 1.0)
    clipped_p2 = np.clip(clipped_p2, 0.0, 1.0)

    return clipped_p1, clipped_p2

def make_og_vor():
    points = np.random.rand(10, 2)

    points = np.append(points, [[-3,-3]],axis =0)
    points = np.append(points, [[-3, 4]], axis=0)
    points = np.append(points, [[4, -3]], axis=0)
    points = np.append(points, [[4, 4]], axis=0)


    vor = Voronoi(points)

    vertices = (vor.vertices)

    #print(vertices)
    # print(vor.points)
    line_points = vor.ridge_vertices
    # print(line_points)
    faces = []

    for line in line_points:
        if -1 not in line:
            faces.append(vertices[line])

    #print(faces)
    lines = []
    for face in faces:
        for point in range(len(face)):
            lines.append([
                np.array(face[point - 1], dtype=float),
                np.array(face[point], dtype=float),
            ])

    fixed_lines = []
    # print(lines)

    clipped_lines = []

    for line in lines:
        p1, p2 = line

        if np.isclose(p1[0], p2[0]) and np.isclose(p1[1], p2[1]):
            continue

        clipped_segment = clip_segment_to_unit_square(p1, p2)
        if clipped_segment is None:
            continue

        clipped_p1, clipped_p2 = clipped_segment
        if np.allclose(clipped_p1, clipped_p2):
            continue

        clipped_lines.append([clipped_p1, clipped_p2])

    clipped_vertices, clipped_line_indices, clipped_line_segments = extract_vertices_and_lines(clipped_lines)
    output_path = Path(__file__).with_name("clipped_geometry.npz")
    np.savez(
        output_path,
        points=clipped_vertices,
        vertices=clipped_vertices,
        line_indices=clipped_line_indices,
        line_segments=clipped_line_segments,
    )

    fig = voronoi_plot_2d(vor, show_vertices=False)
    plt.scatter(vertices[:, 0], vertices[:, 1], color='green', label='Original Points')
    if len(clipped_vertices) > 0:
        plt.scatter(clipped_vertices[:, 0], clipped_vertices[:, 1], color='red', label='Clipped Points')
    plot_line_segments(clipped_line_segments, color='red', linewidth=1.2)
    plot_unit_square()
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    print(f"Saved clipped geometry to {output_path}")
    print(f"Clipped points: {len(clipped_vertices)} | Clipped lines: {len(clipped_line_indices)}")
    plt.show()

#or x1>1 or x2<0 or x2>1 or y1<0 or y1>1 or y2<0 or y2>1:
if __name__ == "__main__":
    make_og_vor()