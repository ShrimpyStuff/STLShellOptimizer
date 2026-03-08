import dolfinx
import ufl
import numpy as np
from mpi4py import MPI
import meshio
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem

mesh_file = "lattice.xdmf"  # your generated mesh

with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    # Optional: read cell markers if present
    if xdmf.has_cell_data():
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")
    else:
        cell_tags = None

print("Mesh loaded:")
print("  Number of vertices:", mesh.geometry.dofmap.index_map.size_global)
print("  Number of cells:", mesh.topology.index_map(mesh.topology.dim).size_global)

V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 1))  # continuous Lagrange, degree 1

E = 1e9  # Young's modulus in Pa
nu = 0.3  # Poisson's ratio

# Lamé constants
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

def sigma(u):
    return 2 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(3)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = ufl.Constant(mesh, (0.0, 0.0, 0.0))  # body force
a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
L = ufl.dot(f, v) * ufl.dx

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_sol = problem.solve()