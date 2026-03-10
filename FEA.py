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
    # # Optional: read cell markers if present
    # cell_tags = xdmf.read_meshtags(mesh, name="Grid")

print("Mesh loaded:")
print("  Number of vertices:", mesh.geometry.x.shape[0])
print("  Number of cells:", mesh.topology.index_map(mesh.topology.dim).size_global)

gdim = mesh.geometry.dim
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))  # continuous Lagrange, degree 1

E = 1e9  # Young's modulus in Pa
nu = 0.3  # Poisson's ratio

# Lamé constants
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda * ufl.div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Constant(mesh, (0.0, 0.0, -9.81))  # body force
a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Define boundary conditions
from dolfinx.fem import dirichletbc, locate_dofs_geometrical
def boundary_fixed(x):
    return np.isclose(x[2], 0) if gdim == 3 else np.isclose(x[0], 0)  # Fix nodes at z=0 (3D) or x=0 (2D)
dofs_fixed = locate_dofs_geometrical(V, boundary_fixed)
bc = dirichletbc(np.zeros(gdim, dtype=dolfinx.default_scalar_type), dofs_fixed, V)

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="solve_")
uh = problem.solve()

compliance_form = dolfinx.fem.form(ufl.inner(sigma(uh), epsilon(uh)) * ufl.dx)
compliance = dolfinx.fem.assemble_scalar(compliance_form)

print("Compliance:", compliance)