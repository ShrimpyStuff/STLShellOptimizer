import meshio

mesh = meshio.read("lattice.msh")
meshio.write("output2.stl", mesh)