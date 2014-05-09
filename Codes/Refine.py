'''
Tests to be able to refine locally the mesh in FEniCS. Aims at refining the mesh only around the moving object.

'''

from dolfin import *

x0 = -4.
x1 = 30.
y0 = -15.
y1 = 15.

x0 = x0/20
x1 = x1/20
y0 = y0/20
y1 = y1/20

mesh = RectangleMesh(x0, y0, x1, y1, 125, 45)


origin = Point(0.0, 0.0)

t = 0.0
dt = 4.
end = 20.0

while (t<end):
    #mesh = RectangleMesh(x0, y0, x1, y1, 50, 20)
    t += dt
    origin = Point(t, 0)
    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    for cell in cells(mesh):
        p = cell.midpoint()
        if p.x() > (t-2.)/20. and p.x() < (t+2.)/20. and p.y() > -2./20. and p.y() < 2./20. :
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)
    print(t)
    plot(mesh)

"""
x0 = -4.
x1 = 30.
y0 = -15.
y1 = 15.

x0 = x0/20
x1 = x1/20
y0 = y0/20
y1 = y1/20

mesh0 = RectangleMesh(x0, y0, x1, y1, 250, 90, 'crossed')

mesh = RectangleMesh(x0, y0, x1, y1, 125, 45, 'crossed')

cell_markers = CellFunction("bool", mesh)
cell_markers.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -2.5/20 and p.y() < 2.5/20 :
        cell_markers[cell] = True
    
mesh1 = refine(mesh, cell_markers)

plot(mesh0,axes=True)
plot(mesh1,axes=True)
"""
interactive()
