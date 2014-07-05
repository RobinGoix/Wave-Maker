"""
Solve the transport equation for a moving object
with different possibilities of schemes
"""

from dolfin import *
#from dolfin_adjoint import * 
import numpy

#Parameters
set_log_level(WARNING) #Output
Nx = 35 #Default 35
Ny = 22 #Default 22
delta_t = 0.03 #[s] Default 0.03 
save = True #Save the (eta,zeta)
ploting= False

name = 'dt' + str(delta_t)\
     + 'Nx' + str(Nx) \
     + 'Ny' + str(Ny) \
     + 'TG3CN'
    
if (save==True):
    hfile = File('results/TransportVideo/MB_' + name +'.pvd') #To save data in a file
    
x0 = -6.
x1 = 60.
y0 = -25.
y1 = 25.
 
#Scaling parameters
g = 9.8
lambda0 = 20. #typical wavelength
a0 = 0.8 #Typical wave height
h0 = 2. #Typical depth
sigma = h0/lambda0
c0 = (h0*g)**(0.5)
epsilon = a0/h0

x0 = x0/lambda0
x1 = x1/lambda0
y0 = y0/lambda0
y1 = y1/lambda0

mesh = RectangleMesh(x0,y0,x1,y1,Nx,Ny)

#Refine the mesh along the object's trajectory
cell_markers0 = CellFunction("bool", mesh)
cell_markers0.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 3./lambda0:
        cell_markers0[cell] = True
    
mesh = refine(mesh, cell_markers0)

cell_markers = CellFunction("bool", mesh)
cell_markers.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 3./lambda0:
        cell_markers[cell] = True
    
mesh = refine(mesh, cell_markers)

cell_markers2 = CellFunction("bool", mesh)
cell_markers2.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 3./lambda0:
        cell_markers2[cell] = True
    
mesh = refine(mesh, cell_markers2)

cell_markers3 = CellFunction("bool", mesh)
cell_markers3.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 3./lambda0:
        cell_markers3[cell] = True
    
mesh = refine(mesh, cell_markers3)

cell_markers4 = CellFunction("bool", mesh)
cell_markers4.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 3./lambda0:
        cell_markers4[cell] = True
    
mesh = refine(mesh, cell_markers4)

cell_markers5 = CellFunction("bool", mesh)
cell_markers5.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 3./lambda0:
        cell_markers5[cell] = True
    
mesh = refine(mesh, cell_markers5)

cell_markers6 = CellFunction("bool", mesh)
cell_markers6.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 3./lambda0:
        cell_markers6[cell] = True
    
mesh = refine(mesh, cell_markers6)

cell_markers7 = CellFunction("bool", mesh)
cell_markers7.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 3./lambda0:
        cell_markers7[cell] = True
    
mesh = refine(mesh, cell_markers7)

cell_markers8 = CellFunction("bool", mesh)
cell_markers8.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 3./lambda0:
        cell_markers8[cell] = True
    
mesh = refine(mesh, cell_markers8)

#plot(mesh, interactive=True)

h = CellSize(mesh)
    
hd = 2. #Depth [m]
hb = 0.3 #Depth at the boundaries [m]
ad = 0.8 #Object's height

hd = hd/h0 #depth
ad = ad/a0 #Object's height
hb = hb/h0  

#Define functions spaces
Q = FunctionSpace(mesh, 'CG', 1)
V = VectorFunctionSpace(mesh, 'CG', 2)

# Create boundary condition 
class Dirichlet_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],x0)   
bc_zeta = DirichletBC(Q, 0.0, Dirichlet_Boundary())

t = 0.0 #time initialization
end = 2.8 #Final Time
delta_t = delta_t*c0/lambda0 #Time step
t = t*c0/lambda0 #Time initialization

#Initial Conditions    
#Trajectory of the object
Vmax = (g*(hd*h0+ad*a0))**0.5 #Physical maximal velocity
Vmax = h0/a0*Vmax/c0 #Scaled maximal velocity
#Time dependant velocity
U_obj = Expression(('Vmax*(1.+4.*lambda0/c0*t/pow((lambda0/c0*t+0.05),2))*exp(-4./(lambda0/c0*t + 0.05))','0.0'),\
                Vmax=Vmax, lambda0=lambda0, c0=c0, t=0.0)

#Initialization of the shape
"""
shape_ub = Function(Q, name="Upper_Bound")
shape_ub = project(Expression(('ad*(x[1]<0. ? 1. : 0.)'\
                +'*(x[1]>-3./lambda0 ? 1. : 0.)*(x[0]>-1.5/lambda0 ? 1. : 0.)'\
                +'*(x[0]<1.5/lambda0 ? 1. : 0.)'), ad=ad, lambda0=lambda0),Q)   
shape_ub.vector()[:] = numpy.rint(shape_ub.vector()[:]) 

movingObject = '(x[1]<3/lambda0 ? 1. : 0.)*(x[1]>=0 ? 1. : 0.)*(lambda0*x[0]>-3 ? 1. : 0.)'\
    +'*ad*0.5*0.5*(1. - tanh(0.5*lambda0*x[1]-2.))*(tanh(10*(1.-lambda0*x[0]-pow(lambda0*x[1],2)/5))'\
    +'+ tanh(2*(lambda0*x[0]+pow(lambda0*x[1],2)/5 + 0.5)))' 
"""
movingObject = '(x[1]<3/lambda0 ? 1. : 0.)*(x[1]>0 ? 1. : 0.)*(lambda0*x[0]>-3 ? 1. : 0.)'\
                    +'*ad*0.5*0.5*(1.-tanh(0.5*lambda0*x[1]-2.))*(tanh(10*(1.-(lambda0*x[0])'\
                    +'-pow(lambda0*x[1],2)/5)) + tanh(2*((lambda0*x[0])+pow(lambda0*x[1],2)/5 + 0.5)))'\
                    +'+(x[1]>-3/lambda0 ? 1. : 0.)*(x[1]<=0 ? 1. : 0.)*(lambda0*x[0]>-3 ? 1. : 0.)'\
                    +'*ad*0.5*0.5*(1.+tanh(0.5*lambda0*x[1]+2.))*(tanh(10*(1.-(lambda0*x[0])'\
                    +'-pow(lambda0*x[1],2)/5)) + tanh(2*((lambda0*x[0])+pow(lambda0*x[1],2)/5 + 0.5)))' 
                
zeta0 = Expression(movingObject, ad=ad, c0=c0, hd=hd, lambda0=lambda0)
zeta_initial = project(zeta0,Q)
zeta_initial = project(zeta_initial)# + 0.9*shape_ub,Q)

U = Function(project(U_obj,V),name="Velocity_(n)")
U_ = Function(project(U_obj,V),name="Velocity_(n-1)")
U__ = Function(project(U_obj,V),name="Velocity_(n-2)")

###############DEFINITION OF THE WEAK FORMULATION############  
zeta_ = Function(zeta_initial, name="zeta_(n)")
zeta = Function(Q, name="zeta_(n+1)")
xi = TestFunction(Q)

#Time stepping methode
alpha = 0.5
zeta_alpha = (1. - alpha)*zeta_ + alpha*zeta
U_alpha = (1. - alpha)*U_ + alpha*U

zeta_t = (zeta - zeta_)/delta_t
U_t = (U - U_)/delta_t
U_tt = (U - 2*U_ + U__)/delta_t**2.

A = zeta_t*xi*dx - epsilon*inner(grad(xi),U_)*zeta_*dx

A += - epsilon*delta_t/2.*inner(grad(xi),U_t)*zeta_alpha*dx \
    + delta_t/2.*epsilon**2*inner(grad(xi),U_alpha)*inner(grad(zeta_alpha),U_alpha)*dx

A += - delta_t**2./6.*(epsilon*inner(grad(xi),U_tt)*zeta_alpha*dx - epsilon**2*inner(U,grad(zeta_alpha))*inner(U_t,grad(xi))*dx \
    - epsilon**2*inner(U,grad(xi))*inner(U_t,grad(zeta_alpha))*dx) + delta_t/6.*epsilon**2*inner(U,grad(zeta-zeta_))*inner(U,grad(xi))*dx
 

###############################ITERATIONS##########################
while (t <= end):
    
    U__.assign(U_)
    U_.assign(U)
    U_obj.t = t
    U.assign(project(U_obj,V))
    #Solve the transport equation 
    solve(A==0, zeta)#, bc_zeta)      
    zeta_.assign(zeta)   
    """
    traj = Vmax*1/c0*t*exp(-4./(lambda0/c0*t + 0.05))
    
    #Move the object 
    x = numpy.array(mesh.coordinates()[:,0])
    y = numpy.array(mesh.coordinates()[:,1])
    x_min = x.min()
    vertex_to_dof = dof_to_vertex_map(Q)
    x = x[vertex_to_dof]
    y = y[vertex_to_dof]
    for i in range(0,len(x)):
        #now find the x to calculate values
        x_t = max(x[i] - traj,x_min) 
        point = numpy.array([x_t, y[i]])
        zeta.vector()[i] = zeta_initial(point)
    zeta_.assign(zeta)
    """
    t += float(delta_t)
    print(t)
    #Plot everything
    if (ploting):                      
        plot(zeta_, mesh, rescale=True, title = "Seabed")
    if (save):
        hfile << zeta_
    ##############################END OF ITERATIONS#################################

