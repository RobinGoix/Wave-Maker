"""
Optimal shape design of a wave maker
"""

from dolfin import *
from dolfin_adjoint import * 
#import pyipopt



Nx = 75
Ny = 55

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

delta_t = 0.03 #timestep [s]
t = 0.0 #time initialization
end = 2.8 #Final Time
delta_t = delta_t*c0/lambda0 #Time step
t = t*c0/lambda0 #Time initialization
end = end*c0/lambda0 #Final time

x0 = x0/lambda0
x1 = x1/lambda0
y0 = y0/lambda0
y1 = y1/lambda0
mesh = RectangleMesh(x0,y0,x1,y1,Nx,Ny)

#Refine the mesh along the object's trajectory
cell_markers = CellFunction("bool", mesh)
cell_markers.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -4./lambda0:
        cell_markers[cell] = True
    
mesh = refine(mesh, cell_markers)

cell_markers2 = CellFunction("bool", mesh)
cell_markers2.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3.5/lambda0 and p.y() < 20./lambda0:
        cell_markers2[cell] = True
    
mesh = refine(mesh, cell_markers2)

cell_markers3 = CellFunction("bool", mesh)
cell_markers3.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 18./lambda0:
        cell_markers3[cell] = True
    
mesh = refine(mesh, cell_markers3)

cell_markers4 = CellFunction("bool", mesh)
cell_markers4.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 15./lambda0:
        cell_markers4[cell] = True
    
mesh = refine(mesh, cell_markers4)

h = CellSize(mesh)

#Other Parameters
save = False
ploting = True

hd = 2. #Depth [m]
hb = 0.3 #Depth at the boundaries [m]
ad = 0.8 #Object's height

hd = hd/h0 #depth
ad = ad/a0 #Object's height
hb = hb/h0

#Shape of the seabed
seabed = 'hd - (hd-hb)/21.*(x[1]>4./lambda0 ? 1. : 0.)*(lambda0*x[1]-4.)' \
        + '+ (hd-hb)/21.*(x[1]<(-4./lambda0) ? 1. : 0.)*(lambda0*x[1]+4.)'
seabed = Expression(seabed, hd=hd, hb=hb,  lambda0=lambda0) 

#Trajectory of the object
Vmax = (g*(hd*h0+ad*a0))**0.5 #Physical maximal velocity
Vmax = h0/a0*Vmax/c0 #Scaled maximal velocity
#Time dependant velocity
U = Expression(('Vmax*(1.+4.*lambda0/c0*t/pow((lambda0/c0*t+0.05),2))*exp(-4./(lambda0/c0*t + 0.05))','0.0'),\
                Vmax=Vmax, lambda0=lambda0, c0=c0, t=0.0)
#Corresponding trajectory
traj = 'Vmax*lambda0/c0*t*exp(-4./(lambda0/c0*t+0.05))'

#Initial Conditions
u0 = Expression(("0.0", "0.0"))
eta0 = Expression("0.0")
movingObject = ' - (x[1]<3/lambda0 ? 1. : 0.)*(x[1]>0 ? 1. : 0.)*(lambda0*x[0]>-6 ? 1. : 0.)'\
    +'*ad*0.5*0.5*(1. - tanh(0.5*lambda0*x[1]-2.))*(tanh(10*(1.-lambda0*x[0]-pow(lambda0*x[1],2)/5))'\
    +'+ tanh(2*(lambda0*x[0]+pow(lambda0*x[1],2)/5 + 0.5))) ' \
    + ' - (x[1]>-3/lambda0 ? 1. : 0.)*(x[1]<=0 ? 1. : 0.)*(lambda0*x[0]>-6 ? 1. : 0.)'\
    +'*ad*0.5*0.5*(1. + tanh(0.5*lambda0*x[1]+2.))*(tanh(10*(1. - lambda0*x[0]-pow(lambda0*x[1],2)/5))'\
    +'+ tanh(2*(lambda0*x[0]+pow(lambda0*x[1],2)/5 + 0.5))) ' 
zeta0 = Expression(movingObject, ad=ad, c0=c0, hd=hd, lambda0=lambda0)

filtre = '(lambda0*x[0] < 2 + '+traj+' ? 1. : 0.)*(lambda0*x[0] > -6 + '+traj+'? 1. : 0.)'\
        +'*(lambda0*x[1] < 3 ? 1. : 0.)*(lambda0*x[1] > -3 ? 1. : 0.)'
filtre = Expression(filtre, Vmax=Vmax, t=0.0, lambda0=lambda0, c0=c0)

#Saving parameters
if (save==True):
    fsfile = File("results/Objectshape1/FS.pvd") #To save data in a file
    hfile = File("results/Objectshape1/MB.pvd") #To save data in a file
    
#Define functions spaces
#Velocity
V = VectorFunctionSpace(mesh,'CG',1)
#Height
H = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(mesh, 'CG',2)
E = MixedFunctionSpace([V, H])

zeta0 = interpolate(zeta0, Q)
D = interpolate(seabed,H)

#Dirichlet BC
# No-slip boundary
class Y_SlipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and \
            (x[1] < y0 + DOLFIN_EPS or x[1] > y1 - DOLFIN_EPS)

class VelocityStream_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and \
            (x[0] < x0 + DOLFIN_EPS or x[0] > x1 - DOLFIN_EPS)

class Entry_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] > x1 - DOLFIN_EPS)
    
class Dirichlet_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Create no-slip boundary condition for velocity
bc_X_u = DirichletBC(E.sub(0), Expression(("0.0","0.0")), VelocityStream_Boundary())
bc_X_eta = DirichletBC(E.sub(1), 0.0, Entry_Boundary())
bc_Y_u = DirichletBC(E.sub(0).sub(1), 0.0, Y_SlipBoundary())

bc_zeta = DirichletBC(Q, 0.0, Dirichlet_Boundary())

bcs = [bc_X_u, bc_Y_u, bc_X_eta] 
###############DEFINITION OF THE WEAK FORMULATION############  
w__ = Function(E)
u__, eta__= split(w__)
zeta__ = Function(Q)

u__ = interpolate(u0, V)
eta__ = interpolate(eta0, H)
zeta__ = interpolate(zeta0, Q)
U = interpolate(U, V)

w_ = Function(E)
u_, eta_ = split(w_)
zeta_ = Function(Q)

u_ = interpolate(u0, V)
eta_ = interpolate(eta0, H)
zeta_ = interpolate(zeta0, Q)

w = Function(E)
u, eta = split(w)
zeta = Function(Q)

v, chi = TestFunctions(E)
xi = TestFunction(Q)

#Time stepping methode
alpha = 0.5
u_alpha = (1.-alpha)*u_+ alpha*u
eta_alpha = (1. - alpha)*eta_ + alpha*eta
zeta_alpha = (1. - alpha)*zeta_ + alpha*zeta

zeta_t = (zeta-zeta_)/delta_t
zeta_tt = (zeta-2.*zeta_+zeta__)/delta_t**2

F = 1./delta_t*inner(u-u_,v)*dx + epsilon*inner(grad(u_alpha)*u_alpha,v)*dx \
    - div(v)*eta_alpha*dx

F += sigma**2.*1./delta_t*div((D + epsilon*zeta_alpha)*(u-u_))*div((D + epsilon*zeta_alpha)*v/2.)*dx \
    - sigma**2.*1./delta_t*div(u-u_)*div((D + epsilon*zeta_alpha)**2*v/6.)*dx \
    + sigma**2.*zeta_tt*div((D + epsilon*zeta_alpha)*v/2.)*dx

F += 1./delta_t*(eta-eta_)*chi*dx + zeta_t*chi*dx \
    - (inner(u_alpha,grad(chi))*(epsilon*eta_alpha + D + epsilon*zeta_alpha))*dx 

F += 0.1*h**(3./2.)*(inner(grad(u_alpha),grad(v)) + inner(grad(eta_alpha),grad(chi)))*dx

A = 1./delta_t*(zeta-zeta_)*xi*dx + inner(grad(zeta_alpha),U)*xi*dx - 3*zeta*xi*dx #+ 1000000.0*(1-filtre)*zeta*xi*dx
A += (1. - filtre)*h**(3./2.)*inner(grad(zeta_alpha),grad(xi))*dx

###############################ITERATIONS##########################
while (t <= end):  
    solve(A==0, zeta, bc_zeta)
    filtre.t = t
    zeta__.assign(zeta_)
    zeta_.assign(zeta)   
    solve(F==0, w, bcs) #Solve the variational form
    w__.assign(w_)
    w_.assign(w)
    #eta_.assign(eta)
    t += float(delta_t) 
    if (ploting==True):
        plot(eta_,rescale=True, title = "Free Surface")
        plot(zeta_, mesh, rescale=True, title = "Seabed")
        plot(filtre,mesh)
    if (save==True):
        fsfile << eta_ #Save heigth
##############################END OF ITERATIONS#################################

