"""
This code solves the Boussinesq System derived by Peregrine 
for a seabed of constant depth with a steady object in a strem velocity.

"""
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

#Mesh discretization
Ny = 10
Nx = 300

#Physical values for the physical problem
g = 9.8 #Gravity [m.s^(-2)]

dt = 0.03 #timestep [s]
t = 0.0 #time initialization
end = 500.0 #Final Time

x0 = 40. #Domain [m]
x1 = 55.
y0 = -1
y1 = 1

hd = 1. #Depth [m]
ad = 0.01 #height of the moving object [m]

#Other Parameters
save = False
ploting = True

Th = RectangleMesh(x0,y0,x1,y1,Nx,Ny)

#Define the profil of the moving seabed
U = (hd*g)**(0.5) #Speed

seabed = 'hd'
movingObject = ' - ad*(x[0] > 49. ? 1. : 0.)*(x[0] < 51? 1. : 0.)*(1+cos(pi*(x[0])))'

bottom = seabed + movingObject
h = Expression(bottom, hd=hd, ad=ad)

#Saving parameters
if (save==True):
    fsfile = File("results/SWESolitaryWave5/FS.pvd") #To save data in a file
    hfile = File("results/SWESolitaryWave5/MB.pvd") #To save data in a file

#Define functions spaces
#Velocity
V = VectorFunctionSpace(Th,"Lagrange",2)
#Height
H = FunctionSpace(Th, "Lagrange", 1)
E = MixedFunctionSpace([V,H])

#Dirichlet BC
def Slip_boundary(x, on_boundary):
    return on_boundary and \
            (x[1] > y1 - DOLFIN_EPS or x[1] < y0 + DOLFIN_EPS)
Slip = DirichletBC(E.sub(0).sub(1), Expression("0.0"), Slip_boundary)
"""
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < x0 + DOLFIN_EPS
Gamma_4 = LeftBoundary()
Gamma_4.mark(boundary_parts, 4)
XOut = DirichletBC(E.sub(0).sub(1), Expression("0.",U=U), boundary_parts, 4)
YOut = DirichletBC(E.sub(0).sub(1), Expression("0.0"), boundary_parts, 4)
No_Slip_Out = DirichletBC(E.sub(1), Expression("0.0"), boundary_parts, 4)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] > x1 - DOLFIN_EPS
Gamma_2 = RightBoundary()
Gamma_2.mark(boundary_parts, 2)
XIn = DirichletBC(E.sub(0).sub(0), Expression("0.",U=U), boundary_parts, 2)
YIn = DirichletBC(E.sub(0).sub(1), Expression("0.0"), boundary_parts, 2)
No_Slip_In = DirichletBC(E.sub(1), Expression("0.0"), boundary_parts, 2)
"""
bc = [Slip]#, XOut, YOut, XIn, YIn, No_Slip_In, No_Slip_Out]

n=FacetNormal(Th) #Normal Vector

###############DEFINITION OF THE WEAK FORMULATION############

u_0 = Expression(("0.0", "0.0"), U=U) #Initialisation of the velocity

eta_0 = Expression("0.0")

w_prev = Function(E)
u_prev, eta_prev = as_vector((w_prev[0],w_prev[1])), w_prev[2]

U_obj = project(Expression(('U','0.0'),U=U),V)

u_prev = interpolate(u_0, V)
eta_prev = interpolate(eta_0,H)

h = interpolate(h,H)

w = TrialFunction(E)
u,eta = as_vector((w[0],w[1])),w[2]

wt = TestFunction(E)
v,xi = as_vector((wt[0],wt[1])),wt[2]

F = 1./dt*inner(u-u_prev,v)*dx + inner(grad(u)*(u-U_obj),v)*dx \
    - g*div(v)*eta*dx

F += 1./dt*(eta-eta_prev)*xi*dx + div((h+eta)*(u-U_obj))*xi*dx #- inner(u,grad(xi))*(epsilon*eta+h)*dx + (h+epsilon*eta)*xi*U*ds(4) - (h+epsilon*eta)*xi*U*ds(2)

 
w_ = Function(E)
(u_, eta_) = w_.split()
F = action(F, w_)

###############################ITERATIONS##########################
while (t <= end):
    solve(F==0, w_, bc) #Solve the variational form
    #eta = project(eta_*filtre_eta,H)
    u_prev.assign(u_) #u_prev = u_
    eta_prev.assign(eta_) #eta_prev = eta_
    t += float(dt)
    print(t)
    if (ploting==True):
        plot(eta_prev,rescale=True, title = "Free Surface")
        plot(h,rescale=False, title = "Seabed")
        #plot(u_,rescale = True)
    if (save==True):
        fsfile << eta_ #Save heigth
        hfile << h

##############################END OF ITERATIONS#################################