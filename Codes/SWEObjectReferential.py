"""
This code solves the Boussinesq System derived by Peregrine for a seabed of constant depth with a moving object.

"""
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

#Mesh discretization
Ny = 10
Nx = 150

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
ad = 0.5 #height of the moving object [m]

#Scaling parameters
lambda0 = 20. #typical wavelength
a0 = 1. #Typical wave height
h0 = 1. #Typical depth
sigma = h0/lambda0
c0 = (h0*g)**(0.5)
epsilon = a0/h0

#Other Parameters
save = False
ploting = True

#Scaled parameters to solve the dimensionless problem
x0 = x0/lambda0
x1 = x1/lambda0
y0 = y0/lambda0
y1 = y1/lambda0
Th = RectangleMesh(x0,y0,x1,y1,Nx,Ny)

dt = dt*c0/lambda0 #Time step
t = t*c0/lambda0 #Time initialization
end = end*c0/lambda0 #Final time

hd = hd/h0 #depth
ad = ad/a0 #height of the moving object

#Define the profil of the moving seabed
U = (hd*g)**(0.5) #Speed
U = h0/(a0*c0)*U

seabed = 'hd'
movingObject = ' - 0.2/2*(x[0] > 49./lambda0 ? 1. : 0.)*(x[0] < 51/lambda0 ? 1. : 0.)*(1+cos(pi*(lambda0*x[0])))'   

bottom = seabed + movingObject 
h = Expression(bottom, hd=hd, ad=ad,lambda0=lambda0)

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
boundary_parts = \
  MeshFunction("uint", Th, Th.topology().dim()-1)


def Slip_boundary(x, on_boundary):
    return on_boundary and \
            (x[1] > y1 - DOLFIN_EPS or x[1] < y0 + DOLFIN_EPS)
Slip = DirichletBC(V.sub(1),  Expression("0.0"), Slip_boundary)

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and  x[0] < x0 + DOLFIN_EPS
Gamma_4 = LeftBoundary()
Gamma_4.mark(boundary_parts, 4)
XOut = DirichletBC(V.sub(1),  Expression("-U",U=U), boundary_parts, 4)
YOut = DirichletBC(V.sub(1),  Expression("0.0"), boundary_parts, 4)
No_Slip_Out = DirichletBC(E.sub(1),  Expression("0.0"), boundary_parts, 4)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] > x1 - DOLFIN_EPS
Gamma_2 = RightBoundary()
Gamma_2.mark(boundary_parts, 2)
XIn = DirichletBC(V.sub(0),  Expression("-U",U=U), boundary_parts, 2)
YIn = DirichletBC(V.sub(1),  Expression("0.0"), boundary_parts, 2)
No_Slip_In = DirichletBC(E.sub(1),  Expression("0.0"), boundary_parts, 2)

bc = [Slip, XOut, YOut, XIn, YIn, No_Slip_In, No_Slip_Out]




n=FacetNormal(Th) #Normal Vector

###############DEFINITION OF THE WEAK FORMULATION############

u_0 = Expression(("0.0", "0.0"), U=U) #Initialisation of the velocity

eta_0 = Expression("0.0")

filtre_eta = Expression("0.5*(x[0]<=(x0+20)? 1. : 0.)*(1+cos(pi*(x[0]-(x0+20))/20)) + (x[0]>(x0+20)? 1. : 0.)",x0=x0)
filtre_eta = interpolate(filtre_eta,H)

w_prev = Function(E)
u_prev, eta_prev = as_vector((w_prev[0],w_prev[1])), w_prev[2]

u_prev = interpolate(u_0, V)
eta_prev = interpolate(eta_0,H)

h = interpolate(h,H)

w = TrialFunction(E)
u,eta = as_vector((w[0],w[1])),w[2]

wt = TestFunction(E)
v,xi = as_vector((wt[0],wt[1])),wt[2]

F = 1./dt*inner(u-u_prev,v)*dx + epsilon*inner(grad(u)*u,v)*dx \
    - div(v)*eta*dx

F += 1./dt*(eta-eta_prev)*xi*dx + div((h+epsilon*eta)*u)*xi*dx #- inner(u,grad(xi))*(epsilon*eta+h)*dx + (h+epsilon*eta)*xi*U*ds(4) - (h+epsilon*eta)*xi*U*ds(2)
     

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
