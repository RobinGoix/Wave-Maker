"""
This code solves the Boussinesq System derived by Peregrine for a seabed of constant depth with a moving object

"""
from dolfin import *

Ny = 32
Nx = 64
Th = UnitSquareMesh(Nx,Ny)

#Define some Parameters
save = True
dt = Constant(0.01) #Time step
t = 0.0	#Time initialization
end = 10.0 #Final time
bmarg = 1.e-3 + DOLFIN_EPS

h0 = 1 #depth
a0 = 0.2 #height of the moving object
bh = 0.1 #width of the moving object
xh = 0.3 #start position of the moving object
vh = 0.005 #speed of the moving object
lambda0 = 0.3 #typical wavelength
epsilon = a0/h0
sigma = h0/lambda0

#Define the profil of the moving seabed
h_prev = Expression("h0-a0*exp(-(x[0]-xh+vh*dt)*(x[0]-xh+vh*dt)/(bh*bh))*(tanh(10*(x[1]-0.2))+tanh(10*(0.8-x[1])))",h0=h0,xh=xh,t=t,bh=bh,a0=a0,vh=vh,dt=dt)
h = Expression("h0-a0*exp(-(x[0]-xh)*(x[0]-xh)/(bh*bh))*(tanh(10*(x[1]-0.2))+tanh(10*(0.8-x[1])))",h0=h0,xh=xh,t=t,bh=bh,a0=a0)
h_next = Expression("h0-a0*exp(-(x[0]-xh-vh*dt)*(x[0]-xh-vh*dt)/(bh*bh))*(tanh(10*(x[1]-0.2))+tanh(10*(0.8-x[1])))",h0=h0,xh=xh,t=t,bh=bh,a0=a0,vh=vh,dt=dt)

#Saving parameters
if (save==True):
  fsfile = File("home/robin/Documents/BCAM/FEniCS_Files/Simulations/PeregrineFS.pvd") #To save data in a file
  hfile = File("home/robin/Documents/BCAM/FEniCS_Files/Simulations/PeregrineBHbis.pvd") #To save data in a file

#Define functions spaces
#Velocity
V = VectorFunctionSpace(Th,"Lagrange",2)
#Height
H = FunctionSpace(Th, "Lagrange", 1) 
E = V * H

#Dirichlet BC

def NoSlip_boundary(x, on_boundary):
        return on_boundary and \
               (x[1] < bmarg or x[1] > 1- bmarg or \
                x[0] < bmarg or x[0] > 1- bmarg)
No_Slip = DirichletBC(E.sub(0), [0.0, 0.0], NoSlip_boundary)

bc = No_Slip

n=FacetNormal(Th) #Normal Vector

#Initial Conditions
u_0 = Expression(("0.0", "0.0")) #Initialisation of the velocity

eta_0 = Expression("0.0") #Initialisation of the free surface

###############DEFINITION OF THE WEAK FORMULATION############

w_prev = Function(E)
(u_prev, eta_prev) = w_prev.split()

u_prev = interpolate(u_0, V)

eta_prev = interpolate(eta_0,H)

h_prev = interpolate(h_prev,H)
h = interpolate(h,H)
h_next = interpolate(h_next,H)

w = TrialFunction(E)
u,eta = as_vector((w[0],w[1])),w[2]

wt = TestFunction(E)
v,xi = as_vector((wt[0],wt[1])),wt[2]

F = 1/dt*inner(u-u_prev,v)*dx + epsilon*inner(grad(u)*u,v)*dx - div(v)*eta*dx \
      + sigma*sigma/2*1/dt*div(v)*div(h*(u-u_prev))*h*dx + sigma*sigma/2*1/dt*inner(v,grad(h))*div(h*(u-u_prev))*dx \
      - sigma*sigma/6*1/dt*div(v)*div(u-u_prev)*h*h*dx - sigma*sigma/6*1/dt*inner(v,grad(h))*2*h*div(u-u_prev)*dx \
      +sigma*sigma/2*1/(dt*dt)*div(v)*h*(h_prev-2*h+h_next)*dx+sigma*sigma/2*1/(dt*dt)*inner(v,grad(h))*(h_prev-2*h+h_next)*dx \
      + 1/dt*(eta-eta_prev)*xi*dx +1/dt*(h-h_prev)*xi*dx - inner(u,grad(xi))*(epsilon*eta+h)*dx 
      
    
w_ = Function(E)
(u_, eta_) = w_.split()
F = action(F, w_)	

###############################ITERATIONS##########################
while (t <= end):
  #plot(h-h_prev, title = "Seabed")
  solve(F==0, w_, bc) #Solve the variational form
  u_prev.assign(u_) #u_prev = u_
  eta_prev.assign(eta_) #p_prev = p_
  h_prev.assign(h)
  h.assign(h_next)
  plot(h,rescale=False, title = "Seabed")
  plot(eta_,rescale=True, title = "Free Surface")
  t += float(dt)
  h_new = Expression("h0-a0*exp(-(x[0]-xh-(t+dt)*vh)*(x[0]-xh-(t+dt)*vh)/(bh*bh))*(tanh(10*(x[1]-0.2))+tanh(10*(0.8-x[1])))",h0=h0,xh=xh,t=t,vh=vh,bh=bh,a0=a0,dt=dt)
  h_new = interpolate(h_new,H)
  h_next.assign(h_new)
  if (save==True):
    fsfile << eta_ #Save heigth
    hfile << h_prev
      
