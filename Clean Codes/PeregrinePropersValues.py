"""
This code solves the Boussinesq System derived by Peregrine for a seabed of constant depth with a moving object

"""
import scipy.integrate as si
from dolfin import *

Ny = 25
Nx = 95

x0 = -4./5
x1 = 2
y0 = -0.4
y1 = 0.4
Th = RectangleMesh(x0,y0,x1,y1,Nx,Ny)

#Define some Parameters
save = True
moving = True
g = 9.8
c0 = g**(1/2)
lambda0 = 5 #typical wavelength
epsilon = 0.4
dt = 0.02*c0/lambda0#Time step
t = 0.0	#Time initialization
end = 7.*c0/lambda0 #Final time
bmarg = 1.e-3 + DOLFIN_EPS

h0 = 1 #depth
a0 = 1 #height of the moving object
bh = 0.7 #width of the moving object
xh = 0.0 #start position of the moving object

sigma = h0/lambda0


#Define the profil of the moving seabed
if (moving == True):
  vfinal = 2./(0.4*c0)
  velocity = lambda tt: 0.5*vfinal*(tanh(2*(lambda0/c0*tt-1))+tanh(3*(3.0-(lambda0/c0)*tt)))
  amplitude = lambda tt: epsilon*0.5*a0*(tanh(8*(3.0-(lambda0/c0)*tt))+tanh(10+(lambda0/c0)*tt))
  vh = velocity(dt)
  ah=amplitude(dt)
  h_prev = Expression("h0-ah*exp(-(lambda0*x[0]-xh)*(lambda0*x[0]-xh)/(bh*bh))",h0=h0,xh=xh,bh=bh,ah=ah, lambda0=lambda0)
  h = Expression("h0-ah*exp(-(lambda0*x[0]-xh)*(lambda0*x[0]-xh)/(bh*bh))",h0=h0,xh=xh,bh=bh,ah=ah, lambda0=lambda0)
  h_next = Expression("h0-ah*exp(-(lambda0*x[0]-xh-vh*dt)*(lambda0*x[0]-xh-vh*dt)/(bh*bh))", dt=dt, h0=h0,xh=xh,bh=bh,ah=ah,vh=vh, lambda0=lambda0)
else:
  h_prev = Constant(h0)
  h = Constant(h0)
  h_next = Constant(h0)
  
#Saving parameters
if (save==True):
  fsfile = File("/home/robin/Documents/BCAM/FEniCS_Files/Simulations/Peregrine/PeregrineProperValues2/PeregrinePVFS2.pvd") #To save data in a file
  hfile = File("/home/robin/Documents/BCAM/FEniCS_Files/Simulations/Peregrine/PeregrineProperValues2/PeregrinePVBH2.pvd") #To save data in a file

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

zeta_t = (h-h_prev)/(epsilon*dt)
zeta_tt = (h_next-2*h+h_prev)/(epsilon*dt*dt)

F = 1/dt*inner(u-u_prev,v)*dx + epsilon*inner(grad(u)*u,v)*dx - div(v)*eta*dx

F += sigma*sigma/2*1/dt*div(v)*div(h*(u-u_prev))*h*dx + sigma*sigma/2*1/dt*inner(v,grad(h))*div(h*(u-u_prev))*dx \
     - sigma*sigma/6*1/dt*div(v)*div(u-u_prev)*h*h*dx - sigma*sigma/6*1/dt*inner(v,grad(h))*2*h*div(u-u_prev)*dx \
     +sigma*sigma/2*div(v)*h*zeta_tt*dx+sigma*sigma/2*inner(v,grad(h))*zeta_tt*dx

F += 1/dt*(eta-eta_prev)*xi*dx + zeta_t*xi*dx - inner(u,grad(xi))*(epsilon*eta+h)*dx 
     
    
w_ = Function(E)
(u_, eta_) = w_.split()
F = action(F, w_)	

###############################ITERATIONS##########################
while (t <= end):
  solve(F==0, w_, bc) #Solve the variational form
  u_prev.assign(u_) #u_prev = u_
  eta_prev.assign(eta_) #eta_prev = eta_
  t += float(dt)
  print(t)
  plot(eta_,rescale=True, title = "Free Surface")
  
  if(moving==True): #Move the object --> assigne new values to h_prev, h_, h_next
    h_prev.assign(h)
    h.assign(h_next)
    intvh=si.quad(velocity, 0, t)
    intvh=intvh[0]
    ah=amplitude(t)
    h_new = Expression("h0-ah*exp(-(lambda0*x[0]-xh-intvh)*(lambda0*x[0]-xh-intvh)/(bh*bh))",intvh=intvh, h0=h0,xh=xh,t=t,vh=vh,bh=bh,ah=ah,dt=dt,lambda0=lambda0)
    h_new = interpolate(h_new,H)
    h_next.assign(h_new)
    plot(h,rescale=False, title = "Seabed")
    
  if (save==True):
    fsfile << eta_ #Save heigth
    hfile << h_prev

##############################END OF ITERATIONS#################################
      
