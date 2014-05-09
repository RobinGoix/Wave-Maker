"""
This code solves the Boussinesq System derived by Peregrine for a seabed of constant depth with a moving object

"""
import scipy.integrate as si
from dolfin import *

Ny = 32
Nx = 128

g = 9.8
lambda0 = 1 #typical wavelength
a0 = 1 #Typical wave height
h0 = 1 #Typical depth
sigma = h0/lambda0
c0 = (h0*g)**(1/2)
epsilon = a0/h0

x0 = -4./lambda0
x1 = 10/lambda0
y0 = -2/lambda0
y1 = 2/lambda0

Th = RectangleMesh(x0,y0,x1,y1,Nx,Ny)

#Define some Parameters
save = True
moving = True

dt = 0.02*c0/lambda0#Time step
t = 0.0	#Time initialization
end = 7.*c0/lambda0 #Final time
bmarg = 1.e-3 + DOLFIN_EPS

hd = 1/h0 #depth
ad = 0.2/a0 #height of the moving object
bh = 0.7 #width of the moving object
xh = 0.0/lambda0 #start position of the moving object

#Define the profil of the moving seabed
if (moving == True):
  vfinal = 1.*h0/(a0*c0)
  velocity = lambda tt: 0.5*vfinal*(tanh(2*(lambda0/c0*tt-1))+tanh(3*(3.0-(lambda0/c0)*tt)))
  amplitude = lambda tt: epsilon*0.5*ad*(tanh(8*(3.0-(lambda0/c0)*tt))+tanh(10+(lambda0/c0)*tt))
  vh = velocity(dt)
  ah=amplitude(dt)
  h_prev = Expression("hd-ah*exp(-(lambda0*x[0]-xh)*(lambda0*x[0]-xh)/(bh*bh))",hd=hd,xh=xh,bh=bh,ah=ah, lambda0=lambda0)
  h = Expression("hd-ah*exp(-(lambda0*x[0]-xh)*(lambda0*x[0]-xh)/(bh*bh))",hd=hd,xh=xh,bh=bh,ah=ah, lambda0=lambda0)
  h_next = Expression("hd-ah*exp(-(lambda0*x[0]-xh-vh*dt)*(lambda0*x[0]-xh-vh*dt)/(bh*bh))", dt=dt, hd=hd,xh=xh,bh=bh,ah=ah,vh=vh, lambda0=lambda0)
else:
  h_prev = Constant(hd)
  h = Constant(hd)
  h_next = Constant(hd)
  
#Saving parameters
if (save==True):
  fsfile = File("/home/robin/Documents/BCAM/FEniCS_Files/Simulations/Peregrine/PeregrineProperValues3/PeregrinePVFS3.pvd") #To save data in a file
  hfile = File("/home/robin/Documents/BCAM/FEniCS_Files/Simulations/Peregrine/PeregrineProperValues3/PeregrinePVBH3.pvd") #To save data in a file

#Define functions spaces
#Velocity
V = VectorFunctionSpace(Th,"Lagrange",2)
#Height
H = FunctionSpace(Th, "Lagrange", 1) 
E = V * H

#Dirichlet BC

def NoSlip_boundary(x, on_boundary):
        return on_boundary and \
               (x[1] < bmarg or x[1] > y1- bmarg or \
                x[0] < bmarg or x[0] > x1- bmarg)
No_Slip = DirichletBC(V, [0.0, 0.0], NoSlip_boundary)

bc = No_Slip

n=FacetNormal(Th) #Normal Vector

#Initial Conditions
u_0 = Expression(("0.0", "0.0")) #Initialisation of the velocity


###############DEFINITION OF THE WEAK FORMULATION############

u_prev = Function(V)
u_prev = interpolate(u_0, V)

h_prev = interpolate(h_prev,H)
h = interpolate(h,H)
h_next = interpolate(h_next,H)

u = TrialFunction(V)

v = TestFunction(V)

zeta_t = (h-h_prev)/(epsilon*dt)
zeta_tt = (h_next-2*h+h_prev)/(epsilon*dt*dt)

F = sigma**2*1/dt*div(h*(u-u_prev))*div(h*v/2)*dx \
    - sigma**2*1/dt*div(u-u_prev)*div(h*h*v/6)*dx \
    + sigma**2*zeta_tt*div(h*v/2)*dx

F -= sigma**2*1/2*1/dt*div(v)*div(h*(u-u_prev))*h*dx + sigma**2*1/2*1/dt*inner(v,grad(h))*div(h*(u-u_prev))*dx \
     - sigma**2*1/6*1/dt*div(v)*div(u-u_prev)*h*h*dx - sigma**2*1/6*1/dt*inner(v,grad(h))*2*h*div(u-u_prev)*dx \
     +sigma**2*1/2*div(v)*h*zeta_tt*dx+sigma**2*1/2*inner(v,grad(h))*zeta_tt*dx
   
F += inner(u,v)*dx 

u_ = Function(V)
F = action(F, u_)	

###############################ITERATIONS##########################
while (t <= end):
  solve(F==0, u_, bc) #Solve the variational form
  u_prev.assign(u_) #u_prev = u_
  t += float(dt)
  print(t)
  plot(u_,rescal=True)
  
  if(moving==True): #Move the object --> assigne new values to h_prev, h_, h_next
    h_prev.assign(h)
    h.assign(h_next)
    intvh=si.quad(velocity, 0, t)
    intvh=intvh[0]
    ah=amplitude(t)
    h_new = Expression("hd-ah*exp(-(lambda0*x[0]-xh-intvh)*(lambda0*x[0]-xh-intvh)/(bh*bh))",intvh=intvh, hd=hd,xh=xh,t=t,vh=vh,bh=bh,ah=ah,dt=dt,lambda0=lambda0)
    h_new = interpolate(h_new,H)
    h_next.assign(h_new)
    plot(h,rescale=False, title = "Seabed")  
##############################END OF ITERATIONS#################################
      
