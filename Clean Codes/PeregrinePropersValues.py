"""
This code solves the Boussinesq System derived by Peregrine for a seabed of constant depth with a moving object

"""
import scipy.integrate as si
from dolfin import *

#Mesh discretization
Ny = 25
Nx = 95

#Physical values for the physical problem
g = 9.8 #Gravity [m.s^(-2)]

dt = 0.02 #timestep [s]
t = 0.0 #time initialization
end = 7.0 #End

x0 = -4. #Domain [m]
x1 = 10.
y0 = -2.
y1 = 2.

hd = 1. #Depth [m]
ad = 0.4 #height of the moving object [m]
bh = 0.7 #width of the moving object 
xh = 0.0 #start position of the moving object [m]
vfinal = 1. #Maximal velocity of the moving object [m.s^(-1)]

u_0 = Expression(("0.0", "0.0")) #Initialisation of the velocity
eta_0 = Expression("0.0") #Initialisation of the free surface

#Scaling parameters
lambda0 = 1 #typical wavelength
a0 = 1 #Typical wave height
h0 = 1 #Typical depth
sigma = h0/lambda0
c0 = (h0*g)**(1/2)
epsilon = a0/h0

#Other Parameters
save = True
moving = True
ploting = False
bmarg = 1.e-3 + DOLFIN_EPS

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
xh = xh/lambda0 #start position of the moving object

#Define the profil of the moving seabed
if (moving == True):
  velocity = lambda tt: 0.5*vfinal*(tanh(2*(lambda0/c0*tt-1))+tanh(3*(3.0-(lambda0/c0)*tt)))
  amplitude = lambda tt: epsilon*0.5*ad*(tanh(3*(3.0-(lambda0/c0)*tt))+tanh(10+(lambda0/c0)*tt))
  vh = velocity(dt)
  ah=amplitude(dt)
  h_prev = Expression("hd-ah*exp(-pow((lambda0*(x[0]-xh))/bh,2))",\
	    hd=hd, ah=ah, xh=xh, bh=bh, lambda0=lambda0)
  h = Expression("hd-ah*exp(-pow((lambda0*(x[0]-xh))/bh,2))", \
	    hd=hd, ah=ah, xh=xh, bh=bh, lambda0=lambda0)
  h_next = Expression("hd-ah*exp(-pow((lambda0*(x[0]-xh)-vh*lambda0/c0*dt)/bh,2))", \
	    hd=hd, ah=ah, xh=xh, bh=bh, vh=vh,lambda0=lambda0, c0=c0, dt=dt)
else:
  h_prev = Constant(hd)
  h = Constant(hd)
  h_next = Constant(hd)
  
#Saving parameters
if (save==True):
  fsfile = File("results/PeregrinePV4/PeregrinePVFS.pvd") #To save data in a file
  hfile = File("results/PeregrinePV4/PeregrinePVMB.pvd") #To save data in a file

#Define functions spaces
#Velocity
V = VectorFunctionSpace(Th,"Lagrange",2)
#Height
H = FunctionSpace(Th, "Lagrange", 1) 
E = V * H

#Dirichlet BC

def NoSlip_boundary(x, on_boundary):
        return on_boundary and \
               (x[1] < y0 + bmarg or x[1] > y1- bmarg or \
                x[0] < x0 + bmarg or x[0] > x1- bmarg)
No_Slip = DirichletBC(E.sub(0), [0.0, 0.0], NoSlip_boundary)

bc = No_Slip

n=FacetNormal(Th) #Normal Vector

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

F = 1/dt*inner(u-u_prev,v)*dx + epsilon*inner(grad(u)*u,v)*dx \
    - div(v)*eta*dx

F += sigma**2*1/dt*div(h*(u-u_prev))*div(h*v/2)*dx \
      - sigma**2*1/dt*div(u-u_prev)*div(h*h*v/6)*dx \
      + sigma**2*zeta_tt*div(h*v/2)*dx

F += 1/dt*(eta-eta_prev)*xi*dx + zeta_t*xi*dx \
      - inner(u,grad(xi))*(epsilon*eta+h)*dx 
     
    
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

  if(moving==True): #Move the object --> assigne new values to h_prev, h_, h_next
    h_prev.assign(h)
    h.assign(h_next)
    intvh=si.quad(velocity, 0, t)
    intvh=intvh[0] 
    ah=amplitude(t)
    h_new = Expression("hd-ah*exp(-pow((lambda0*(x[0]-xh)-lambda0/c0*intvh)/bh,2))", \
		hd=hd, ah=ah, xh=xh, bh=bh, intvh=intvh, lambda0=lambda0, c0=c0)
    h_new = interpolate(h_new,H)
    h_next.assign(h_new)
    
  if (ploting==True):
    plot(eta_,rescale=True, title = "Free Surface")
    plot(h,rescale=False, title = "Seabed")
    
  if (save==True):
    fsfile << eta_ #Save heigth
    hfile << h_prev

##############################END OF ITERATIONS#################################
