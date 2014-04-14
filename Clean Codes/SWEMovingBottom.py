"""
This code solve Shallow Water Equations on a unit square \
with a moving seabed
"""

from dolfin import *

Ny = 32
Nx = 64
Th = UnitSquareMesh(Nx,Ny)
plot(Th,interacte=True)

#Define Parameters
g = 9.8 #Gravity
dt = Constant(0.0001) #Time step
t = 0.0	#Time initialization
end = 1.0 #Final time
bmarg = 1.e-3 + DOLFIN_EPS
save = True

#Define the profil of the moving seabed
dh = 1 #depth
h0 = 0.3 #height
bh = 0.05 #width
xh = 0.3 #start position
vh = 2 #speed

h_prev = Expression("dh-h0*exp(-(x[0]-xh+vh*dt)*(x[0]-xh+vh*dt)/(bh*bh))",h0=h0,xh=xh,t=t,bh=bh,dh=dh,vh=vh,dt=dt)
h = Expression("dh-h0*exp(-(x[0]-xh)*(x[0]-xh)/(bh*bh))",h0=h0,xh=xh,t=t,bh=bh,dh=dh)

if (save == True):
  fsfile = File("/home/robin/Documents/BCAM/FEniCS_Files/Simulations/SWEMBFS.pvd") #To save data in a file
  hfile = File("/home/robin/Documents/BCAM/FEniCS_Files/Simulations/SWEMBH.pvd") #To save data in a file

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
u_0 = Expression(("0.0", "0.0")) #Initial velocity equal to 0
eta_0 = Expression("0.0") #Initial Height equal to 0

###############DEFINITION OF THE WEAK FORMULATION############

w_prev = Function(E)
(u_prev, eta_prev) = w_prev.split()

u_prev = interpolate(u_0, V)

eta_prev = interpolate(eta_0,H)

h_prev = interpolate(h_prev,H)
h = interpolate(h,H)

w = TrialFunction(E)
u,eta = as_vector((w[0],w[1])),w[2]

wt = TestFunction(E)
v,xi = as_vector((wt[0],wt[1])),wt[2]

F = 1/dt*inner(u-u_prev,v)*dx + inner(grad(u)*u,v)*dx  \
      - g*div(v)*eta*dx + 1/dt*(eta-eta_prev)*xi*dx +1/dt*(h-h_prev)*xi*dx \
      - div(u)*(eta+h)*xi*dx - inner(u,grad(xi))*(eta+h)*dx \
      + (eta+h)*div(u)*xi*dx 
    
w_ = Function(E)
(u_, eta_) = w_.split()
F = action(F, w_)	

###############################ITERATIONS##########################
while (t <= end):
  solve(F==0, w_, bc) #Solve the variational form
  u_prev.assign(u_) 
  h_prev.assign(h)
  eta_prev.assign(eta_) 
  plot(h,rescale=False, title = "Seabed")
  plot(eta_,rescale=True, title = "Free Surface")
  t += float(dt)
  h_new = Expression("dh-h0*exp(-(x[0]-xh-t*vh)*(x[0]-xh-t*vh)/(bh*bh))",h0=h0,xh=xh,t=t,vh=vh,bh=bh,dh=dh)
  h_new = interpolate(h_new,H)
  h.assign(h_new)
  if (save == True):
    fsfile << eta_ #Save heigth
    hfile << h_prev
