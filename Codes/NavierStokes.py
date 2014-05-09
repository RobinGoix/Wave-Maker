"""
This code solve Navier-Stokes time dependant equations 
for a Newtonian Fluid around around a cylinder
"""
from dolfin import *

#Define Parameters
Re = Constant(1000) #Reynolds Number
mu = Constant(0.05/Re) #Viscosity
rho = Constant(1.0) #Density
gy = 0.0 #Gravity
g = Constant([0.0, gy]) # Gravity vector
dt = Constant(0.01) #Time step
t = 0.0	#Time initialization
end = 1000 #Final time
save = True

if (save==True):
  ufile = File("/home/robin/Documents/BCAM/FEniCS_Files/Simulations/NSE.pvd") #To save data in a file

#Load Mesh
Th = Mesh("cylinder_2d.xml.gz")

#Parameters corresponding to this mesh : DO NOT CHANGE THEM
bmarg = 1.e-3 + DOLFIN_EPS
xmin = 0.0
xmax = 2.2
ymin = 0.0
ymax = 0.41
xcenter = 0.2
ycenter = 0.2
radius = 0.05

#Define functions spaces
#Velocity
V = VectorFunctionSpace(Th,"Lagrange",2)
#Pressure
Q = FunctionSpace(Th, "Lagrange", 1) 
E = V * Q

#Define boundary conditions

#Define parts of the boundary
boundary_parts =  MeshFunction("uint", Th, 1) #Create the meshfunction

class LeftBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and x[0]<DOLFIN_EPS
Gamma_N_in = LeftBoundary()
Gamma_N_in.mark(boundary_parts,0)

class RightBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and x[0]>xmax-DOLFIN_EPS
Gamma_N_out = RightBoundary()
Gamma_N_out.mark(boundary_parts,2)

class NoSlipBoundary(SubDomain):
  def inside(self, x, on_boundary):
	dx = x[0] - xcenter
        dy = x[1] - ycenter
        r = sqrt(dx*dx + dy*dy)
        return on_boundary and \
               (x[1] < ymin + bmarg or x[1] > ymax - bmarg or \
                r < radius + bmarg)
Gamma_NoSlip = NoSlipBoundary()
Gamma_NoSlip.mark(boundary_parts,1)

#Define new mesure ds
ds = Measure("ds")[boundary_parts]

# No slip boundary condition (on the cylinder, on y=ymin and on y=max
def NoSlip_boundary(x, on_boundary):
        dx = x[0] - xcenter
        dy = x[1] - ycenter
        r = sqrt(dx*dx + dy*dy)
        return on_boundary and \
               (x[1] < ymin + bmarg or x[1] > ymax - bmarg or \
                r < radius + bmarg)
No_Slip = DirichletBC(E.sub(0), [0.0, 0.0], NoSlip_boundary)

#Pressure BC
p_in = 0
p_out = 0
p_imposed = Expression("p_in+rho*g*x[1]+(p_out-p_in)*x[0]/2.2",\
			p_in = p_in, p_out = p_out, rho = rho, g = gy)
p_imposed = interpolate(p_imposed, Q) 

#Velocity BC
u_in = 1.0 #Inflow velocity
def Uin_boundary(x, on_boundary):
        return on_boundary and (x[0] < bmarg )
In_Flow = DirichletBC(E.sub(0), [u_in, 0.0], Uin_boundary)

bcs = [No_Slip,In_Flow]

n=FacetNormal(Th) #Normal Vector

###############DEFINITION OF THE WEAK FORMULATION############

w_prev = Function(E)
(u_prev, p_prev) = w_prev.split()
u_0 = Expression(("(0.0001*x[1]*(1-x[1]))", "(0.0001*x[0]*x[1]*(1-x[1]))")) #Initialisation of the velocity
u_prev = interpolate(u_0, V)

w = TrialFunction(E)
u,p = as_vector((w[0],w[1])),w[2]

wt = TestFunction(E)
v,q = as_vector((wt[0],wt[1])),wt[2]

F = 1/dt*rho*inner(u-u_prev,v)*dx + rho*inner(grad(u)*u,v)*dx  \
  + mu*inner(grad(u), grad(v))*dx - mu*inner(v, (grad(u)*n))*ds(0)  \
  - mu*inner(v, (grad(u)*n))*ds(2) - p*div(v)*dx - q*div(u)*dx \
  - rho*inner(v,g)*dx

#Computing the solution

w_ = Function(E)
(u_, p_) = w_.split()
F = action(F, w_)

###############################ITERATIONS##########################
while (t <= end):
  solve(F==0, w_, bcs) #Solve the variational form
  u_prev.assign(u_) #u_prev = u_
  p_prev.assign(p_) #p_prev = p_
  plot(u_)
  t += float(dt)
  if (save==True):
    ufile << u_ #Save velocity
  
##########################END OF ITERATION##########################