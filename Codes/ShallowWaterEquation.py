"""
This code solve Shallow Water Equations on a unit square \
with a constant depth
"""

from dolfin import *
Nx = 64
Ny = 64
Th = UnitSquareMesh(Nx,Ny)

#Define Parameters
g = 9.8 #Gravity [m.s^(-2)]
dt = Constant(0.0005) #Time step [s]
t = 0.0	#Time initialization [s]
end = 0.5 #Final time [s]
bmarg = 1.e-3 + DOLFIN_EPS
save = True

#bottom profil
h = Constant(0.1) #[m]
#h = Expression("2-2*x[0]")

if (save == True):
  ufile = File("/home/robin/Documents/BCAM/FEniCS_Files/Simulations/SWE/SWECircle2/SWECircle2.pvd") #To save data in a file

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
#Initial Velocity
u_0 = Expression(("0.0", "0.0")) #Initialisation of the velocity

#Initial Height
#Parameter for the circle
sigma = 0.0005 
R = 0.1 #Radius
xC = 0.3 #center x
yC = 0.7#center y 
A = 0.01 #Amplitude [m]
#eta_0 = Expression("0.01*(exp(-(x[0]+2*x[1]-2)*(x[0]+2*x[1]-2)/0.01))")
#eta_0 = Expression("0.5*(exp(-(x[0])*(x[0])/0.01))")

eta_0 = Expression("A*(exp(-((x[0]-xC)*(x[0]-xC)+ \
		  (x[1]-yC)*(x[1]-yC)-R*R)*((x[0]-xC)*(x[0]-xC)+ \
		  (x[1]-yC)*(x[1]-yC)-R*R)/sigma))", \
		    sigma=sigma, R=R, xC=xC, yC=yC, A=A)


###############DEFINITION OF THE WEAK FORMULATION############

w_prev = Function(E)
(u_prev, eta_prev) = w_prev.split()

u_prev = interpolate(u_0, V)

eta_prev = interpolate(eta_0,H)

w = TrialFunction(E)
u,eta = as_vector((w[0],w[1])),w[2]

wt = TestFunction(E)
v,xi = as_vector((wt[0],wt[1])),wt[2]

F = 1/dt*inner(u-u_prev,v)*dx + inner(grad(u)*u,v)*dx  \
      - g*div(v)*eta*dx + 1/dt*(eta-eta_prev)*xi*dx \
      - div(u)*(eta)*xi*dx - inner(u,grad(xi))*(eta+h)*dx \
      + (eta+h)*div(u)*xi*dx 
    
w_ = Function(E)
(u_, eta_) = w_.split()
F = action(F, w_)

###############################ITERATIONS##########################
while (t <= end):
  solve(F==0, w_, bc) #Solve the variational form
  u_prev.assign(u_) #u_prev = u_
  eta_prev.assign(eta_) #p_prev = p_
  plot(eta_,rescale=False)
  t += float(dt)
  if (save == True):
    ufile << eta_ #Save heigth
      
