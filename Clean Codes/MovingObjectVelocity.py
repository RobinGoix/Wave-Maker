
import scipy.integrate as si
from dolfin import *

Ny = 16
Nx = 64

x0 = -4
x1 = 10
y0 = -2
y1 = 2
Th = RectangleMesh(x0,y0,x1,y1,Nx,Ny)


#Define some Parameters
dt = 0.005 #Time step
t = 0.0	#Time initialization
end1 = 5.0 #Final time for the object

g = 9.8 #Gravity [m.s^(-2)]
h0 = 1 #depth  [m]
a0 = 0.4 #height of the moving object  [m]
bh = 0.7 #width of the moving object  [m]
xh = 0.0 #start position of the moving object  [m]
v0 = 2.0

#vh = Expression("v0*t",v0=v0,t=t) #speed of the moving object  [m.s^(-1)]
#def velocity(tt):
 # return 0.5*(tanh(3*(tt-0.6))+tanh(7*(4.0-tt)))
velocity = lambda tt: 0.5*(tanh(3*(tt-0.6))+tanh(7*(4.0-tt)))
vh = velocity(dt)
h_prev = Expression("h0-a0*exp(-(x[0]-xh)*(x[0]-xh)/(bh*bh))",h0=h0,xh=xh,bh=bh,a0=a0)
h = Expression("h0-a0*exp(-(x[0]-xh)*(x[0]-xh)/(bh*bh))",h0=h0,xh=xh,bh=bh,a0=a0)
h_next = Expression("h0-a0*exp(-(x[0]-xh-vh*dt)*(x[0]-xh-vh*dt)/(bh*bh))", dt=dt, h0=h0,xh=xh,bh=bh,a0=a0,vh=vh)



#Define functions spaces
#Height
H = FunctionSpace(Th, "Lagrange", 1) 

h_prev = interpolate(h_prev,H)
h = interpolate(h,H)
h_next = interpolate(h_next,H)
	
###############################ITERATIONS##########################
while (t <= end1):
  t += float(dt)
  h_prev.assign(h)
  h.assign(h_next)
  vh = velocity(t+dt)
  intvh=si.quad(velocity, 0, t)
  print(intvh[0])
  intvh=intvh[0]
  h_new = Expression("h0-a0*exp(-(x[0]-xh-intvh)*(x[0]-xh-intvh)/(bh*bh))",intvh=intvh, h0=h0,xh=xh,t=t,vh=vh,bh=bh,a0=a0,dt=dt)
  h_new = interpolate(h_new,H)
  h_next.assign(h_new)
  plot(h,rescale=False, title = "Seabed")
 
##############################END OF ITERATIONS#################################