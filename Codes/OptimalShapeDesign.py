"""
Optimal shape design of a wave maker
"""

from dolfin import *
from dolfin_adjoint import * 
#import pyipopt
import numpy

#set_log_active(False)
Nx = 20
Ny = 10

x0 = -6.
x1 = 60.
y0 = -25.
y1 = 25.
    
#Scaling parameters
g = 9.8
lambda0 = 20. #typical wavelength
a0 = 0.8 #Typical wave height
h0 = 2. #Typical depth
sigma = h0/lambda0
c0 = (h0*g)**(0.5)
epsilon = a0/h0

delta_t = 0.03 #timestep [s]
t = 0.0 #time initialization
end = 2.8 #Final Time
delta_t = delta_t*c0/lambda0 #Time step
t = t*c0/lambda0 #Time initialization
#end = end*c0/lambda0 #Final time

x0 = x0/lambda0
x1 = x1/lambda0
y0 = y0/lambda0
y1 = y1/lambda0
mesh = RectangleMesh(x0,y0,x1,y1,Nx,Ny)

#Refine the mesh along the object's trajectory
cell_markers = CellFunction("bool", mesh)
cell_markers.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -4./lambda0:
        cell_markers[cell] = True
    
mesh = refine(mesh, cell_markers)

cell_markers2 = CellFunction("bool", mesh)
cell_markers2.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3.5/lambda0 and p.y() < 20./lambda0:
        cell_markers2[cell] = True
    
mesh = refine(mesh, cell_markers2)

cell_markers3 = CellFunction("bool", mesh)
cell_markers3.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 18./lambda0:
        cell_markers3[cell] = True
    
mesh = refine(mesh, cell_markers3)

cell_markers4 = CellFunction("bool", mesh)
cell_markers4.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 15./lambda0:
        cell_markers4[cell] = True
    
mesh = refine(mesh, cell_markers4)

cell_markers5 = CellFunction("bool", mesh)
cell_markers5.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -3./lambda0 and p.y() < 3./lambda0:
        cell_markers5[cell] = True
    
mesh = refine(mesh, cell_markers5)

h = CellSize(mesh)

#Other Parameters
save = False
ploting = False

hd = 2. #Depth [m]
hb = 0.3 #Depth at the boundaries [m]
ad = 0.8 #Object's height

hd = hd/h0 #depth
ad = ad/a0 #Object's height
hb = hb/h0

#Shape of the seabed
seabed = 'hd - (hd-hb)/21.*(x[1]>4./lambda0 ? 1. : 0.)*(lambda0*x[1]-4.)' \
        + '+ (hd-hb)/21.*(x[1]<(-4./lambda0) ? 1. : 0.)*(lambda0*x[1]+4.)'
seabed = Expression(seabed, hd=hd, hb=hb, lambda0=lambda0) 

#Trajectory of the object
Vmax = (g*(hd*h0+ad*a0))**0.5 #Physical maximal velocity
Vmax = h0/a0*Vmax/c0 #Scaled maximal velocity
#Time dependant velocity
U_obj = Expression(('Vmax*(1.+4.*lambda0/c0*t/pow((lambda0/c0*t+0.05),2))*exp(-4./(lambda0/c0*t + 0.05))','0.0'),\
                Vmax=Vmax, lambda0=lambda0, c0=c0, t=0.0)

#Saving parameters
if (save==True):
    fsfile = File("results/Objectshape2/FS.pvd") #To save data in a file
    hfile = File("results/Objectshape2/MB.pvd") #To save data in a file
    
#Define functions spaces
#Velocity
V = VectorFunctionSpace(mesh,'CG',1)
#Height
H = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(mesh, 'CG',1)
E = MixedFunctionSpace([V, H])

#Dirichlet BC
# No-slip boundary
class Y_SlipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and \
            (x[1] < y0 + DOLFIN_EPS or x[1] > y1 - DOLFIN_EPS)

class VelocityStream_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and \
            (x[0] < x0 + DOLFIN_EPS or x[0] > x1 - DOLFIN_EPS)

class Entry_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] > x1 - DOLFIN_EPS)
    
class Dirichlet_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Create no-slip boundary condition for velocity
bc_X_u = DirichletBC(E.sub(0), Expression(("0.0","0.0")), VelocityStream_Boundary())
bc_X_eta = DirichletBC(E.sub(1), 0.0, Entry_Boundary())
bc_Y_u = DirichletBC(E.sub(0).sub(1), 0.0, Y_SlipBoundary())

bc_zeta = DirichletBC(Q, 0.0, Dirichlet_Boundary())

bcs = [bc_X_u, bc_Y_u, bc_X_eta] 

def main(zeta_initial, p, annotate=True):
    parameters["adjoint"]["test_derivative"] = True
    delta_t = 0.1 #timestep [s]
    t = 0.0 #time initialization
    end = 2.8 #Final Time
    #delta_t = delta_t*c0/lambda0 #Time step
    t = t*c0/lambda0 #Time initialization
    #end = end*c0/lambda0 #Final time
    
    #Initial Conditions
    w0 = project(Expression(("0.0", "0.0", "0.0")),E)
    
    U = Function(project(U_obj,V),name="Velocity_(n)", annotat=annotate)
    U_ = Function(project(U_obj,V),name="Velocity_(n-1)", annotat=annotate)
    
    D = Function(project(seabed,H), name="Seabed", annotate=annotate)
    
    ###############DEFINITION OF THE WEAK FORMULATION############  
    zeta__ = Function(zeta_initial, name="zeta_(n-1)", annotate=annotate)
    #zeta__.assign(zeta_initial)

    w_ = Function(w0, name="(u,eta)_(n)", annotate=annotate)
    u_, eta_ = split(w_)

    zeta_ = Function(zeta_initial, name="zeta_(n)", annotate=annotate)
    #zeta_.assign(zeta_initial)

    w = Function(E, name="(u,eta)_(n+1)", annotate=annotate)
    u, eta = split(w)
    zeta = Function(Q, name="zeta_(n+1)", annotate=annotate)

    v, chi = TestFunctions(E)
    xi = TestFunction(Q)

    #Time stepping methode
    alpha = 0.5
    u_alpha = (1.-alpha)*u_+ alpha*u
    eta_alpha = (1. - alpha)*eta_ + alpha*eta
    zeta_alpha = (1. - alpha)*zeta_ + alpha*zeta

    alpha2 = 1.
    U_alpha = (1. - alpha2)*U_ + alpha2*U
    U_t = (U - U_)/delta_t
        
    zeta_t = p*(zeta - zeta_)/delta_t
    zeta_tt = p*(zeta - 2.*zeta_ + zeta__)/delta_t**2

    F = 1./delta_t*inner(u-u_,v)*dx + epsilon*inner(grad(u_alpha)*u_alpha,v)*dx \
        - div(v)*eta_alpha*dx
    """
    F += sigma**2.*1./delta_t*div((D + epsilon*zeta_alpha)*(u-u_))*div((D + epsilon*zeta_alpha)*v/2.)*dx \
        - sigma**2.*1./delta_t*div(u-u_)*div((D + epsilon*zeta_alpha)**2*v/6.)*dx \
        + sigma**2.*zeta_tt*div((D + epsilon*zeta_alpha)*v/2.)*dx
    """
    F += 1./delta_t*(eta-eta_)*chi*dx + zeta_t*chi*dx \
        - (inner(u_alpha,grad(chi))*(epsilon*eta_alpha + D + epsilon*zeta_alpha))*dx 

    F += 0.1*h**(3./2.)*(inner(grad(u_alpha),grad(v)) + inner(grad(eta_alpha),grad(chi)))*dx

    A = zeta_t*xi*dx - epsilon*inner(grad(xi),U_alpha)*zeta_*dx - epsilon*delta_t/2.*inner(grad(xi),U_t)*zeta_alpha*dx \
        + delta_t/2.*epsilon**2*inner(grad(xi),U_alpha)*inner(grad(zeta_alpha),U_alpha)*dx

    #First iteration to start the timestepper
    adj_start_timestep(time=0.0)
    #Solve transport equation
    U_obj.t = t
    U.assign(project(U_obj,V))
    solve(A==0, zeta, bc_zeta, annotate=annotate)

    #Solve Peregrine
    solve(F==0, w, bcs, annotate=annotate)
    zeta__.assign(zeta_)
    zeta_.assign(zeta)
    w_.assign(w)
    u_, eta_ = w_.split()

    t += float(delta_t) 

    ###############################ITERATIONS##########################
    while (t <= end):  
        adj_inc_timestep(time=t,finished=False)

        #Solve the transport equation 
        U_.assign(U)
        U_obj.t = t
        U.assign(project(U_obj,V))
        solve(A==0, zeta, bc_zeta, annotate=annotate)
        #Solve the Peregrine system
        solve(F==0, w, bcs, annotate=annotate) #Solve the variational form
        w_.assign(w)
        u_, eta_ = w_.split()  
        
        zeta__.assign(zeta_)
        zeta_.assign(zeta) 
        
        t += float(delta_t)
        print(t)
        #Plot everything
        if (ploting==True):
            if(t<=5*delta_t/2.):
                VizE = plot(eta_,rescale=True, title = "Free Surface")
            else:
                VizE.plot(eta_)#,rescale=True, title = "Free Surface")
            plot(zeta_, mesh, rescale=True, title = "Seabed")
        if (save==True):
            fsfile << eta_ #Save heigth
            
    adj_inc_timestep(time=t,finished=True)
        ##############################END OF ITERATIONS#################################

    return eta_
#####################OPTIMIZATION############################

if __name__ == "__main__":
    #Call-back to vizualize the shape at each iteration
    """
    controls = File("results/OptimalShape/shape_iterations.pvd")
    shape_viz = Function(Q, name="ShapeVisualisation")
    J_values=[]
    def eval_cb(j, shape):
        plot(shape)
        J_values.append(j)
        shape_viz.assign(shape)
        controls << shape_viz
        #Saving parameters for J_values
        arrayname = 'results/OptimalShape/functional_values.npy'
        numpy.save(arrayname, J_values)
    """
    #Mark the subdomain on which the wave is assessed
    """
    class OptDomain(SubDomain):
        def inside(self, x, on_boundary):
            X0 = 50./lambda0
            X1 = 58./lambda0
            Y0 = 4./lambda0
            Y1 = 16./lambda0
            
            return x[0]>=X0 and x[0]<=X1 and x[1]>=Y0 and x[1]<=Y1
        

    domains = CellFunction("size_t", mesh)
    domains.set_all(0)
    OptDomain().mark(domains, 1)
    dx_ = Measure("dx")[domains]
    """
    #Initialization of the shape
    movingObject = ' - (x[1]<3/lambda0 ? 1. : 0.)*(x[1]>0 ? 1. : 0.)*(lambda0*x[0]>-4 ? 1. : 0.)'\
        +'*ad*0.5*0.5*(1. - tanh(0.5*lambda0*x[1]-2.))*(tanh(10*(1.-lambda0*x[0]-pow(lambda0*x[1],2)/5))'\
        +'+ tanh(2*(lambda0*x[0]+pow(lambda0*x[1],2)/5 + 0.5))) ' \
        + ' - (x[1]>-3/lambda0 ? 1. : 0.)*(x[1]<=0 ? 1. : 0.)*(lambda0*x[0]>-4 ? 1. : 0.)'\
        +'*ad*0.5*0.5*(1. + tanh(0.5*lambda0*x[1]+2.))*(tanh(10*(1. - lambda0*x[0]-pow(lambda0*x[1],2)/5))'\
        +'+ tanh(2*(lambda0*x[0]+pow(lambda0*x[1],2)/5 + 0.5))) ' 
    
    zeta0 = Expression(movingObject, ad=ad, c0=c0, hd=hd, lambda0=lambda0)
    zeta_initial = project(zeta0,Q)
    #shape = InitialConditionParameter("zeta_(n)")
    p = Constant(1.0)
    m = InitialConditionParameter(zeta_initial)
    #Computation of the functional
    eta = main(zeta_initial, p)
    success = replay_dolfin(tol=0.0, stop=True)
    
    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")
    
    J = Functional(inner(eta,eta)*dx*dt)
    #J_adj = compute_adjoint(J,m)
    #print type(J_adj)
    
    dJdshape = compute_gradient(J, m)#, forget=False)
    plot(dJdshape, mesh, interactive=True)
    
    Jshape = assemble(inner(eta,eta)*dx)
    
    #parameters["adjoint"]["stop_annotating"] = True # stop registering equations
    
    def Jhat(zeta_initial): # the functional as a pure function of nu
        eta = main(zeta_initial, p, False)
        return assemble(inner(eta,eta)*dx)
    
    conv_rate = taylor_test(Jhat, m, Jshape, dJdshape)#, seed=0.001)  
    
    #Define the constraints/bounds
    """
    shape_ub = Function(Q, name="Upper_Bound")
    shape_ub.vector()[:] = 0   
    shape_lb = Function(Q, name="Lower_Bound")
    shape_lb = project(Expression(('(-ad*(x[1]<3./lambda0 ? 1. : 0.)'\
                    +'*(x[1]>-3./lambda0 ? 1. : 0.)*(x[0]>-4./lambda0 ? 1. : 0.)'\
                    +'*(x[0]<2./lambda0 ? 1. : 0.) < -0.5 ? - ad*(x[1]<3./lambda0 ? 1. : 0.)'\
                    +'*(x[1]>-3./lambda0 ? 1. : 0.)*(x[0]>-4./lambda0 ? 1. : 0.)'\
                    +'*(x[0]<2./lambda0 ? 1. : 0.) : 0 )'),ad=ad, lambda0=lambda0),Q) 
    shape_lb.vector()[:] = numpy.rint(shape_lb.vector()[:])
    """    
    
    #Jhat = ReducedFunctional(J, shape, eval_cb=eval_cb)  
    

    #J_adj = compute_adjoint(J)
    #for (adjoint, var) in compute_adjoint(J, forget=False):
    #   if var.name == "(u,eta)_(n)":
    #      (adj_u, adj_p) = split(adjoint)
    #     plot(adj_p)
    
    #shape_opt = minimize(Jhat, bounds=(shape_lb, shape_ub), options = {'disp':True, 'maxiter':2})
    
    #With pyipopt
    """
    jhat = ReducedFunctionalNumpy(Jhat)
    
    nlp = jhat.pyipopt_problem(bounds=(shape_lb.vector(), shape_ub.vector()))
    
    shape_opt = nlp.solve(full=False)
    """


