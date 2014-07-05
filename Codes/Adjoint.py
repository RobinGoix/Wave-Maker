"""
This code contains several functions and a main code 
which is an optimization algorithm with variable step
"""
from dolfin import *
import numpy

global x0; x0 = -6.
global x1; x1 = 60.
global y0; y0 = -25.
global y1; y1 = 25.
 
Nx = 10 #Default 35
Ny = 5 #Default 22

#Scaling parameters
global g; g = 9.8
global lambda0; lambda0 = 20. #typical wavelength
global a0; a0 = 0.8 #Typical wave height
global h0; h0 = 2. #Typical depth
global sigma; sigma = h0/lambda0
global c0; c0 = (h0*g)**(0.5)
global epsilon; epsilon = a0/h0

global delta_t; delta_t = 0.05 #[s] Default 0.03 
delta_t = delta_t*c0/lambda0 #Time step
global end; end = 2.8 #Final Time
global T; T = numpy.arange(0,end,delta_t)
global N; N = T.size #number of timestep iterations

x0 = x0/lambda0
x1 = x1/lambda0
y0 = y0/lambda0
y1 = y1/lambda0

global mesh; mesh = RectangleMesh(x0,y0,x1,y1,Nx,Ny)

#Refine the mesh along the object's trajectory
cell_markers0 = CellFunction("bool", mesh)
cell_markers0.set_all(False)

for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() > -5./lambda0:
        cell_markers0[cell] = True
    
mesh = refine(mesh, cell_markers0)

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
global h; h = CellSize(mesh)

#Define functions spaces
#Velocity
global V; V = VectorFunctionSpace(mesh,'CG',2)
#Height
global H; H = FunctionSpace(mesh, 'CG', 1)
global E; E = MixedFunctionSpace([V, H])
#Object
global Z; Z = FunctionSpace(mesh, 'CG', 1)


#Shape of the seabed
hd = 2. #Depth [m]
hb = 0.3 #Depth at the boundaries [m]
ad = 0.8 #Object's height

hd = hd/h0 #depth
ad = ad/a0 #Object's height
hb = hb/h0  

seabed = 'hd - (hd-hb)/21.*(x[1]>4./lambda0 ? 1. : 0.)*(lambda0*x[1]-4.)' \
        + '+ (hd-hb)/21.*(x[1]<(-4./lambda0) ? 1. : 0.)*(lambda0*x[1]+4.)'
seabed = Expression(seabed, hd=hd, hb=hb, lambda0=lambda0) 
D = Function(project(seabed,H), name="Seabed")



#Trajectory of the object
Vmax = (g*(hd*h0+ad*a0))**0.5 #Physical maximal velocity
#Vmax = h0/a0*Vmax/c0 #Scaled maximal velocity
global Traj; Traj = [Vmax*lambda0/c0*T[k]*exp(-4./(lambda0/c0*T[k] + 0.05)) for k in range(0,N-1)]

#Dirichlet BC
# No-slip boundary
class Y_SlipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and \
            (near(x[1], y0) or near(x[1], y1))

class VelocityStream_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and \
            (near(x[0],x0) or near(x[0], x1))

class Entry_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],x1)
    
class Dirichlet_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],x0)

# Create no-slip boundary condition for velocity
bc_X_u = DirichletBC(E.sub(0), Expression(("0.0","0.0")), VelocityStream_Boundary())
bc_X_eta = DirichletBC(E.sub(1), 0.0, Entry_Boundary())
bc_Y_u = DirichletBC(E.sub(0).sub(1), 0.0, Y_SlipBoundary())

global bcs; bcs = [bc_X_u, bc_Y_u, bc_X_eta] 

def forward(zeta0):
    """
    This function takes as an argument an initial 
    shape and return the arrays of the solutions of
    the forward problem (U, Eta, Zeta) at each timestep
    """ 
    #Create the array of the object's positions
    Zeta = [translate(Z,zeta0,Traj[k]) for k in range(0,N-1)]
    
    #Initial Conditions
    w0 = project(Expression(("0.0", "0.0", "0.0")),E)
    ###############DEFINITION OF THE WEAK FORMULATION############  
    zeta__ = Function(zeta_initial, name="zeta_(n-1)")

    w_ = Function(w0, name="(u,eta)_(n)")
    u_, eta_ = split(w_)

    zeta_ = Function(zeta0, name="zeta_(n)")

    w = Function(E, name="(u,eta)_(n+1)")
    u, eta = split(w)

    v, chi = TestFunctions(E)
    
    #Time stepping methode
    alpha = 0.5
    u_alpha = (1.-alpha)*u_+ alpha*u
    eta_alpha = (1. - alpha)*eta_ + alpha*eta
    zeta_alpha = (1. - alpha)*zeta_ + alpha*zeta
    U_alpha = (1. - alpha)*U_ + alpha*U
    
    zeta_t = (zeta - zeta_)/delta_t
    zeta_tt = (zeta - 2.*zeta_ + zeta__)/delta_t**2
    
    F = 1./delta_t*inner(u-u_,v)*dx + epsilon*inner(grad(u_alpha)*u_alpha,v)*dx \
        - div(v)*eta_alpha*dx
    
    F += sigma**2.*1./delta_t*div((D + epsilon*zeta_alpha)*(u-u_))*div((D + epsilon*zeta_alpha)*v/2.)*dx \
        - sigma**2.*1./delta_t*div(u-u_)*div((D + epsilon*zeta_alpha)**2*v/6.)*dx \
        + sigma**2.*zeta_tt*div((D + epsilon*zeta_alpha)*v/2.)*dx
    
    F += 1./delta_t*(eta-eta_)*chi*dx + zeta_t*chi*dx \
        - (inner(u_alpha,grad(chi))*(epsilon*eta_alpha + D + epsilon*zeta_alpha))*dx 

    F += h**(3./2.)*(inner(grad(u_alpha),grad(v)) + inner(grad(eta_alpha),grad(chi)))*dx
    
    U_t = (U - U_)/delta_t
    
    A = zeta_t*xi*dx - epsilon*inner(grad(xi),U_)*zeta_*dx - epsilon*delta_t/2.*inner(grad(xi),U_t)*zeta_alpha*dx \
        + delta_t/2.*epsilon**2*inner(grad(xi),U_alpha)*inner(grad(zeta_alpha),U_alpha)*dx

    #First iteration to start the timestepper
    adj_start_timestep(time=0.0)
    #Solve transport equation
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
        U_.assign(U)
        U_obj.t = t
        U.assign(project(U_obj,V))
        #Solve the transport equation 
        solve(A==0, zeta, bc_zeta, annotate=annotate)
        #Solve the Peregrine system
        solve(F==0, w, bcs, annotate=annotate) #Solve the variational form
        w_.assign(w)
        u_, eta_ = w_.split()          
        zeta__.assign(zeta_)
        zeta_.assign(zeta)        
        t += float(delta_t)
        
        #Plot everything
        if (ploting):           
            if(t<=5*delta_t/2.):
                VizE = plot(eta_,rescale=True, title = "Free Surface")
            else:
                VizE.plot(eta_)#,rescale=True, title = "Free Surface")            
            plot(zeta_, mesh, rescale=False, title = "Seabed")
        if (save==True):
            fsfile << eta_ #Save heigth
            hfile << zeta_
    adj_inc_timestep(time=t,finished=True)
        ##############################END OF ITERATIONS#################################
    
    
    return U, Eta, Zeta, J


def adjoint(U,Eta,Zeta):
    """
    This function takes as an argument two arrays (U, Eta) of the solutions
    of the forward problem and returns two arrays of solutions of 
    adjoint problem (Va,Xi).
    """ 
    N = U.size() #number of time steps
    
    #Computing the adjoint at time step N
    wa = Function(W)
    v,xi = wa.split()
    u = TestFunction(V)
    eta = TestFunction(H) 
        
    A = 1/delta_t*inner(v,u)*dx + epsilon*inner(grad(U[N])*u + grad(u)*U[N],v)*dx \
        - (epsilon*Eta[N] + D + epsilon*Zeta[N])*inner(u,grad(xi))*dx 
    
    A += sigma**2./2.*1./delta_t*div((D+epsilon*Zeta[N])*u)*div((D+epsilon*Zeta[N])*v)*dx
    A -= sigma**2./6.*1./delta_t*div(u)*div((D+epsilon*Zeta[N])**2.*v)
    
    A += 1/delta_t*xi*eta*dx - eta*div(v)*dx - epsilon*eta*inner(U[N],grad(xi))*dx
    
    #A += Partial derivative of the functionnal w.r.t. Eta[N]
    
    solve(A == 0, wa, bca)
    wa_.assign(wa)
    v_,xi_ = wa_.split()
    v,xi = wa.split()
    Va[N] = v
    Xi[N] = xi
    
    for i in range(0,N):      
        #Next times step
        v_t = (v - v_)/delta_t
        xi_t = (xi - xi_)/delta_t
        
        #Adjoint system
        A = inner(v_t,u)*dx + epsilon*inner(grad(U[N-i])*u + grad(u)*U[N-i],v)*dx \
            - (epsilon*Eta[N-i] + D + epsilon*Zeta[N-i])*inner(u,grad(xi))*dx 
        
        A += sigma**2./2.*1./delta_t*div((D+epsilon*Zeta[N-i])*u)*div((D+epsilon*Zeta[N-i])*v-(D+epsilon*Zeta[N-i+1])*v_)*dx
        A -= sigma**2./6.*1./delta_t*div(u)*div((D+epsilon*Zeta[N-i])**2.*v - (D+epsilon*Zeta[N-i+1])**2.*v_)
        
        A += xi_t*eta*dx - eta*div(v)*dx - epsilon*eta*inner(U[N-i],grad(xi))*dx
            
        solve(A == 0, wa, bcs)
        wa_.assign(wa)
        v_,xi_ = wa_.split()
        v,xi = wa.split()
        
        Va[N-i] = v
        Xi[N-i] = xi
        
    return Va, Xi



def gradient(Zeta, U, Eta, Va, Xi):
    """
    This function takes as arguments the arrays of all 
    the computed solutions (forward, adjoint and object) 
    at each timestep and return the gradient of the
    functionnal for the corresponding shape.
    """
    J = Function(H)
    J = project(Expression('0.0'),H)
    
    for k in range(1,N):
            J += translate(H, Xi[k], Traj[k]) \
                - epsilon*1/delta_t*(dot(translate(V, U[k], Traj[k]),grad(translate(H, Xi[k], Traj[k]))) \
                    - dot(translate(V, U[k], Traj[k-1]),grad(translate(H, Xi[k], Traj[k-1]))))
            
            J += -epsilon*sigma**2./2.*1./delta_t*dot(translate(V, U[k], Traj[k]) - translate(V, U[k-1], Traj[k]), \
                 grad(div((D+epsilon*Zeta[0])*translate(V, Va[k], Traj[k]))))
            
            J += - epsilon*sigma**2./2.*1./delta_t*dot(translate(V, Va[k], Traj[k]), \
                grad(div((D+epsilon*Zeta[0])*(translate(V, U[k], Traj[k]) - translate(V, U[k-1], Traj[k])))))
            
            J += epsilon*sigma**2./3.*1./delta_t*dot((D+epsilon*Zeta[0])*translate(V, Va[k], Traj[k]), \
                grad(div(translate(V, U[k], Traj[k]) - translate(V, U[k-1], Traj[k]))))
            
            J += sigma**2./2.*1./delta_t**2*div((D+epsilon*translate(H, Zeta[k], Traj[k+1]))*translate(V, Va[k], Traj[k+1])\
                - 2*(D+epsilon*Zeta[0])*translate(V, Va[k], Traj[k]) \
                + (D+epsilon*translate(H, Zeta[k], Traj[k-1]))*translate(V, Va[k], Traj[k-1]))
            
            J += - epsilon*sigma**2./2.*1./delta_t*dot(grad(translate(H, Zeta[k+1], Traj[k]) - 2*Zeta[0] \
                + translate(H, Zeta[k-1], Traj[k])),translate(V, Va[k], Traj[k]))
    
    charac_obj = Function(H)
    charac_obj = project(Expression(('(x[1]<3./lambda0 ? 1. : 0.)'\
                    +'*(x[1]>-3./lambda0 ? 1. : 0.)*(x[0]>-3./lambda0 ? 1. : 0.)'\
                    +'*(x[0]<1.5/lambda0 ? 1. : 0.)'),ad=ad, lambda0=lambda0),Q) 
    charac_obj.vector()[:] = numpy.rint(shape_lb.vector()[:]) 
    J = charac_obj*J 
    
    return J


def translate(Q, h, fx):
    """
    This function takes as arguments a function h(x,y)
    of a space function Q and a contant fx and return 
    the translated of the function h(x+fx,y), with 
    value 0 if x-fx is not in the mesh
    """ 
    h_fx = Function(Q)
    h_fx = project(Expression('0.0'),Q)
    
    #Move the object using arrays
    x = numpy.array(mesh.coordinates()[:,0])
    y = numpy.array(mesh.coordinates()[:,1])
    x_min = x.min() #Define a boundary so that our object stays in the mesh
    x_max = x.max()
    #vertex_to_dof = Q.dofmap().vertex_to_dof_map(mesh)
    vertex_to_dof = dof_to_vertex_map(Q)
    x = x[vertex_to_dof]
    y = y[vertex_to_dof]
    for i in range(0,len(x)):
        #now find the x to calculate values
        x_fx = min(max(x[i] + fx,x_min),x_max) 
        point = numpy.array([x_fx, y[i]])#New coordinates after motion
        h_fx.vector()[i] = h(point) 
        plot(h_fx,mesh, interactive=True)
    
    return h_fx
    
    
mu_min = 0.00001 #Tolerance for the step
max_iter = 30 #Maximal number of iterations
counter = 0 #Initialize the iteration's counter

#Initialization of the shape
movingObject = '(x[1]<3/lambda0 ? 1. : 0.)*(x[1]>=0 ? 1. : 0.)*(lambda0*x[0]>-3 ? 1. : 0.)'\
    +'*ad*0.5*0.5*(1. - tanh(0.5*lambda0*x[1]-2.))*(tanh(10*(1.-lambda0*x[0]-pow(lambda0*x[1],2)/5))'\
    +'+ tanh(2*(lambda0*x[0]+pow(lambda0*x[1],2)/5 + 0.5)))'        
zeta0 = Expression(movingObject, ad=ad, c0=c0, hd=hd, lambda0=lambda0)
zeta0 = project(zeta0,H)
plot(zeta0, interactive=True)
controls = File('results/OptimalShape/Shape.pvd')
shape_viz = Function(H, name="ShapeVisualisation")
J_values=[]
def eval_cb(j, shape):
    '''
    This call-back function is called at each optimization iteration
    in order to store the shape and the functionnal values
    '''
    J_values.append(j)
    shape_viz.assign(shape)
    controls << shape_viz
    #Saving parameters for J_values
    arrayname = 'results/OptimalShape/Functional.npy'
    numpy.save(arrayname, J_values) #Save the array

#Initialization of the step by one gradient iteration
'''
So as to start with a coherent step mu, we do a first 
optimization iteration and use the maximal value of the computed
gradient to define that initial step.
'''
U, Eta, Zeta, J = forward(zeta0)
eval_cb(J, zeta0)
Va, Xi = adjoint(U, Eta, Zeta)
dJdzeta0 = gradient(Zeta, U, Eta, Va, Xi)
mu = 0.1/abs(dJdzeta0.vector()).max() 
zeta0_new = zeta0 + mu*dJdzeta0
U_new, Eta_new, Zeta_new, J_new = forward(zeta0_new)

if(J_new > J):
        mu = 1.5*mu
else:
    while(J_new < J and float(mu) > mu_min):
        mu = mu/2.
        zeta0_new = zeta0 + mu*dJdzeta0
        U_new, Eta_new, Zeta_new, J_new = forward(zeta0_new)
        
zeta0.assign(zeta0_new)
U, Eta, Zeta, J = U_new, Eta_new, Zeta_new, J_new
eval_cb(J, zeta0)

#################OPTIMIZATION ITERATIONS########################
while(float(mu) > mu_min and counter < max_iter):
    counter += 1 #increment the iteration counter
    Va, Xi = adjoint(U, Eta, Zeta)
    dJdzeta0 = gradient(Zeta, U, Eta, Va, Xi)      
    zeta0_new = zeta0 + mu*dJdzeta0
    U_new, Eta_new, Zeta_new, J_new = forward(zeta0_new)
    
    if(J_new > J):
        '''
        If the functional evaluated is bigger, we continue the 
        maximisation with a bigger step mu.
        '''
        mu = 1.5*mu
    else: #Otherwise we decrease the step until the functional is bigger
        while(J_new < J and float(mu) > mu_min):
            mu = mu/2.
            zeta0_new = zeta0 + mu*dJdzeta0
            U_new, Eta_new, Zeta_new, J_new = forward(zeta0_new)
            
    zeta0.assign(zeta0_new)
    U, Eta, Zeta, J = U_new, Eta_new, Zeta_new, J_new
    eval_cb(J, zeta0)


