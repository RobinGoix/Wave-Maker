"""
Optimal coefficients for transport equation
"""

from dolfin import *
from dolfin_adjoint import * 
#import pyipopt

set_log_active(False)

def main(alpha):#,source,stab):
    Nx = 75
    Ny = 55

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

    mesh_zeta = RectangleMesh(x0,y0,x1,y1,Nx,Ny)

    #Refine the mesh along the object's trajectory
    cell_markers = CellFunction("bool", mesh_zeta)
    cell_markers.set_all(False)

    for cell in cells(mesh_zeta):
        p = cell.midpoint()
        if p.y() > -4./lambda0 and p.y() < 4./lambda0:
            cell_markers[cell] = True
        
    mesh_zeta = refine(mesh_zeta, cell_markers)

    cell_markers2 = CellFunction("bool", mesh_zeta)
    cell_markers2.set_all(False)

    for cell in cells(mesh_zeta):
        p = cell.midpoint()
        if p.y() > -3./lambda0 and p.y() < 3./lambda0:
            cell_markers2[cell] = True
        
    mesh_zeta = refine(mesh_zeta, cell_markers2)

    cell_markers3 = CellFunction("bool", mesh_zeta)
    cell_markers3.set_all(False)

    for cell in cells(mesh_zeta):
        p = cell.midpoint()
        if p.y() > -3./lambda0 and p.y() < 3./lambda0:
            cell_markers3[cell] = True
        
    mesh_zeta = refine(mesh_zeta, cell_markers3)

    cell_markers4 = CellFunction("bool", mesh_zeta)
    cell_markers4.set_all(False)

    for cell in cells(mesh_zeta):
        p = cell.midpoint()
        if p.y() > -3./lambda0 and p.y() < 3./lambda0:
            cell_markers4[cell] = True
        
    mesh_zeta = refine(mesh_zeta, cell_markers4)
    
    cell_markers4 = CellFunction("bool", mesh_zeta)
    cell_markers4.set_all(False)

    for cell in cells(mesh_zeta):
        p = cell.midpoint()
        if p.y() > -3./lambda0 and p.y() < 3./lambda0:
            cell_markers4[cell] = True
        
    mesh_zeta = refine(mesh_zeta, cell_markers4)
    
    plot(mesh_zeta)
    h = CellSize(mesh_zeta)


    #Other Parameters
    save = False
    ploting = True

    hd = 2. #Depth [m]
    hb = 0.3 #Depth at the boundaries [m]
    ad = 0.8 #Object's height

    hd = hd/h0 #depth
    ad = ad/a0 #Object's height
    hb = hb/h0

    #Trajectory of the object
    Vmax = (g*(hd*h0+ad*a0))**0.5 #Physical maximal velocity
    Vmax = h0/a0*Vmax/c0 #Scaled maximal velocity
    #Time dependant velocity
    U = Expression(('Vmax*(1.+4.*lambda0/c0*t/pow((lambda0/c0*t+0.05),2))*exp(-4./(lambda0/c0*t + 0.05))','0.0'),\
                    Vmax=Vmax, lambda0=lambda0, c0=c0, t=0.0)
    U_ = Expression(('Vmax*(1.+4.*lambda0/c0*t/pow((lambda0/c0*t+0.05),2))*exp(-4./(lambda0/c0*t + 0.05))','0.0'),\
                Vmax=Vmax, lambda0=lambda0, c0=c0, t=0.0)
    U__ = Expression(('Vmax*(1.+4.*lambda0/c0*t/pow((lambda0/c0*t+0.05),2))*exp(-4./(lambda0/c0*t + 0.05))','0.0'),\
            Vmax=Vmax, lambda0=lambda0, c0=c0, t=0.0)
    #Corresponding trajectory
    traj = 'epsilon*c0*Vmax*lambda0/c0*t*exp(-4./(lambda0/c0*t + 0.05))'

    #Initial Conditions
    movingObject = ' - (x[1]<3/lambda0 ? 1. : 0.)*(x[1]>0 ? 1. : 0.)*(lambda0*x[0]>-6 ? 1. : 0.)'\
        +'*ad*0.5*0.5*(1. - tanh(0.5*lambda0*x[1]-2.))*(tanh(10*(1.-lambda0*x[0]-pow(lambda0*x[1],2)/5))'\
        +'+ tanh(2*(lambda0*x[0]+pow(lambda0*x[1],2)/5 + 0.5))) ' \
        + ' - (x[1]>-3/lambda0 ? 1. : 0.)*(x[1]<=0 ? 1. : 0.)*(lambda0*x[0]>-6 ? 1. : 0.)'\
        +'*ad*0.5*0.5*(1. + tanh(0.5*lambda0*x[1]+2.))*(tanh(10*(1. - lambda0*x[0]-pow(lambda0*x[1],2)/5))'\
        +'+ tanh(2*(lambda0*x[0]+pow(lambda0*x[1],2)/5 + 0.5))) ' 
    zeta0 = Expression(movingObject, ad=ad, c0=c0, hd=hd, lambda0=lambda0, epsilon=epsilon)

    zeta_comparison = ' - (x[1]<3/lambda0 ? 1. : 0.)*(x[1]>0 ? 1. : 0.)*(lambda0*x[0]-'+traj+'>-6 ? 1. : 0.)'\
        +'*ad*0.5*0.5*(1. - tanh(0.5*lambda0*x[1]-2.))*(tanh(10*(1.-(lambda0*x[0]-'+traj+')-pow(lambda0*x[1],2)/5))'\
        +'+ tanh(2*(lambda0*x[0]-'+traj+'+pow(lambda0*x[1],2)/5 + 0.5))) ' \
        + ' - (x[1]>-3/lambda0 ? 1. : 0.)*(x[1]<=0 ? 1. : 0.)*(lambda0*x[0]-'+traj+'>-6 ? 1. : 0.)'\
        +'*ad*0.5*0.5*(1. + tanh(0.5*lambda0*x[1]+2.))*(tanh(10*(1. - (lambda0*x[0]-'+traj+')-pow(lambda0*x[1],2)/5))'\
        +'+ tanh(2*(lambda0*x[0]-'+traj+'+pow(lambda0*x[1],2)/5 + 0.5))) '
    zeta_comparison = Expression(zeta_comparison, Vmax=Vmax, ad=ad, c0=c0, hd=hd, lambda0=lambda0,t=0.0, epsilon=epsilon)

    filtre = '(lambda0*x[0] < 2 + '+traj+' ? 1. : 0.)*(lambda0*x[0] > -6 + '+traj+'? 1. : 0.)'\
            +'*(lambda0*x[1] < 3 ? 1. : 0.)*(lambda0*x[1] > -3 ? 1. : 0.)'
    filtre = Expression(filtre, Vmax=Vmax, t=0.0, lambda0=lambda0, c0=c0, epsilon=epsilon)

    #Saving parameters
    if (save==True):
        hfile = File("results/Objectshape1/MB.pvd") #To save data in a file
        
    #Define functions spaces
    V = VectorFunctionSpace(mesh_zeta, 'CG',1)
    Q = FunctionSpace(mesh_zeta, 'CG',1)

    #Dirichlet BC   
    class Dirichlet_Boundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < x0 + DOLFIN_EPS

    bc_zeta = DirichletBC(Q, 0.0, Dirichlet_Boundary())

    ###############DEFINITION OF THE WEAK FORMULATION############      
    zeta_ = Function(Q)
    zeta_ = interpolate(zeta0, Q)

    zeta = Function(Q)
    zeta_c = Function(Q)
    zeta_c = interpolate(zeta_comparison,Q)
    xi = TestFunction(Q)

    #Time stepping methode
    zeta_alpha = (1. - alpha)*zeta_ + alpha*zeta
    zeta_t = (zeta-zeta_)/delta_t
    alpha2 = 1.
    U_alpha = (1. - alpha2)*U_ + alpha2*U
    U_t = (U - U_)/delta_t
    U_tt = (U-2*U_+U__)/delta_t**2
    
    A = zeta_t*xi*dx - epsilon*inner(grad(xi),U_alpha)*zeta_*dx - epsilon*delta_t/2.*inner(grad(xi),U_t)*zeta_alpha*dx \
        + delta_t/2.*epsilon**2*inner(grad(xi),U_alpha)*inner(grad(zeta_alpha),U_alpha)*dx
    """
    #Third order Taylor-Galerkin method term
    A += delta_t**2/6.*epsilon**2*inner(grad(xi),U_alpha)*inner(U_alpha,grad(zeta_t))*dx

    A += - delta_t**2/6*(epsilon*inner(grad(xi),U_tt)*zeta_alpha*dx\
        - 2*epsilon**2*inner(grad(xi),U_t)*inner(grad(zeta_alpha),U_alpha)*dx\
        - epsilon**2*inner(grad(xi),U_alpha)*inner(grad(zeta_alpha),U_t)*dx)
    """
    """
    + delta_t/6.*inner(epsilon*U_alpha,grad(xi))*inner(epsilon*U_alpha,grad(zeta_alpha))*dx \
         - delta_t/2.*inner(grad(grad(xi))*epsilon*U_alpha,epsilon*U_alpha)*zeta_alpha*dx 
    """
    #A += - source*epsilon*inner(U_alpha,U_alpha)**0.5*zeta*xi*dx
    #A += stab*epsilon*inner(U_alpha,U_alpha)**0.5*h**(3/2)*inner(grad(zeta_alpha),grad(xi))*dx
    """
    A = zeta_t*xi*dx - epsilon*inner(grad(xi),U)*zeta_alpha*dx - source*epsilon*Source_T*zeta*xi*dx
    A += stab*epsilon*Source_T*h**(3/2)*inner(grad(zeta_alpha),grad(xi))*dx
    """

    #First iteration to initialize the loop
    adj_start_timestep(time=0.0)
    U__.t = t
    U_.t = t
    U.t = t
    solve(A==0, zeta, bc_zeta)
    zeta_.assign(zeta) 
    zeta_comparison.t = t
    zeta_c.assign(zeta_comparison) 
    t += float(delta_t) 

    ###############################ITERATIONS##########################
    while (t <= end):
        adj_inc_timestep(time=t,finished=False)
        U.t = t
        U_.t = t - delta_t
        U__.t = t-2*delta_t
        #U.assign(interpolate(U_Object,V))
        solve(A==0, zeta, bc_zeta)
        zeta_.assign(zeta) 
        zeta_comparison.t = t
        zeta_c.assign(zeta_comparison)
        #filtre.t = t  
        t += float(delta_t) 
        if (ploting==True):
            plot(zeta_, mesh_zeta, rescale=True, title = "Moving Object")
            plot(zeta_c,mesh_zeta,rescale=True, title = "Comparions")
            plot(U_,mesh)
            #plot(zeta-zeta_c,mesh_zeta,title=error)
        if (save==True):
            hfile << zeta_ #Save heigth
    ##############################END OF ITERATIONS#################################
    adj_inc_timestep(time=t,finished=True)
    
    return zeta, zeta_c


if __name__ == "__main__":
    #Optimization
    alpha = Constant(0.5)
    #source = Constant(0.0)
    #stab = Constant(0.0)
    zeta, zeta_c = main(alpha)#,source,stab)

    J = Functional(inner(zeta-zeta_c,zeta-zeta_c)*dx*dt)

    p_alpha = ScalarParameter(alpha)
    #p_source = ScalarParameter(source)
    #p_stab = ScalarParameter(stab)

    Jhat = ReducedFunctional(J, p_alpha)#, p_source, p_stab])

    #m_opt = minimize(Jhat, bounds=[[0.0, 0.0, 0.0],[1.0, 10., 10]], options = {'disp':True, 'maxiter':10})
    m_opt = minimize(Jhat, bounds=(0.0, 1.0), options = {'disp':True, 'maxiter':10})
    
    print('alpha = ', float(m_opt))
    """
    print('alpha = ', float(m_opt[0]))
    print('source = ', float(m_opt[1]))
    print('stab = ', float(m_opt[2]))

    """

