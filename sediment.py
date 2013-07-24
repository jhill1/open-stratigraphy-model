from dolfin import *
import numpy
import sys

class SedimentModel:
    """A class to track a single sediment type
       on a non-uniform domain, using diffusion only
       to move sediment around
    """

    def __init__(self):
        """Set up sensible initial values
        """
        self.V = None
        self.FacetNorm = None
        self.mesh = None
        self.s0 = None
        self.end_time = 1
        self.dt = 1
        self.alpha = Constant(0.1)

    def set_mesh(self,mesh):
        """Set which mesh to use and define function space"""
        self.mesh = mesh
        self.V = FunctionSpace(self.mesh, "Lagrange", 1)
        self.n = FacetNormal(self.mesh)


    def set_initial_conditions(self,topography,sediment):
        # Set the initial topography and sediment with
        # UFL expressions
        self.h0 = topography
        self.s0 = sediment

    def set_timestep(self,dt):
        self.dt = dt

    def set_end_time(self,time):
        self.end_time = time

    def init(self,plot_init=False):
        """Initialise the solvers, etc, and setup the problem
        """

        tiny = 1e-16

        # initial guess of solution
        self.s_1 = interpolate(self.s0, self.V)
        self.s = TrialFunction(self.V)
        self.h = interpolate(self.h0, self.V)

        if (plot_init):
            plot(get_total_height(),interactive=True)

        self.v = TestFunction(self.V)
        self.f = Constant(0)
        
        # This limiter stops the diffuive process where the amount of sediment (s) is zero
        limit = (self.s_1 + abs(self.s_1))/(2*self.s_1 + tiny) # limiting term

        # RHS
        self.a = (self.s)*self.v*dx + limit*self.dt*inner(nabla_grad(self.s+self.h), nabla_grad(self.v))*self.alpha*dx
        # LHS
        self.L = (self.s_1 + self.dt*self.f)*self.v*dx
        
        F = self.a - self.L
        self.a, self.L = system(F)
        self.b = None
        self.A = assemble(self.a)   # assemble only once, before the time stepping

        self.s = Function(self.V)   # the unknown at a new time level
        self.T = self.end_time      # total simulation time

    def solve(self):
        """Solve the problem"""
        t = 0
        while t <= self.T:
            self.b = assemble(self.L)
            #bc.apply(A, b)
            solve(self.A, self.s.vector(), self.b)

            self.b = assemble(self.L, tensor=self.b)
            t += self.dt
            #plot(u, interactive=True)
            self.s_1.assign(self.s)
            #plot(model.sediment_height(),interactive=True)

    def get_total_height_array(self):
        return self.s_1.vector().array()+self.h.vector().array()

    def get_total_height(self):
        return self.s_1+self.h

    def get_sed_height_array(self):
        return self.s_1.vector().array()

    def get_sed_height(self):
        return self.s_1

    def get_topographic_height_array(self):
        return self.h_1.vector().array()

    def get_topographic_height(self):
        return self.h_1

    def boundary(x, on_boundary):  # define the Dirichlet boundary
        return on_boundary

    def set_diffusion_coeff(self,coeff):
        self.alpha = Constant(coeff)

if __name__ == "__main__":
    
    #create a simple testcase
    model = SedimentModel()
    mesh = UnitSquare(10,10)
    model.set_mesh(mesh)
    init_cond = Expression('x[0]') # simple slope
    init_sed = Expression('x[0]') # this gives
    # total of above gives a slope of 0 to 2 (over the unit square)
    model.set_initial_conditions(init_cond,init_sed)
    model.set_end_time(10)
    model.set_diffusion_coeff(1)
    model.init()
    model.solve()
    # answer should be 1 everywhere
    plot(model.get_total_height(),interactive=True)
    print model.get_total_height_array()


