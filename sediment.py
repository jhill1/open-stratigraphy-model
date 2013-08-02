from dolfin import *
import numpy
import sys

# left boundary marked as 0, right as 1
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.5 and on_boundary

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
        self.alpha = Constant(1)
        self.inflow_rate = Expression('0')
        self.output_time = 100000

    def set_mesh(self,mesh):
        """Set which mesh to use and define function space"""
        self.mesh = mesh
        self.V = FunctionSpace(self.mesh, "Lagrange", 2)
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

        tiny = 1e-10

        # initial guess of solution
        self.s_1 = interpolate(self.s0, self.V)
        self.s = TrialFunction(self.V)
        self.h = interpolate(self.h0, self.V)

        if (plot_init):
            plot(self.get_sed_height(),interactive=True)

        self.v = TestFunction(self.V)
        self.f = Constant(0)
        
        # This limiter stops the diffuive process where the amount of sediment (s) is zero
        limit = (self.s_1 + abs(self.s_1))/(2*self.s_1 + tiny) # limiting term
        
        # Set up inflow boundary
        left_boundary = LeftBoundary()
        self.exterior_facet_domains = FacetFunction("uint", self.mesh)
        self.exterior_facet_domains.set_all(1)
        left_boundary.mark(self.exterior_facet_domains, 0)
        self.ds = Measure("ds")[self.exterior_facet_domains] 

        # RHS
        self.a = (self.s)*self.v*dx + limit*self.dt*inner(nabla_grad(self.s+self.h), nabla_grad(self.v))*self.alpha*dx
        # LHS
        self.L = (self.s_1 + self.dt*self.f)*self.v*dx - self.inflow_rate*self.v*self.ds(0)
        
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
            self.b = assemble(self.L, exterior_facet_domains=self.exterior_facet_domains)
            solve(self.A, self.s.vector(), self.b)

            self.b = assemble(self.L, tensor=self.b, exterior_facet_domains=self.exterior_facet_domains)
            t += self.dt
            #plot(u, interactive=True)
            self.s_1.assign(self.s)
            if (t%self.output_time == 0):
                plot(self.get_total_height(),interactive=True)

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
    mesh = UnitInterval(100)
    model.set_mesh(mesh)
    init_cond = Expression('0') # simple slope
    init_sed = Expression('0') # this gives
    # total of above gives a slope of 0 to 2 (over the unit square)
    model.set_initial_conditions(init_cond,init_sed)
    model.set_end_time(10)
    model.set_diffusion_coeff(0.0001)
    model.init(plot_init=True)
    model.solve()
    # answer should be 1 everywhere
    plot(model.get_sed_height(),interactive=True)
    print model.get_sed_height_array()


