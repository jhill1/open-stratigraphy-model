from dolfin import *
import numpy

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
        self.u0 = None

    def set_mesh(self,mesh):
        """Set which mesh to use and define function space"""
        self.mesh = mesh
        self.V = FunctionSpace(self.mesh, "Lagrange", 1)
        self.n = FacetNormal(self.mesh)

    def set_initial_conditions(self,expression):
        self.u0 = expression

    def init(self):
        """Initialise the solvers, etc, and setup the problem
        """
        #self.bc = DirichletBC(self.V, self.u0, self.boundary)

        # initial guess of solution
        self.u_1 = interpolate(self.u0, self.V)

        self.dt = 1      # time step

        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.f = Constant(0)
        self.alpha = Constant(0.001)

        self.t = self.dt
        self.a = self.u*self.v*dx + self.dt*inner(nabla_grad(self.u), nabla_grad(self.v))*self.alpha*dx
        self.L = (self.u_1 + self.dt*self.f)*self.v*dx
        self.b = None
        self.A = assemble(self.a)   # assemble only once, before the time stepping

        self.u = Function(self.V)   # the unknown at a new time level
        self.T = 10                 # total simulation time

    def solve(self):
        """Solve the problem"""
        t = 0
        while t <= self.T:
            self.b = assemble(self.L)
            #bc.apply(A, b)
            solve(self.A, self.u.vector(), self.b)

            self.b = assemble(self.L, tensor=self.b)
            t += self.dt
            #plot(u, interactive=True)
            self.u_1.assign(self.u)

    def sediment_height(self):
        return self.u.vector().array()

    def boundary(x, on_boundary):  # define the Dirichlet boundary
        return on_boundary

if __name__ == "__main__":
    
    #create a simple testcase
    model = SedimentModel()
    mesh = UnitSquare(10, 10)
    model.set_mesh(mesh)
    init_cond = Expression('x[0]') # simple slope
    model.set_initial_conditions(init_cond)
    model.init()
    model.solve()
    print model.sediment_height()


        

# left boundary marked as 0, right as 1
#class LeftBoundary(SubDomain):
#    def inside(self, x, on_boundary):
#        return x[0] < 0.5 + DOLFIN_EPS and on_boundary
#left_boundary = LeftBoundary()
#exterior_facet_domains = FacetFunction("uint", self.mesh)
#exterior_facet_domains.set_all(1)
#left_boundary.mark(exterior_facet_domains, 0)
#self.ds = Measure("ds")[exterior_facet_domains] 

alpha = 3; beta = 1.2

#sea_level = Expression('a', a=0.0)
#h0 = project(Expression('x[0]'),V)
#h1 = project(Expression('x[0]'),V)

def boundary(x, on_boundary):  # define the Dirichlet boundary
    return on_boundary




# Define variational problem
#v = TestFunction(V)
#A = Constant(1.0)
#alpha = Constant(0.00001)
#beta = Constant(0.01)
#Fl = Constant(1.0)
#h_td = 0.5*h0 + 0.5*h1
#dt = 1
#k = Constant(dt)
#F = v*(h0 - h1)*dx + inner(grad(v),grad(h0))*k*alpha*dx - v*sea_level*k*dx
#- v*alpha*inner(grad(h_td),n)*k*ds0
#F = v*Fl*(h0 - h1)*dx + inner(grad(v),grad(h_td))*k*Fl*alpha*dx - v*sea_level*k*dx

# Compute solution
#plot(h0, interactive=True)
#print h0.vector().array()

#t = 0.0
#T = 10.0
#while t < T:
#    solve(F == 0, h0)
#    h1.assign(h0)
#
#    plot(h0, interactive=True)
#    print h0.vector().array()
#
#    t = t+dt
 #   sea_level.a+=dt
 #   print sea_level.a
#
#    print t
