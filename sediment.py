from dolfin import *
import numpy

# Create mesh and define function space
mesh = UnitSquare(10, 10)
V = FunctionSpace(mesh, "Lagrange", 1)
n = FacetNormal(mesh)

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
#u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
#                alpha=alpha, beta=beta, t=0)
u0 = Expression('x[0]+t',t=0)
u0.t = 0

#sea_level = Expression('a', a=0.0)
#h0 = project(Expression('x[0]'),V)
#h1 = project(Expression('x[0]'),V)

def boundary(x, on_boundary):  # define the Dirichlet boundary
    return on_boundary

bc = DirichletBC(V, u0, boundary)

# initial guess of solution
u_1 = interpolate(u0, V)

dt = 1      # time step

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
alpha = Constant(1.0)

t = dt
a = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*alpha*dx - v*alpha*dx
L = (u_1 + dt*f)*v*dx
b = None
A = assemble(a)   # assemble only once, before the time stepping

u = Function(V)   # the unknown at a new time level
T = 10             # total simulation time


while t <= T:
    b = assemble(L)
    u0.t = t
    #bc.apply(A, b)
    solve(A, u.vector(), b)

    b = assemble(L, tensor=b)
    t += dt
    plot(u, interactive=True)
    u_1.assign(u)

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
