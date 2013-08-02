from dolfin import *
import numpy
import sys
import unittest
sys.path.insert(0,"../")
from sediment import *


class DiffusionTest(unittest.TestCase):
    def test_slope_slope_diffusion(self):        #slope topography, slope sediment, check diffusion
        #create a simple testcase
        model = SedimentModel()
        mesh = UnitSquare(10,10)
        model.set_mesh(mesh)
        init_cond = Expression('x[0]') # simple slope
        init_sed = Expression('x[0]') # this gives
        # total of above gives a slope of 0 to 2 (over the unit square)
        model.set_initial_conditions(init_cond,init_sed)
        model.set_end_time(10)
        model.set_diffusion_coeff(10)
        model.init()
        model.solve()
        # answer should be 1 everywhere
        answer = model.get_total_height_array()
        for i in answer:
            self.assert_(-1e-8 < (i - 1) < 1e-8)

    def test_flat_flat(self):        #flat topography(none), flat sediment(none), should have no diffusion
        #create a simple testcase
        model = SedimentModel()
        mesh = UnitSquare(10,10)
        model.set_mesh(mesh)
        init_cond = Expression('0') # simple slope
        init_sed = Expression('0') # this gives
        # total of above gives a slope of 0 to 0 (over the unit square)
        model.set_initial_conditions(init_cond,init_sed)
        model.set_end_time(10)
        model.set_diffusion_coeff(10)
        model.init()
        model.solve()
        # answer should be 1 everywhere
        #plot(model.get_total_height(),interactive=True)
        answer = model.get_total_height_array()
        for i in answer:
            self.assert_(-1e-10 < i < 1e-10)

    def test_slope_flat(self):        #sloped topography flat sediment, should have no diffusion #####
        #create a simple testcase
        model = SedimentModel()
        mesh = UnitSquare(10,10)
        model.set_mesh(mesh)
        init_cond = Expression('5*x[0]') # simple slope
        init_sed = Expression('0') # this gives
        model.set_initial_conditions(init_cond,init_sed)
        model.set_end_time(10)
        model.set_diffusion_coeff(10)
        model.init()
        model.solve()
        answer = model.get_total_height_array()
        #print answer
        expected = numpy.array([0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
        for i in range(0,110,11):
            row = answer[i:i+11]
            self.assert_(numpy.allclose(row,expected))

    def test_flat_slope(self):        #flat topography sloped sediment, should diffuse
        #create a simple testcase
        model = SedimentModel()
        mesh = UnitSquare(10,10)
        model.set_mesh(mesh)
        init_cond = Expression('0') # simple slope
        init_sed = Expression('x[0]') # this gives
        # total of above gives a slope of 0 to 0 (over the unit square)
        model.set_initial_conditions(init_cond,init_sed)
        model.set_end_time(10)
        model.set_diffusion_coeff(10)
        model.init()
        model.solve()
        # answer should be 1 everywhere
        #plot(model.get_total_height(),interactive=True)
        answer = model.get_total_height_array()
        for i in answer:
            self.assert_(-1e-10 < i - 0.5 < 1e-10)

    def test_flat_flat_no_diffusion1(self):        #flat topography flat sediment, no diffusion
        #create a simple testcase
        model = SedimentModel()
        mesh = UnitSquare(10,10)
        model.set_mesh(mesh)
        init_cond = Expression('1') # simple slope
        init_sed = Expression('1') # this gives
        # total of above gives a slope of 0 to 0 (over the unit square)
        model.set_initial_conditions(init_cond,init_sed)
        model.set_end_time(10)
        model.set_diffusion_coeff(0)
        model.init()
        model.solve()
        # answer should be 1 everywhere
        #plot(model.get_total_height(),interactive=True)
        answer = model.get_total_height_array()
        for i in answer:
            self.assert_(-1e-10 < i - 2 < 1e-10)

if __name__ == "__main__":

    unittest.main()
