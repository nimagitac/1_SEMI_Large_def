''' Explanations about some imported functions and classes
- NURBS.surface.derivative method provids a list. Each
row number shows the derivative order w. r. t. u and column
number shows the derivative order w. r. t. v
-NURBS.surface.normal provides the unit normal vector
'''
# geomdl.helpers.basis_function(degree, knot_vector, span, knot)
# span:the number of the known span starts from 0, knot: the knot coordinate
# there is maybe a problem in this function both here an in mathlab. If we provide 
# a knot coordinate (knot) that does not belog to the 'span' it provides the result 
# and not an error or an exception. Dresbasisfunk

import os
from geomdl import exchange
from geomdl import helpers
from geomdl import operations
from geomdl import NURBS
from geomdl import multi
from geomdl.visualization import VisMPL as vis
#from geomdl.visualization import VisVTK as vis
import numpy as np
import sys
import cProfile
import pstats
import io
from pstats import SortKey
import subprocess


def profile(func):
    '''This function is used for profiling the file
    It will be used as a decorator.
    output:
    A powershell output and process.profile file'''
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        value = func(*args, **kwargs)
        pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
        pr.dump_stats('process_surface.profile')
        return value
    return inner


class SurfaceGeo(NURBS.Surface):
    """The calss Surface contains the data attributes and mehtod to 
    work with NURBS surfaces. 
    Data attributs
        cpw : list of control points and weights
        cp_size_u : number of cps in u direction
        cp_size_v 
        knotvector_u : knot vector in u direction (same as parent class)
        knotvector_v 
        degree_u :degree of polynomial in u direction
        degree_v 
        thk: thickness
        cpw_3d ; the control points and weiths in a 3D matrix
                      i,j shows control point number and k=1,2,3,4
                      are related to x, y, z, w respectively
    Methods:
        make_cpw_3d : changes cpw to cpw_3d
        knot_find_span : find the knot span number of knot with the 
                      help of helpers module
        physical_crd : return the physical coordinate of u, v, t
        ders_uvt : return the matrix of derivatives of x, y ,z 
                   w. r. t. u, v, t (mapping gradient)
        jacobi : returns the jacobaina of mapping. """
        
    def __init__(self, surface_container_list, surface_number=0, thickness=1):
        '''Why inheritency does not work well for this class:
        If we assign NURBS.Surface as the parent class of Surface, it needs 
        to be initialized by calling NURBS.Surface 
        (either by super() or explicitly).While we import all of the attributes
        from a CAD file. Therefore, we should  either skip inheritency (ver2)
        or initialize the NURBS.Surface and assigning it to a varibale 
        (here geomdl_surace) and then equating
        the class dictionaries. Because in this way all the attributes are
        assigned to the child class no super function is needed really.
        Also, beside adding new attributes and methods, the namespace 
        also modified E. g. cpw->ctrlptsw .....
        '''
        # super().__init__()
        geomdl_surface = surface_container_list[surface_number]
        self.__dict__ = geomdl_surface.__dict__
        # self.cp = geomdl_surface.ctrlpts
        self.cp_size_u = geomdl_surface.ctrlpts_size_u 
        self.cp_size_v = geomdl_surface.ctrlpts_size_v
        self.knotvector_u = geomdl_surface.knotvector_u
        self.knotvector_v = geomdl_surface.knotvector_v
        self.degree_u = geomdl_surface.degree_u
        self.degree_v = geomdl_surface.degree_v
        self.thk = thickness
        self.cpw_3d = self.make_cpw_3d()       
        
        
    def make_cpw_3d (self):
        arr = np.zeros((self.cp_size_u, self.cp_size_v, 4))
        i = 0
        count = 0
        while i < self.cp_size_u:
            j = 0
            while j < self.cp_size_v:
                # arr[j, i , 0] = self.ctrlpts[j + count][0] # This  order and shape is more  usefult to be used in Mathematica file
                # arr[j, i , 1] = self.ctrlpts[j + count][1]
                # arr[j, i , 2] = self.ctrlpts[j + count][2]
                # arr[j, i , 3] = self.ctrlptsw[j + count][3]
                arr[i, j , 0] = self.ctrlpts[j + count][0]  # This is the original format
                arr[i, j , 1] = self.ctrlpts[j + count][1]
                arr[i, j , 2] = self.ctrlpts[j + count][2]
                arr[i, j , 3] = self.ctrlptsw[j + count][3]
                j += 1
            i += 1
            count = count + self.cp_size_v
        return arr 
           
                                                                 
    def knot_find_span (self, knot, direction='u'):
        if direction =='u':
            knt_span_num = helpers.find_span_binsearch(self.degree_u,\
                self.knotvector_u, self.cp_size_u, knot )
        elif direction == 'v':
            knt_span_num = helpers.find_span_binsearch(self.degree_v,\
                self.knotvector_v, self.cp_size_v, knot )
        else:
            raise ValueError(' The direction must be either "u" or "v" ')
        return knt_span_num
    
    # @profile  
    def director(self, u , v):
        normal_vector =  np.array(operations.normal(self,(u, v)))[1]
        director_unit =  normal_vector/\
                np.linalg.norm(normal_vector)
        return director_unit
    
    
    def physical_crd(self, u=0, v=0, t=0, *args):
        if u < 0 or u > 1 or v < 0 or v > 1 or t < -1 or t > 1:
             raise ValueError('0<=u<=+1, 0<=v<=+1 and -1<=t<=+1 must be followed')
        else: 
             vdir = self.director(u, v)
             local_der_zero = self.derivatives(u, v, order=0)
             x = local_der_zero [0][0][0] + t * self.thk/2 * vdir[0]
             y = local_der_zero [0][0][1] + t * self.thk/2 * vdir[1]
             z = local_der_zero [0][0][2] + t * self.thk/2 * vdir[2]
        return x, y, z
    
      
    def ders_uvt (self, t, second_surf_der):
        '''In order to calculete the derivatives of the x, y, and z
        we need to have the derivatives of the unit director vector. We use:
        c = g_1 x g_2   |c| = |g_1 x g_2 |
        d(c-unit)/du = d (c/|c|)/du
        d (c/|c|)/du = ( d(g_1xg_2)/du |c|-(g_1xg_2)d|c|/du) )/|c|^2
        |c|^2 = g_1xg_2 . c = det[g_1 g_2 c]
        2|c| ∂|c|/∂u =
        (det[(∂g_1)/∂u g_2 c] + det[g_1  (∂g_2)/∂u c] + det[g_1 g_2  ∂c/∂u])
        and 
         ∂c/∂u = dg1/du x g2 + g1 x dg2/du
        c or in the code, vdir_notunit, 
        is the director vector which is not the unit vector, g1 and g2 are 
        covariant base vectors that are tangent to the surface.
        the definition of self.derivative fucntion can be found in the manual
        of geomdl library
        '''
        if  t < -1 or t > 1:
             raise ValueError('-1<=t<=+1 must be followed')
        else:
            # vdir = self.director(u, v)
            # tang_uv_unit = operations.tangent(self, (u, v)) #unit vector       
            g1 = np.array(second_surf_der[1][0])
            g2 = np.array(second_surf_der[0][1])
            dg1_du = np.array((second_surf_der[2][0][0],\
                     second_surf_der[2][0][1], second_surf_der[2][0][2]))
            dg1_dv = np.array((second_surf_der[1][1][0],\
                     second_surf_der[1][1][1], second_surf_der[1][1][2]))
            dg2_dv = np.array((second_surf_der[0][2][0], \
                     second_surf_der[0][2][1], second_surf_der[0][2][2]))
            dg2_du = dg1_dv
            # vdir_notunit = np.array([g1[1]*g2[2]-g1[2]*g2[1],\
            #                       -g1[0]*g2[2]+g1[2]*g2[0],\
            #                        g1[0]*g2[1]-g1[1]*g2[0]])      
            vdir_notunit = np.cross(g1, g2)
            # dvdir_notunit_du = np.array([dg1_du[1]*g2[2]-dg1_du[2]*g2[1],\
            #                       -dg1_du[0]*g2[2]+dg1_du[2]*g2[0],\
            #                        dg1_du[0]*g2[1]-dg1_du[1]*g2[0]]) +\
            #                    np.array([g1[1]*dg2_du[2]-g1[2]*dg2_du[1],\
            #                       -g1[0]*dg2_du[2]+g1[2]*dg2_du[0],\
            #                        g1[0]*dg2_du[1]-g1[1]*dg2_du[0]])
            
            dvdir_notunit_du = np.cross(dg1_du, g2) + np.cross(g1, dg2_du)
            norm_vdir_notunit = np.linalg.norm(vdir_notunit)
            part_1 = np.array([[dg1_du[0], g2[0], vdir_notunit[0]],
                               [dg1_du[1], g2[1], vdir_notunit[1]],
                               [dg1_du[2], g2[2], vdir_notunit[2]]])
            part_2 = np.array([[g1[0], dg2_du[0], vdir_notunit[0]],
                               [g1[1], dg2_du[1], vdir_notunit[1]],
                               [g1[2], dg2_du[2], vdir_notunit[2]]])
            part_3 = np.array([[g1[0], g2[0], dvdir_notunit_du[0]],
                               [g1[1], g2[1], dvdir_notunit_du[1]],
                               [g1[2], g2[2], dvdir_notunit_du[2]]])
            dnorm_dvdir_notunit_du = (np.linalg.det(part_1) +\
                                      np.linalg.det(part_2) +\
                                      np.linalg.det(part_3))/\
                                     (2*norm_vdir_notunit)
            dvdir_unit_du = (dvdir_notunit_du*norm_vdir_notunit -\
                             vdir_notunit*dnorm_dvdir_notunit_du)/norm_vdir_notunit**2
            
            # dvdir_notunit_dv = np.array([dg1_dv[1]*g2[2]-dg1_dv[2]*g2[1],\
            #                       -dg1_dv[0]*g2[2]+dg1_dv[2]*g2[0],\
            #                        dg1_dv[0]*g2[1]-dg1_dv[1]*g2[0]]) +\
            #                    np.array([g1[1]*dg2_dv[2]-g1[2]*dg2_dv[1],\
            #                       -g1[0]*dg2_dv[2]+g1[2]*dg2_dv[0],\
            #                        g1[0]*dg2_dv[1]-g1[1]*dg2_dv[0]])
            dvdir_notunit_dv = np.cross(dg1_dv, g2) + np.cross(g1, dg2_dv)
            part_1 = np.array([[dg1_dv[0], g2[0], vdir_notunit[0]],
                               [dg1_dv[1], g2[1], vdir_notunit[1]],
                               [dg1_dv[2], g2[2], vdir_notunit[2]]])
            part_2 = np.array([[g1[0], dg2_dv[0], vdir_notunit[0]],
                               [g1[1], dg2_dv[1], vdir_notunit[1]],
                               [g1[2], dg2_dv[2], vdir_notunit[2]]])
            part_3 = np.array([[g1[0], g2[0], dvdir_notunit_dv[0]],
                               [g1[1], g2[1], dvdir_notunit_dv[1]],
                               [g1[2], g2[2], dvdir_notunit_dv[2]]])
            dnorm_dvdir_notunit_dv = (np.linalg.det(part_1) +\
                                      np.linalg.det(part_2) +\
                                      np.linalg.det(part_3))/\
                                     (2*norm_vdir_notunit)
            dvdir_unit_dv = (dvdir_notunit_dv*norm_vdir_notunit -\
                             vdir_notunit*dnorm_dvdir_notunit_dv)/ norm_vdir_notunit**2
            
            vdir_unit = vdir_notunit / np.linalg.norm(vdir_notunit)
            
            dx_du = g1[0] + t/2 * self.thk * dvdir_unit_du[0]
            dy_du = g1[1] + t/2 * self.thk * dvdir_unit_du[1]
            dz_du = g1[2] + t/2 * self.thk * dvdir_unit_du[2]
                        
            dx_dv = g2[0] + t/2 * self.thk * dvdir_unit_dv[0]
            dy_dv = g2[1] + t/2 * self.thk * dvdir_unit_dv[1]
            dz_dv = g2[2] + t/2 * self.thk * dvdir_unit_dv[2]
             
            dx_dt = 1/2 * self.thk * vdir_unit[0]
            dy_dt = 1/2 * self.thk * vdir_unit[1]
            dz_dt = 1/2 * self.thk * vdir_unit[2]
        
        ders = np.array(((dx_du, dy_du, dz_du),(dx_dv, dy_dv, dz_dv),\
               (dx_dt, dy_dt, dz_dt)))    
        return ders    
    
      
    def jacobi(self, t, second_surf_derivatives):
        if  t < -1 or t > 1:
            raise ValueError('0<=u<=+1, 0<=v<=+1 and -1<=t<=+1 must be followed')
        else: 
            print("Please notice: in parametric space: -1<= t =<+1 \n") 
            jcb = np.linalg.det(self.ders_uvt(t, second_surf_derivatives))
            return jcb
        
def lbto_pw(data_file_address):
    '''This function gets the data from
    Mathematica provided .dat file. The data is 
    lobatto (lbto) points and weights of numeriucal integration
    -output:
    a 2D np.array in that first column is the 
    spectral node coordinates and the second column 
    is the ascociated weights
    '''
    node_data = np.genfromtxt(data_file_address)
    return node_data


def visualization(data):
    surf_cont = multi.SurfaceContainer(data)
    surf_cont.sample_size = 30
    surf_cont.vis = vis.VisSurface(ctrlpts=False, trims=False)
    surf_cont.render()
    return



                  
if __name__ == '__main__':
    os.system('cls')
    u = 0.001
    v = 0.001       
    data = exchange.import_json("half-circle.json") #sphere_clean_2 generic_shell_kninsertion_sense_True square square_kninsertion generic_shell_kninsertion foursided_curved_kninsertion foursided_curved_kninsertion2  rectangle_kninsertion
    visualization(data)
    surfs = SurfaceGeo(data, 0, 1)

    print(surfs.knotvector_u)
    print(surfs.knotvector_v)
    
    print(surfs.degree_u)
    print(surfs.degree_v)
    
    p_1 = surfs.physical_crd(0., 0.)
    p_2 = surfs.physical_crd(1., 0.)
    p_3 = surfs.physical_crd(1., 1.)
    print("p_1:", p_1, "  p_2:", p_2, "  p_3:", p_3)
    t = 0
    second_surf_der = surfs.derivatives(0.01, 0.1, 2)
    print(surfs.jacobi( t, second_surf_der))
    second_surf_der2 = surfs.derivatives(0.99, 0.1, 2)
    print(surfs.jacobi( t, second_surf_der2))
    print(surfs.director(0., 0.5))
    print(surfs.director(1., 0.5))
    # data = exchange.import_json("pinched_shell.json") #  pinched_shell.json rectangle_cantilever square square_kninsertion generic_shell_kninsertion foursided_curved_kninsertion foursided_curved_kninsertion2  rectangle_kninsertion
    # surfs = SurfaceGeo(data, 0, 3)
    # lobatto_pw = lbto_pw("node_weight.dat")
    # len = lobatto_pw.shape[0]
    # for i in range(len):
    #         print(i, '\n')
    #         xi1 = lobatto_pw[i, 0]
    #         w1 = lobatto_pw[i, 1]
    #         for j in range(len):
    #             xi2 = lobatto_pw[j, 0]
    #             w2 = lobatto_pw[j, 1]
    #             u = 1/2 * (xi1 + 1)
    #             v = 1/2 * (xi2 + 1)
    #             print(surfs.director(u, v))
                
    # subprocess.call("C:\\Users\\azizi\Desktop\\DFG_Python\\.P394\\Scripts\\snakeviz.exe process_surface.profile ", \
    #                  shell=False)