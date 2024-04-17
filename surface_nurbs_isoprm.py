import surface_geom_SEM as sgs
import lagrange_der as lagd
import numpy as  np
from geomdl import exchange

ex = np.array((1, 0, 0), dtype=np.int32)
ey = np.array((0, 1, 0), dtype=np.int32)
ez = np.array((0, 0, 1), dtype=np.int32)       

class SurfaceGeneratedSEM(object):
    '''
    
    
    
    '''
    def __init__(self, nurbs_parent_surface, lobatto_pw, node_1_ub,\
                  node_1_vb, node_3_ub, node_3_vb):
        
        self.parent_nurbs_surf = nurbs_parent_surface
        self.lobatto_pw = lobatto_pw
        self.node_1_u = node_1_ub
        self.node_1_v = node_1_vb
        self.node_3_u = node_3_ub
        self.node_3_v = node_3_vb
        self.thk = nurbs_parent_surface.thk 
    
    #     The more Pythonic way to deal with attributes is using decorators.
    #     However, as I won't change the attributes outside of the class I 
    #     do not use it to make the class simpler.
    #     self._parent_nurbs_surf = nurbs_surface
    #     self._lobatto_pw = lobatto_pw
    #     self._node_1_u = node_1_ub
    #     self._node_1_v = node_1_vb
    #     self._node_3_u = node_3_ub
    #     self._node_3_v = node_3_vb
        
    # @property
    # def parent_nurbs_surf(self):
    #     return  self._parent_nurbs_surf 
    # @property
    # def lobatto_pw(self):
    #     return self._lobatto_pw
    # @property
    # def node_1_u(self):
    #     return self._node_1_u
    # @property
    # def node_1_v(self):
    #     return self._node_1_v
    # @property
    # def node_3_u(self):
    #     return self._node_3_u
    # @property
    # def node_3_v(self):
    #     return self._node_3_v
    
    # @parent_nurbs_surf.setter
    # def parent_nurbs_surf(self, value):
    #     self._parent_nurbs_surf = value
    # @node_1_u.setter
    # def node_1_u(self, value):
    #     self._node_1_u = value
    # @node_1_v.setter
    # def node_1_v(self, value):
    #     self._node_1_v = value   
    # @node_3_u.setter
    # def node_3_u(self, value):
    #     self._node_3_u = value
    # @node_3_v.setter
    # def node_3_v(self, value):
    #     self._node_3_v = value
    
    def xi_to_uv(self, xi1, xi2, node_1_u, node_1_v, node_3_u, node_3_v):
        '''This function is the mapping form xi1-xi2 which
        are parametric space of SEM to u-v space which are
        parametric space of the IGA
        -output:
        u and v according to the input xi1 and xi2 and two oppsite
        corners of the rectangle.
        '''
        if xi1<-1 and xi1>1 and xi2<-1 and xi2>1:
            raise ValueError('-1<=xi1<=+1, -1<=xi2<=+1 must be followed')
        else:
            node_2_u = node_3_u 
            u = 1/2*((1-xi1)*node_1_u + (1+xi1)*node_2_u)
            v = 1/2*((1-xi2)*node_1_v + (1+xi2)*node_3_v)
        return u, v
    
    def physical_crd_xi(self, xi1, xi2, node_1_u, node_1_v,\
                    node_3_u, node_3_v, t=0):
        u = self.xi_to_uv(xi1, xi2, node_1_u, node_1_v, node_3_u, node_3_v)[0]
        v = self.xi_to_uv(xi1, xi2, node_1_u, node_1_v, node_3_u, node_3_v)[1]
        physical_coor_var = self.parent_nurbs_surf.physical_crd(u, v, t)
        x = physical_coor_var[0]
        y = physical_coor_var[1]
        z = physical_coor_var[2]
        return x, y, z
        
        
    def nodes_physical_coordinate(self):
        '''This function generate the physical coordinate of the 
        lobatto points.
        -Output:
        A nxnx3 numpy array, in that n is the number of lobatto points
        '''
        dim = self.lobatto_pw.shape[0]
        nodes_coor_mtx = np.zeros((dim, dim, 3))
        for i in range(dim):
            xi2 = self.lobatto_pw[i, 0]
            for j in range(dim):
                xi1 = self.lobatto_pw[j, 0]
                physical_coor_local = self.physical_crd_xi(xi1, xi2,\
                                     self.node_1_u, self.node_1_v,\
                                     self.node_3_u, self.node_3_v)
                nodes_coor_mtx[i, j, 0] = physical_coor_local[0]
                nodes_coor_mtx[i, j, 1] = physical_coor_local[1]
                nodes_coor_mtx[i, j, 2] = physical_coor_local[2]         
        return nodes_coor_mtx

   
    def coorsys_director_tanvec_allnodes(self):
        '''
        In this method, the nodal coordinate system at each node
        specified by the Lagrangian functions at that point is calculated.
        -Output:
        A nxnx5x3 np.array cotains the coordinate system and tangent vectors. 
        
        '''
        nodes_physical_coor = self.nodes_physical_coordinate()
        dim = self.lobatto_pw.shape[0]
        coor_tan_mtx = np.zeros((dim, dim, 5, 3))
       
        for i_main in range(dim):
            xi2 = self.lobatto_pw[i_main, 0]
            lag_xi2 = lagd.lagfunc(self.lobatto_pw, xi2)
            der_lag_dxi2 = lagd.der_lagfunc_dxi(self.lobatto_pw, xi2)
            for j_main in range(dim):
                xi1 = self.lobatto_pw[j_main, 0]
                lag_xi1 = lagd.lagfunc(self.lobatto_pw, xi1)
                der_lag_dxi1 = lagd.der_lagfunc_dxi(self.lobatto_pw, xi1)
                g1 = np.zeros(3)
                g2 = np.zeros(3)
                for i in range(dim):
                    for j in range(dim):
                        g1 = g1 + np.array([der_lag_dxi1[j] * lag_xi2[i] * nodes_physical_coor[i, j, 0],\
                            der_lag_dxi1[j] * lag_xi2[i] * nodes_physical_coor[i, j, 1],\
                                der_lag_dxi1[j] * lag_xi2[i] * nodes_physical_coor[i, j, 2]])
                        g2 = g2 + np.array([lag_xi1[j] * der_lag_dxi2[i] * nodes_physical_coor[i, j, 0],\
                        lag_xi1[j] * der_lag_dxi2[i] * nodes_physical_coor[i, j, 1],\
                        lag_xi1[j] * der_lag_dxi2[i] * nodes_physical_coor[i, j, 2]])
                        
                vdir_not_unit = np.cross(g1, g2)
                vdir_unit = vdir_not_unit / np.linalg.norm(vdir_not_unit)
                
                if abs(abs(vdir_unit[1])-1) > 0.00001 :
                    v1 = (np.cross(ey, vdir_unit))
                    v1_unit= v1 / np.linalg.norm(v1)
                    v2_unit = np.cross(vdir_unit, v1_unit)
                else:
                    if vdir_unit[1] > 0:
                        v1_unit = np.array((1, 0, 0))
                        v2_unit = np.array((0, 0, -1))
                        vdir_unit = np.array((0, 1, 0))
                    else:
                        v1_unit = np.array((1, 0, 0))
                        v2_unit = np.array((0, 0, 1))
                        vdir_unit = np.array((0, 1, 0))
                coor_tan_mtx[i_main, j_main, 0] = v1_unit
                coor_tan_mtx[i_main, j_main, 1] = v2_unit
                coor_tan_mtx[i_main, j_main, 2] = vdir_unit
                coor_tan_mtx[i_main, j_main, 3] = g1
                coor_tan_mtx[i_main, j_main, 4] = g2
        return coor_tan_mtx
    

    
    def jacobian_mtx(self, coorsys_tanvec_mtx, row_num, col_num,\
        xi3, lag_xi1, lag_xi2, der_lag_dxi1, der_lag_dxi2):
        '''
        In this method, the Jacobian matrix is calculated. Unlike our developed
        method, in isoparametric elements, we need the director vectors at
        all nodes to iterpolate it, which is necessary for calculation of the
        derivatives of the director at a specific point. This is the reason that
        the coorsys_tanvec_matx which is calculated from 
        "coorsys_director_tanvec_allnodes" method is imported.
        -Output:
        A 3x3 np.array which is Jacobian matrix.
        
        '''
        vdir_unit = coorsys_tanvec_mtx[row_num, col_num, 2]
        g1 = coorsys_tanvec_mtx[row_num, col_num, 3]
        g2 = coorsys_tanvec_mtx[row_num, col_num, 4]
        dim = self.lobatto_pw.shape[0]
        dvdir_unit_dxi1 = np.zeros(3)
        dvdir_unit_dxi2 = np.zeros(3)
        # with open('director_unit_isoprm.dat', 'a') as du:
        #     np.savetxt(du, coorsys_tanvec_mtx[row_num, col_num, 2])
        #     # du.write('\n')
        for i in range(dim):
            for j in range(dim):
                dvdir_unit_dxi1 = dvdir_unit_dxi1 + \
                    np.array([der_lag_dxi1[j] * lag_xi2[i] * coorsys_tanvec_mtx[i, j, 2, 0],\
                            der_lag_dxi1[j] * lag_xi2[i] * coorsys_tanvec_mtx[i, j, 2, 1],\
                                der_lag_dxi1[j] * lag_xi2[i] * coorsys_tanvec_mtx[i, j, 2, 2]])
                dvdir_unit_dxi2 = dvdir_unit_dxi2 +\
                    np.array([lag_xi1[j] * der_lag_dxi2[i] * coorsys_tanvec_mtx[i, j, 2, 0],\
                        lag_xi1[j] * der_lag_dxi2[i] * coorsys_tanvec_mtx[i, j, 2, 1],\
                        lag_xi1[j] * der_lag_dxi2[i] * coorsys_tanvec_mtx[i, j, 2, 2]])
        dx_dxi1 = g1[0] + xi3/2 * self.thk * dvdir_unit_dxi1[0]
        dy_dxi1 = g1[1] + xi3/2 * self.thk * dvdir_unit_dxi1[1]
        dz_dxi1 = g1[2] + xi3/2 * self.thk * dvdir_unit_dxi1[2]
                    
        dx_dxi2 = g2[0] + xi3/2 * self.thk * dvdir_unit_dxi2[0]
        dy_dxi2 = g2[1] + xi3/2 * self.thk * dvdir_unit_dxi2[1]
        dz_dxi2 = g2[2] + xi3/2 * self.thk * dvdir_unit_dxi2[2]
            
        dx_dxi3 = 1/2 * self.thk * vdir_unit[0]
        dy_dxi3 = 1/2 * self.thk * vdir_unit[1]
        dz_dxi3 = 1/2 * self.thk * vdir_unit[2]
        
        jac = np.array(((dx_dxi1, dy_dxi1, dz_dxi1),(dx_dxi2, dy_dxi2, dz_dxi2),\
               (dx_dxi3, dy_dxi3, dz_dxi3)))    
        return jac 
    
    
    def curvature_mtx(self, coorsys_tanvec_mtx, row_num, col_num,\
         lag_xi1, lag_xi2, der_lag_dxi1, der_lag_dxi2):
        '''
        To examine more, the first part of this method is exaclty copied from the jacobian_mtx.
        Then the derivatives of the unit director vector and the tangent vector
        is used to calculate the curvature tensor and Gauss curvature. Fro example:
        "Models and Finite Elements for Thin-Walled Structures" Manfred Bischoff et al.
        Eq. 21 and 22.
        -Output:
        Gauss curvature
        '''
        g1 = coorsys_tanvec_mtx[row_num, col_num, 3]
        g2 = coorsys_tanvec_mtx[row_num, col_num, 4]
        dim = self.lobatto_pw.shape[0]
        dvdir_unit_dxi1 = np.zeros(3)
        dvdir_unit_dxi2 = np.zeros(3)
        # with open('director_unit_isoprm.dat', 'a') as du:
        #     np.savetxt(du, coorsys_tanvec_mtx[row_num, col_num, 2])
        #     # du.write('\n')
        for i in range(dim):
            for j in range(dim):
                dvdir_unit_dxi1 = dvdir_unit_dxi1 + \
                    np.array([der_lag_dxi1[j] * lag_xi2[i] * coorsys_tanvec_mtx[i, j, 2, 0],\
                            der_lag_dxi1[j] * lag_xi2[i] * coorsys_tanvec_mtx[i, j, 2, 1],\
                                der_lag_dxi1[j] * lag_xi2[i] * coorsys_tanvec_mtx[i, j, 2, 2]])
                dvdir_unit_dxi2 = dvdir_unit_dxi2 +\
                    np.array([lag_xi1[j] * der_lag_dxi2[i] * coorsys_tanvec_mtx[i, j, 2, 0],\
                        lag_xi1[j] * der_lag_dxi2[i] * coorsys_tanvec_mtx[i, j, 2, 1],\
                        lag_xi1[j] * der_lag_dxi2[i] * coorsys_tanvec_mtx[i, j, 2, 2]])
        
        crv_tnsr = 1/2 * np.array([[  2*np.dot(dvdir_unit_dxi1, g1),\
                                    np.dot(dvdir_unit_dxi1, g2) + np.dot(dvdir_unit_dxi2, g1)],\
                                    [np.dot(dvdir_unit_dxi1, g2) + np.dot(dvdir_unit_dxi2, g1),\
                                        2*np.dot(dvdir_unit_dxi2, g2)]])
        
        metric_tnsr = np.array([[np.dot(g1, g1), np.dot(g1, g2)],\
                                [np.dot(g1, g2), np.dot(g2, g2)]])
        
        gauss_crv = -np.linalg.det(crv_tnsr) / np.linalg.det(metric_tnsr)
        
        return  gauss_crv
        
    def curvature_mtx_exact(self, coorsys_tanvec_mtx, row_num, col_num,\
         lag_xi1, lag_xi2, der_lag_dxi1, der_lag_dxi2, der2_lag_dxi1, der2_lag_dxi2):
        '''
        In crvature_mtx function, as it can be seen, for calculation of the derivatives
        of the unit director vector, the CONVENTIONAL method of derivative of interpolated director fuction or
        vdir_unit = N_I vdir_unit_I which is
        d(vdir_unit)/dxi = d(N_I) vdir_unit_I is used. However, this is not a precise method and the order of d(vdir)/dxi is 
        not p-1. But, We know that if the order of interpolating functions is p, the order of derivate of the director
        governs by d(vdir_unit)/dxi = d(g1xg2/norm(v)). So, in curvature_matrix_exact function only the location vector
        and geometry are defined by x=N_I x_I and then everything calculated by direct use of the isoparametric definition.
        So, for example:
        d^2(vdir_unit)/dxi1^2 = d^2(N_I)/dxi^2 x_I.
        The curvature fomulation can be found in:
        "Models and Finite Elements for Thin-Walled Structures" Manfred Bischoff et al.
        Eq. 21 and 22.
        -Output:
        Gauss curvature
        '''
        g1 = coorsys_tanvec_mtx[row_num, col_num, 3]
        g2 = coorsys_tanvec_mtx[row_num, col_num, 4]
        dim = self.lobatto_pw.shape[0]
        nodes_phys_coor = self.nodes_physical_coordinate()
        vdir_notunit = np.cross(g1, g2)
        norm_vdir_notunit = np.linalg.norm(vdir_notunit)
        dg1_dxi1 = np.zeros(3)
        dg1_dxi2 = np.zeros(3)
        dg2_dxi1 = np.zeros(3) # = dg1_dxi2
        dg2_dxi2 = np.zeros(3)
        
        dvdir_notunit_dxi1 = np.zeros(3)
        dvdir_notunit_dxi2 = np.zeros(3)
        
        for i in range(dim):
            for j in range(dim):
                dg1_dxi1 = dg1_dxi1 + \
                    np.array([der2_lag_dxi1[j] * lag_xi2[i] * nodes_phys_coor[i, j, 0],\
                            der2_lag_dxi1[j] * lag_xi2[i] * nodes_phys_coor[i, j, 1],\
                                der2_lag_dxi1[j] * lag_xi2[i] * nodes_phys_coor[i, j, 2]])
                dg2_dxi2 = dg2_dxi2 +\
                    np.array([lag_xi1[j] * der2_lag_dxi2[i] * nodes_phys_coor[i, j, 0],\
                        lag_xi1[j] * der2_lag_dxi2[i] * nodes_phys_coor[i, j, 1],\
                        lag_xi1[j] * der2_lag_dxi2[i] * nodes_phys_coor[i, j, 2]])
                dg1_dxi2 = dg1_dxi2 + \
                    np.array([der_lag_dxi1[j] * der_lag_dxi2[i] * nodes_phys_coor[i, j, 0],\
                            der_lag_dxi1[j] * der_lag_dxi2[i] * nodes_phys_coor[i, j, 1],\
                                der_lag_dxi1[j] * der_lag_dxi2[i] * nodes_phys_coor[i, j, 2]])
        dg2_dxi1 = dg1_dxi2
        
        dvdir_notunit_dxi1 = np.cross(dg1_dxi1, g2) + np.cross(g1, dg2_dxi1)
        dnorm_vdir_notunit_dxi1 = np.dot(dvdir_notunit_dxi1, vdir_notunit)/ \
                                     (norm_vdir_notunit)
        dvdir_unit_dxi1 = (dvdir_notunit_dxi1*norm_vdir_notunit -\
                             vdir_notunit*dnorm_vdir_notunit_dxi1)/norm_vdir_notunit**2
        
        dvdir_notunit_dxi2 = np.cross(dg1_dxi2, g2) + np.cross(g1, dg2_dxi2)
        dnorm_vdir_notunit_dxi2 = np.dot(dvdir_notunit_dxi2, vdir_notunit)/ \
                                     (norm_vdir_notunit)
        dvdir_unit_dxi2 = (dvdir_notunit_dxi2*norm_vdir_notunit -\
                             vdir_notunit*dnorm_vdir_notunit_dxi2)/norm_vdir_notunit**2
        
        crv_tnsr = 1/2 * np.array([[  2*np.dot(dvdir_unit_dxi1, g1),\
                                   np.dot(dvdir_unit_dxi1, g2) + np.dot(dvdir_unit_dxi2, g1)],\
                                  [np.dot(dvdir_unit_dxi1, g2) + np.dot(dvdir_unit_dxi2, g1),\
                                      2*np.dot(dvdir_unit_dxi2, g2)]])
        metric_tnsr = np.array([[np.dot(g1, g1), np.dot(g1, g2)],\
                                [np.dot(g1, g2), np.dot(g2, g2)]])
        gauss_crv = -np.linalg.det(crv_tnsr) / np.linalg.det(metric_tnsr)
        
        return gauss_crv
        
                
        # dvdir_unit_dxi1 = np.zeros(3)
        # dvdir_unit_dxi2 = np.zeros(3)
        # # with open('director_unit_isoprm.dat', 'a') as du:
        # #     np.savetxt(du, coorsys_tanvec_mtx[row_num, col_num, 2])
        # #     # du.write('\n')
        # for i in range(dim):
        #     for j in range(dim):
        #         dvdir_unit_dxi1 = dvdir_unit_dxi1 + \
        #             np.array([der_lag_dxi1[j] * lag_xi2[i] * coorsys_tanvec_mtx[i, j, 2, 0],\
        #                     der_lag_dxi1[j] * lag_xi2[i] * coorsys_tanvec_mtx[i, j, 2, 1],\
        #                         der_lag_dxi1[j] * lag_xi2[i] * coorsys_tanvec_mtx[i, j, 2, 2]])
        #         dvdir_unit_dxi2 = dvdir_unit_dxi2 +\
        #             np.array([lag_xi1[j] * der_lag_dxi2[i] * coorsys_tanvec_mtx[i, j, 2, 0],\
        #                 lag_xi1[j] * der_lag_dxi2[i] * coorsys_tanvec_mtx[i, j, 2, 1],\
        #                 lag_xi1[j] * der_lag_dxi2[i] * coorsys_tanvec_mtx[i, j, 2, 2]])
        
        # crv_tnsr = 1/2 * np.array([[  2*np.dot(dvdir_unit_dxi1, g1),\
        #                             np.dot(dvdir_unit_dxi1, g2) + np.dot(dvdir_unit_dxi2, g1)],\
        #                             [np.dot(dvdir_unit_dxi1, g2) + np.dot(dvdir_unit_dxi2, g1),\
        #                                 2*np.dot(dvdir_unit_dxi2, g2)]])
        
        # metric_tnsr = np.array([[np.dot(g1, g1), np.dot(g1, g2)],\
        #                         [np.dot(g1, g2), np.dot(g2, g2)]])
        
        # gauss_crv = -np.linalg.det(crv_tnsr) / np.linalg.det(metric_tnsr)
        
        return  gauss_crv
    
    
    def area(self):
        '''This function is only for the test of the 
        validity of derivatives of mapping and jacobian'''
        node_coor  = self.nodes_physical_coordinate()
        coorsys_tanvec_mtx = self.coorsys_director_tanvec_allnodes()
        dim = self.lobatto_pw.shape[0]
        local_area = 0
        for i in range(dim):
            xi2 = self.lobatto_pw[i, 0]
            w2 = self.lobatto_pw[i, 1]
            lag_xi2 = lagd.lagfunc(self.lobatto_pw, xi2)
            der_lag_dxi2 = lagd.der_lagfunc_dxi(self.lobatto_pw, xi2)   
            for j in range(dim):
                xi1 = self.lobatto_pw[j, 0]
                w1 = self.lobatto_pw[j, 1]
                lag_xi1 = lagd.lagfunc(self.lobatto_pw, xi1)
                der_lag_dxi1 = lagd.der_lagfunc_dxi(self.lobatto_pw, xi1)
                jac_mtx = self.jacobian_mtx( coorsys_tanvec_mtx, i, j, 0, lag_xi1, lag_xi2, der_lag_dxi1, der_lag_dxi2)
                vector_1 = jac_mtx[0,:]
                vector_2 = jac_mtx[1,:]
                local_area = local_area +\
                    np.linalg.norm((np.cross(vector_1,vector_2)))* w1 * w2
                                
        return  local_area 
    
    
    def volume(self, number_gauss_point):
        '''This function is only for the test of the 
        validity of derivatives of mapping and jacobian'''
        node_coor  = self.nodes_physical_coordinate()
        coorsys_tanvec_mtx = self.coorsys_director_tanvec_allnodes()
        dim = self.lobatto_pw.shape[0]
        gauss_pw_nparray = np.array(lagd.gauss_pw[number_gauss_point-2])
        dim_gauss = gauss_pw_nparray.shape[0] 
        local_volume = 0
        for i in range(dim):
            xi2 = self.lobatto_pw[i, 0]
            w2 = lobatto_pw[i, 1]
            lag_xi2 = lagd.lagfunc(self.lobatto_pw, xi2)
            der_lag_dxi2 = lagd.der_lagfunc_dxi(self.lobatto_pw, xi2)   
            for j in range(dim):
                xi1 = self.lobatto_pw[j, 0]
                w1 = lobatto_pw[j, 1]
                lag_xi1 = lagd.lagfunc(self.lobatto_pw, xi1)
                for k in range(dim_gauss):
                    xi3 = gauss_pw_nparray[k, 0]
                    w3 =gauss_pw_nparray[k, 1]
                    der_lag_dxi1 = lagd.der_lagfunc_dxi(self.lobatto_pw, xi1)
                    jac_mtx = self.jacobian_mtx( coorsys_tanvec_mtx, i, j, xi3,\
                        lag_xi1, lag_xi2, der_lag_dxi1, der_lag_dxi2)
                    
                    local_volume = local_volume +\
                        np.linalg.det(jac_mtx)* w1 * w2 * w3
                                   
        return  local_volume 
        
        
        
        

            
            
if __name__=="__main__":
    thk = 0.1
    data = exchange.import_json("sphere_clean_notrim.json") #curved_beam_lineload_2.json double-curved-free-form.json
    nurbs_surface = sgs.SurfaceGeo(data, 0, thk)
    
    lobatto_pw_all = lagd.lbto_pw("node_weight_all.dat")
    i_main = 12
    if i_main == 1:
        lobatto_pw = lobatto_pw_all[1:3,:]
    else:  
        index = np.argwhere(lobatto_pw_all==i_main)
        lobatto_pw = lobatto_pw_all[index[0, 0] + 1:\
                            index[0, 0] + (i_main+1) +1, :]
    dim = lobatto_pw.shape[0]
    node_1_ub = 0
    node_1_vb = 0
    node_3_ub = 1
    node_3_vb = 1
    surf_lag = SurfaceGeneratedSEM(nurbs_surface, lobatto_pw, node_1_ub,\
                  node_1_vb, node_3_ub, node_3_vb)
    pp = surf_lag.physical_crd_xi(0.5, 0.5, 0, 0, 1, 1)
    print(pp)
    print(surf_lag.area())
    nodal_coor = surf_lag.nodes_physical_coordinate()
    with open("coor_lobp.dat", 'w') as cl:
        pass
    for i in range(i_main+1):
        for j in range(i_main+1):
            with open("coor_lobp.dat", 'a') as cl:
                np.savetxt(cl, nodal_coor[i, j])
                
                
                
    with open("coor_lobp_left_line.dat", 'w') as cl:
        pass
    coor_lbp_left_line = np.zeros((dim, 4))
    nodal_coor = surf_lag.nodes_physical_coordinate()
    j = 0
    for i in range(dim-1, -1, -1): #(14, 13, 12,..., 1, 0)
        coor_lbp_left_line[j, :] = [j + 1, nodal_coor[0, i, 0], nodal_coor[0, i, 1], nodal_coor[0, i, 2]]
        j += 1
 
    with open("coor_lobp_left_line.dat", 'a') as cl:
            np.savetxt(cl, coor_lbp_left_line)
                
                
    with open("coor_cp.dat",'w') as cp:
        pass
    cp_number_u = nurbs_surface.cp_size_u
    cp_number_v = nurbs_surface.cp_size_v
    cp_coor = nurbs_surface.cpw_3d
    
    for i in range(cp_number_v): #For generating a surface in Mathematica
            with open('coor_cp.dat','a') as cp:
                np.savetxt(cp, cp_coor[i], newline = '\n')
    with open("coor_cp_left_line.dat","w"):
        pass
    with open("coor_cp_left_line.dat", "a") as cp: # For generating the left curve of the surface.
        np.savetxt(cp, cp_coor[0, :])
        
                
    coorsys_tanvec_mtx = surf_lag.coorsys_director_tanvec_allnodes()
    dim = lobatto_pw.shape[0]
    xi3 = 0
    with open("jacobian_isoprm.dat", "w") as f:
        pass
    
    for i in range(dim):
            xi2 = surf_lag.lobatto_pw[i, 0]
            w2 = surf_lag.lobatto_pw[i, 1]
            lag_xi2 = lagd.lagfunc(surf_lag.lobatto_pw, xi2)
            der_lag_dxi2 = lagd.der_lagfunc_dxi(surf_lag.lobatto_pw, xi2)   
            for j in range(dim):
                xi1 = surf_lag.lobatto_pw[j, 0]
                w1 = surf_lag.lobatto_pw[j, 1]
                lag_xi1 = lagd.lagfunc(surf_lag.lobatto_pw, xi1)
                der_lag_dxi1 = lagd.der_lagfunc_dxi(surf_lag.lobatto_pw, xi1)
                jac_mtx = surf_lag.jacobian_mtx( coorsys_tanvec_mtx, i, j, xi3,\
                    lag_xi1, lag_xi2, der_lag_dxi1, der_lag_dxi2)
                with open ('jacobian_isoprm.dat', 'a') as jac_file:
                    np.savetxt(jac_file, jac_mtx)
                    jac_file.write("\n")
    print("End")
    
    
    
    # print(surf_lag.thk)
    #  print(surf_lag.nodes_physical_coordinate())
    # node_coor = surf_lag.nodes_physical_coordinate()
    # print(surf_lag.coorsys_director_tanvec_allnodes())
    # print(surf_lag.area())
    # print(surf_lag.volume(2))
    # print('End')
    
   
            
            
        
        
        
            
        
        
    
        
        