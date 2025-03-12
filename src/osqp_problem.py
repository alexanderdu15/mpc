import numpy as np
import pinocchio as pin
from scipy.sparse import bmat, csc_matrix, triu
import osqp

class OsqpProblem:
    def __init__(self, robot, config):

        self.N = config['N']  # num knotpoints
        self.dt = config['dt']  # time step
        self.dQ_cost = config['joint_vel_cost']
        self.R_cost = config['control_cost']
        self.QN_cost = config['terminal_cost']
        self.mu = config['mu']
        # properties
        self.nq = robot.nq  # j_pos
        self.nv = robot.nv  # j_vel
        self.nx = self.nq + self.nv  # state size
        self.nu = len(robot.model.joints) - 1  # num controls
        self.traj_len = (self.nx+self.nu)*self.N - self.nu  # length of trajectory

        # problem data
        self.P = self.initialize_P() # cost matrix
        self.Pdata = np.zeros(self.P.nnz)
        self.g = np.zeros(self.traj_len) # cost vector
        self.A = self.initialize_A()  # linearized dynamics
        self.Adata = np.zeros(self.A.nnz)
        self.l = np.zeros(self.N*self.nx)  # dynamics constraint vector

        self.osqp = osqp.OSQP()
        osqp_settings = {'verbose': False}
        self.osqp.setup(P=self.P, q=self.g, A=self.A, l=self.l, u=self.l, **osqp_settings) # lower = upper constraint

        # temp variables
        self.A_k = np.vstack([-1.0 * np.eye(self.nx), np.vstack([np.hstack([np.eye(self.nq), self.dt * np.eye(self.nq)]), np.ones((self.nq, 2*self.nq))])])
        self.B_k = np.vstack([np.zeros((self.nq, self.nq)), np.zeros((self.nq, self.nq))])
        self.cx_k = np.zeros(self.nx)

    # cost matrix
    def initialize_P(self):
        block = np.eye(self.nx + self.nu)
        block[:self.nq, :self.nq] = np.ones((self.nq, self.nq))
        bd = np.kron(np.eye(self.N), block)[:-self.nu, :-self.nu]
        return csc_matrix(triu(bd), shape=(self.traj_len, self.traj_len))
    
    def initialize_A(self):
        blocks = [[np.ones((self.nx,self.nx))] + [None] * (2*self.N)]
        for i in range(self.N-1):
            row = []
            # Add initial zeros if needed
            for j in range(2*i):
                row.append(None)  # None is interpreted as zero block
            # Add A, B, I
            row.extend([np.ones((self.nx,self.nx)), 2 * np.ones((self.nx,self.nu)), -1 * np.ones((self.nx,self.nx))])
            # Pad remaining with zeros
            while len(row) < 2*self.N + 1:
                row.append(None)
            blocks.append(row)

        return bmat(blocks, format='csc')
    
    def compute_dynamics_jacobians(self, q, v, u):
        d_dq, d_dv, d_du = pin.computeABADerivatives(self.model, self.data, q, v, u)
        self.A_k[self.nx + self.nq:, :self.nq] = d_dq * self.dt
        self.A_k[self.nx + self.nq:, self.nq:2*self.nq] = d_dv * self.dt + np.eye(self.nv)
        self.B_k[self.nq:, :] = d_du * self.dt
        
        a = self.data.ddq
        qnext = pin.integrate(self.model, q, v * self.dt)
        vnext = v + a * self.dt
        xnext = np.hstack([qnext, vnext])
        xcur = np.hstack([q,v])
        self.cx_k = xnext - self.A_k[self.nx:] @ xcur - self.B_k @ u

    def update_constraints(self, xu, xs):
        # Fast update of the existing CSC matrix
        self.l[:self.nx] = -1 * xs  # negative because top left is negative identity
        Aind = 0
        for k in range(self.N-1):
            xu_stride = (self.nx + self.nu)
            qcur = xu[k*xu_stride : k*xu_stride + self.nq]
            vcur = xu[k*xu_stride + self.nq : k*xu_stride + self.nx]
            ucur = xu[k*xu_stride + self.nx : (k+1)*xu_stride]
            
            self.compute_dynamics_jacobians(qcur, vcur, ucur)
            
            self.Adata[Aind:Aind+self.nx*self.nx*2]=self.A_k.T.reshape(-1)
            Aind += self.nx*self.nx*2
            self.Adata[Aind:Aind+self.nx*self.nu]=self.B_k.T.reshape(-1)
            Aind += self.nx*self.nu

            self.l[(k+1)*self.nx:(k+2)*self.nx] = -1.0 * self.cx_k
        self.Adata[Aind:] = -1.0 * np.eye(self.nx).reshape(-1)

    def update_cost(self, XU, eepos_g):
        Pind = 0
        for k in range(self.N):
            if k < self.N-1:
                XU_k = XU[k*(self.nx + self.nu) : (k+1)*(self.nx + self.nu)]
            else:
                XU_k = XU[k*(self.nx + self.nu) : (k+1)*(self.nx + self.nu)-self.nu]
                
            eepos = self.robot.forward_kinematics(XU_k[:self.nq])
            eepos_err = np.array(eepos.T) - eepos_g[k*3:(k+1)*3]
            deepos = self.robot.compute_ee_jacobian(XU_k[:self.nq])

            joint_err = eepos_err @ deepos

            g_start = k*(self.nx + self.nu)
            self.g[g_start : g_start + self.nx] = np.concatenate([
                joint_err.T,
                self.dQ_cost * XU_k[self.nq:self.nx]
            ])

            phessian = np.outer(joint_err, joint_err)
            pos_costs = phessian[np.tril_indices_from(phessian)]
            self.Pdata[Pind:Pind+len(pos_costs)] = pos_costs
            Pind += len(pos_costs)
            self.Pdata[Pind:Pind+self.nv] = np.full(self.nv, self.dQ_cost)
            Pind+=self.nv
            if k < self.N-1:
                self.Pdata[Pind:Pind+self.nu] = np.full(self.nu, self.R_cost)
                Pind+=self.nu
                self.g[g_start + self.nx : g_start + self.nx + self.nu] = self.R_cost * XU_k[self.nx:self.nx+self.nu].reshape(-1)

    def setup_qp(self, xu, xs, eepos_g):
        self.update_constraints(xu, xs)
        self.update_cost(xu, eepos_g)
        self.osqp.update(Px=self.Pdata)
        self.osqp.update(Ax=self.Adata)
        self.osqp.update(q=self.g, l=self.l, u=self.l)
        
    def solve(self):
        return self.osqp.solve()