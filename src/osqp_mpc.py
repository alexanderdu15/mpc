import numpy as np
import pinocchio as pin
from utils import rk4
from osqp_problem import OsqpProblem
from sqp import SQP
from robot import Manipulator

class OsqpMpc:
    def __init__(self, urdf_filename, config):
        self.robot = Manipulator(urdf_filename)
        self.problem = OsqpProblem(self.robot, config)
        self.sqp = SQP(self.robot, self.problem)
        
        self.N = config['N']  # num knotpoints
        self.dt = config['dt']  # time step
        self.sim_dt = config['sim_dt']
        self.nq = self.robot.nq  # j_pos
        self.nv = self.robot.nv  # j_vel
        self.nx = self.nq + self.nv  # state size
        self.nu = len(self.robot.model.joints) - 1  # num controls
        
    def run_mpc(self, xstart, endpoints, num_steps=500): # (nx, (3, ), num_steps)
        
        N = self.N
        dt = self.dt
        
        nq = self.robot.nq  # j_pos
        nv = self.robot.nv  # j_vel
        nx = self.nx  # state size
        nu = self.nu  # num controls
                
        xcur = xstart
        endpoint_ind = 0
        endpoint = endpoints[endpoint_ind]
        eepos_goal = np.tile(endpoint, N).T
        
        xpath = []
        
        # Initialize trajectory
        XU = np.zeros(N*(nx+nu)-nu)
        XU = self.problem.sqp(xcur, eepos_goal, XU)
        
        for i in range(num_steps):
            # Check if we need to switch goals
            cur_eepos = self.robot.forward_kinematics(xcur[:nq])
            goaldist = np.linalg.norm(cur_eepos - eepos_goal[:3])
            
            if goaldist < 1e-1:
                print('switching goals')
                endpoint_ind = (endpoint_ind + 1) % len(endpoints)
                endpoint = endpoints[endpoint_ind]
                eepos_goal = np.tile(endpoint, N).T
            
            print(goaldist)
            if goaldist > 1.1:
                print("breaking on big goal dist")
                break
                
            # Optimize trajectory
            xu_new = self.sqp.sqp(xcur, eepos_goal, XU)
            
            # Simulate forward using control
            sim_time = self.sim_dt
            sim_steps = 0  # full steps taken
            while sim_time > 0:
                timestep = min(sim_time, self.sim_dt)
                control = xu_new[sim_steps*(nx+nu)+nx:(sim_steps+1)*(nx+nu)]
                xcur = np.vstack(rk4(self.robot.model, self.robot.data, xcur[:nq], xcur[nq:nx], control, timestep)).reshape(-1)
                
                if timestep > 0.5 * self.sim_dt:
                    sim_steps += 1
                
                sim_time -= timestep
                xpath.append(xcur[:nq])
                
            # Update trajectory with new solution
            if sim_steps > 0:
                XU[:-(sim_steps)*(nx+nu) or len(XU)] = xu_new[(sim_steps)*(nx+nu):]
                
            # Update first and last states
            XU[:nx] = xcur.reshape(-1)  # first state is current state
            XU[-nx:] = np.zeros(nx)
            
        return xpath