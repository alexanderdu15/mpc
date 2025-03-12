import numpy as np
import pinocchio as pin

class SQP:
    def __init__(self, robot, problem, stats=None):
        self.robot = robot
        self.problem = problem
        self.stats = stats or {
            'qp_iters': {'values': [], 'unit': '', 'multiplier': 1},
            'linesearch_alphas': {'values': [], 'unit': '', 'multiplier': 1},
            'sqp_stepsizes': {'values': [], 'unit': '', 'multiplier': 1}
        }
    
    def eepos_cost(self, eepos_goals, XU):
        qcost = 0
        vcost = 0
        ucost = 0
        for k in range(self.problem.N):
            if k < self.problem.N-1:
                XU_k = XU[k*(self.problem.nx + self.problem.nu) : (k+1)*(self.problem.nx + self.problem.nu)]
                Q_modified = 1
            else:
                XU_k = XU[k*(self.problem.nx + self.problem.nu) : (k+1)*(self.problem.nx + self.problem.nu)-self.problem.nu]
                Q_modified = self.problem.QN_cost
            eepos = self.robot.forward_kinematics(XU_k[:self.problem.nq])
            eepos_err = eepos.T - eepos_goals[k*3:(k+1)*3]
            qcost += Q_modified * np.dot(eepos_err, eepos_err)
            vcost += self.problem.dQ_cost * np.dot(XU_k[self.problem.nq:self.problem.nx].reshape(-1), XU_k[self.problem.nq:self.problem.nx].reshape(-1))
            if k < self.problem.N-1:
                ucost += self.problem.R_cost * np.dot(XU_k[self.problem.nx:self.problem.nx+self.problem.nu].reshape(-1), XU_k[self.problem.nx:self.problem.nx+self.problem.nu].reshape(-1))
        return qcost, vcost, ucost

    def integrator_err(self, XU):
        err = 0
        for k in range(self.problem.N-1):
            xu_stride = (self.problem.nx + self.problem.nu)
            qcur = XU[k*xu_stride : k*xu_stride + self.problem.nq]
            vcur = XU[k*xu_stride + self.problem.nq : k*xu_stride + self.problem.nx]
            ucur = XU[k*xu_stride + self.problem.nx : (k+1)*xu_stride]

            a = pin.aba(self.robot.model, self.robot.data, qcur, vcur, ucur)
            qnext = pin.integrate(self.robot.model, qcur, vcur*self.problem.dt)
            vnext = vcur + a*self.problem.dt

            qnext_err = qnext - XU[(k+1)*xu_stride : (k+1)*xu_stride + self.problem.nq]
            vnext_err = vnext - XU[(k+1)*xu_stride + self.problem.nq : (k+1)*xu_stride + self.problem.nx]
            err += np.linalg.norm(qnext_err) + np.linalg.norm(vnext_err)
        return err

    def linesearch(self, XU, XU_fullstep, eepos_goals):
        base_qcost, base_vcost, base_ucost = self.eepos_cost(eepos_goals, XU)
        integrator_err = self.integrator_err(XU)
        baseCV = integrator_err + np.linalg.norm(XU[:self.problem.nx] - XU[:self.problem.nx])
        basemerit = base_qcost + base_vcost + base_ucost + self.problem.mu * baseCV
        diff = XU_fullstep - XU

        alphas = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
        fail = True
        for alpha in alphas:
            XU_new = XU + alpha * diff
            qcost_new, vcost_new, ucost_new = self.eepos_cost(eepos_goals, XU_new)
            integrator_err = self.integrator_err(XU_new)
            CV_new = integrator_err + np.linalg.norm(XU_new[:self.problem.nx] - XU[:self.problem.nx])
            merit_new = qcost_new + vcost_new + ucost_new + self.problem.mu * CV_new
            exit_condition = (merit_new <= basemerit)

            if exit_condition:
                fail = False
                break

        alpha = 0.0 if fail else alpha
        self.stats['linesearch_alphas']['values'].append(alpha)
        return alpha
    
    def sqp(self, xcur, eepos_goals, XU):
        for qp in range(2):
            self.problem.setup_qp(XU, xcur, eepos_goals)
            sol = self.problem.solve()
            
            alpha = self.linesearch(XU, sol.x, eepos_goals)
            if alpha != 0.0:
                dz_step = alpha * (sol.x - XU)
                XU = XU + dz_step
                
                dz_step_norm = np.linalg.norm(dz_step)
                self.stats['sqp_stepsizes']['values'].append(dz_step_norm)

            if dz_step_norm < 1e-3:
                break
            
        self.stats['qp_iters']['values'].append(qp+1)
        return XU
    
    def get_stats(self):
        return self.stats