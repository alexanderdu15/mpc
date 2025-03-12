import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as R

class Manipulator(pin.RobotWrapper):

    def __init__(self, urdf_filename):
        
        model = pin.buildModelFromUrdf(
            urdf_filename
        )
        # visual_model = pin.buildGeomFromUrdf(
        #     model,
        #     urdf_filename,
        #     pin.GeometryType.VISUAL
        # )
        # collision_model = pin.buildGeomFromUrdf(
        #     model,
        #     urdf_filename,
        #     pin.GeometryType.COLLISION
        # )
        
        super(Manipulator, self).__init__(model, None, None)
        
        # Find the end effector frame (child of the last joint)
        last_joint_idx = self.model.njoints - 1
        
        self.ee_idx = None
        for frame_id, frame in enumerate(self.model.frames):
            if frame.parentJoint == last_joint_idx and frame.type == pin.FrameType.BODY:
                self.ee_idx = frame_id
                break
        # otherwise use the joint frame
        if self.ee_idx is None:
            self.ee_idx = self.model.getFrameId(self.model.names[last_joint_idx])
        
    def forward_kinematics(self, q):
        """
        Compute forward kinematics for a given joint configuration.
        
        Returns: end effector pose (x, y, z, r, p, y)
        """
        p = self.framePlacement(self, q, self.ee_idx, update_kinematics=True)
        return p.translation
    
    def compute_ee_jacobian(self, q):
        return self.getFrameJacobian(self, q, self.ee_idx, pin.ReferenceFrame.WORLD)