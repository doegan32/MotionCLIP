import torch
import src.utils.rotation_conversions as geometry

from .smpl import SMPL, JOINTSTYPE_ROOT
from src.models.tools.jointstypes import JOINTSTYPES


class Rotation2xyz:
    def __init__(self, device):
        self.device = device
        self.smpl_model = SMPL().eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, get_rotations_back=False, **kwargs):
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        if translation:  # x is 24 x 25 x 6 x 120
            x_translations = x[:, -1, :3] # 24 x 3 x 120 translations are just first three elements of the last joint
            x_rotations = x[:, :-1] #   24 x 24 x 6 x 120 rotations are are the first 24 joints? but in MotionClip it seems to remove rotation of root 0 - so whoch is the root???
            print("x_translations ", x_translations.shape)
            print("x_rotations ", x_rotations.shape)
        else:
            print("or is there no translation")
            x_rotations = x

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape  # 24, 120, 24, 6

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
            print("rotations ", rotations.shape) # 2880 x 24 x 3 x 3
        else:
            raise NotImplementedError("No geometry for this one.")

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0]
            print("global_orient ", global_orient.shape) # 2880 x 3 x 3
            rotations = rotations[:, 1:] # 2880 x 23 x 3 x 3
            print("rotations ", rotations.shape)

        print("betas: ", betas)
        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
        print("betas: ", betas)
        print("betas: ", betas.shape) # 2880 x 10
            # import ipdb; ipdb.set_trace()
        out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)
        for k in out.keys():
            print(k, type(out[k]), out[k].shape)

        # get the desirable joints
        joints = out[jointstype]
        print("joints: ", joints.shape) # 2880 x 24 x 3

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints
        print("x_xyz ", x_xyz.shape) # 24 x 120 x 24 x 3

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()
        print("x_xyz ", x_xyz.shape) # 24 x 24 x 3 x 120

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]
            print("rootindex ",  rootindex) # rootindex  is 0

        if translation and vertstrans:
            print("translation and vertstrans:")
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz
