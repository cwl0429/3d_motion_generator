import numpy as np
import torch

"""
Definition:
- torso: head, neck, pelvis, rshoulder, lshoulder, lthigh, rthigh (7 joints)
- left hand: lwrist, lelbow, lshoulder, neck, pelvis, rshoulder (6 joints)
- right hand: rwrist, relbow, rshoulder, neck, pelvis,  lshoulder (6 joints)
- left leg: lthigh, lknee, lankle, neck, pelvis,  rthigh (6 joints)
- right leg: rthigh, rknee, rankle, neck, pelvis, lthigh (6 joints)
"""
class JointDefV3:
    def __init__(self) -> None:
        self.n_joints_part = {}
        self.n_joints_part['entire'] = 45
        self.n_joints_part['torso'] = 21
        self.n_joints_part['limb'] = 18
        self.n_joints_part['leftarm'] = 18
        self.n_joints_part['rightarm'] = 18
        self.n_joints_part['leftleg'] = 18
        self.n_joints_part['rightleg'] = 18
        self.part_list = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso']

    def cat_torch(self, part, data):
        if part == 'torso':
            part_data = torch.cat((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        return part_data

    def cat_numpy(self, part, data):
        if part == 'torso':
            part_data = np.concatenate((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        return part_data

    def combine_numpy(self, part_datas):
        torso = part_datas['torso']
        larm = part_datas['leftarm']
        lleg = part_datas['leftleg']
        rarm = part_datas['rightarm']
        rleg = part_datas['rightleg']
        result = np.concatenate((torso[:, 0:9], rarm[:, -6:], torso[:, 9:12], larm[:, -6:], 
                                    torso[:, 12:18], rleg[:, -6:], torso[:, 18:21], lleg[:, -6:]), 1)
        return result

"""
Definition:
- torso: head, neck, pelvis, rshoulder, lshoulder, lthigh, rthigh (7 joints)
- left hand: lwrist, lelbow, lshoulder, neck, pelvis (5 joints)
- right hand: rwrist, relbow, rshoulder, neck, pelvis (5 joints)
- left leg:	lthigh, lknee, lankle, neck, pelvis (5 joints)
- light leg: rthigh, rknee, rankle, neck, pelvis (5 joints)
"""
class JointDefV2:
    def __init__(self) -> None:
        self.n_joints_part = {}
        self.n_joints_part['entire'] = 45
        self.n_joints_part['torso'] = 21
        self.n_joints_part['limb'] = 15
        self.n_joints_part['leftarm'] = 15
        self.n_joints_part['rightarm'] = 15
        self.n_joints_part['leftleg'] = 15
        self.n_joints_part['rightleg'] = 15
        self.part_list = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso']

    def cat_torch(self, part, data):
        if part == 'torso':
            part_data = torch.cat((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = torch.cat((data[:, 3:6], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = torch.cat((data[:, 3:6], data[:, 6:9], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:27], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:27], data[:, 27:30], data[:, 30:36]), axis=1)
        return part_data

    def cat_numpy(self, part, data):
        if part == 'torso':
            part_data = np.concatenate((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = np.concatenate((data[:, 3:6], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = np.concatenate((data[:, 3:6], data[:, 6:9], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:27], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:27], data[:, 27:30], data[:, 30:36]), axis=1)
        return part_data
    
    def combine_numpy(self, part_datas):
        torso = part_datas['torso']
        larm = part_datas['leftarm']
        lleg = part_datas['leftleg']
        rarm = part_datas['rightarm']
        rleg = part_datas['rightleg']
        result = np.concatenate((torso[:, 0:9], rarm[:, -6:], torso[:, 9:12], larm[:, -6:], 
                                    torso[:, 12:18], rleg[:, -6:], torso[:, 18:21], lleg[:, -6:]), 1)
        return result
    
"""
Definition:
- Torso: head, neck, pelvis (3 joints)
- Left hand: lwrist, lelbow, lshoulder, neck, pelvis (5 joints)
- Right hand: rwrist, relbow, rshoulder, neck, pelvis (5 joints)
- Left leg: lthigh, lknee, lankle, neck, pelvis (5 joints)
- Right leg: rthigh, rknee, rankle, neck, pelvis (5 joints)
"""
class JointDefV1:
    def __init__(self) -> None:
        self.n_joints_part = {}
        self.n_joints_part['entire'] = 45
        self.n_joints_part['torso'] = 9
        self.n_joints_part['limb'] = 15
        self.n_joints_part['leftarm'] = 15
        self.n_joints_part['rightarm'] = 15
        self.n_joints_part['leftleg'] = 15
        self.n_joints_part['rightleg'] = 15
        self.part_list = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso']

    def cat_torch(self, part, data):
        if part == 'torso':
            part_data = torch.cat((data[:, 0:3], data[:, 3:6], data[:, 24:27]), axis=1)
        elif part == 'leftarm':
            part_data = torch.cat((data[:, 3:6], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = torch.cat((data[:, 3:6], data[:, 6:9], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:27], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:27], data[:, 27:30], data[:, 30:36]), axis=1)
        return part_data

    def cat_numpy(self, part, data):
        if part == 'torso':
            part_data = np.concatenate((data[:, 0:3], data[:, 3:6], data[:, 24:27]), axis=1)
        elif part == 'leftarm':
            part_data = np.concatenate((data[:, 3:6], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = np.concatenate((data[:, 3:6], data[:, 6:9], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:27], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:27], data[:, 27:30], data[:, 30:36]), axis=1)
        return part_data
        
    def combine_numpy(self, part_datas):
        torso = part_datas['torso']
        larm = part_datas['leftarm']
        lleg = part_datas['leftleg']
        rarm = part_datas['rightarm']
        rleg = part_datas['rightleg']
        result = np.concatenate((torso[:, 0:6], rarm[:, 3:6], rarm[:, -6:], larm[:, 3:6], larm[:, -6:], 
                                        torso[:, 6:9], rleg[:, -9:-6], rleg[:, -6:], lleg[:, -9:-6], lleg[:, -6:]), 1)
        return result