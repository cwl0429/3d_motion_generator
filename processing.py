import os
import numpy as np
import pickle

class Processing:
    jointIndex = {"head":0, "neck":3, "rshoulder":6, "rarm":9, "rhand":12, 
                "lshoulder":15, "larm":18, "lhand":21, "pelvis":24, 
                "rthigh":27, "rknee":30,"rankle":33,"lthigh":36, "lknee":39, "lankle":42}

    joint = {"head":0, "neck":1, "rshoulder":2, "rarm":3, "rhand":4, 
                    "lshoulder":5, "larm":6, "lhand":7, "pelvis":8, 
                    "rthigh":9, "rknee":10,"rankle":11,"lthigh":12, "lknee":13, "lankle":14}

    jointChain = [["neck","pelvis"], ["head","neck"],  ["rshoulder", "neck"], ["rarm", "rshoulder"], ["rhand", "rarm"],
                    ["rthigh", "pelvis"], ["rknee", "rthigh"], ["rankle", "rknee"],
                    ["lshoulder", "neck"], ["larm", "lshoulder"], ["lhand", "larm"], 
                    ["lthigh", "pelvis"], ["lknee", "lthigh"], ["lankle", "lknee"]]
    
    def get_single_data(self, dir, filename, file):
        if dir != "":
            filepath = os.path.join("../Dataset", dir, filename, file)
        else:
            filepath = os.path.join("../Dataset", file)

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

    def normalize(self, fullbody):
        normal = np.zeros_like(fullbody)
        hips = self.jointIndex['hips']
        for i, frame in enumerate(fullbody):
            for joint in self.jointIndex.values():
                normal[i][joint:joint+3] = frame[joint:joint+3] - frame[hips:hips+3]
        return normal

    def get_angle(self, v):
        axis_x = np.array([1,0,0])
        axis_y = np.array([0,1,0])
        axis_z = np.array([0,0,1])

        thetax = axis_x.dot(v)/(np.linalg.norm(axis_x) * np.linalg.norm(v))
        thetay = axis_y.dot(v)/(np.linalg.norm(axis_y) * np.linalg.norm(v))
        thetaz = axis_z.dot(v)/(np.linalg.norm(axis_z) * np.linalg.norm(v))

        return thetax, thetay, thetaz

    def get_position(self, v, angles):
        r = np.linalg.norm(v)
        x = r*angles[0]
        y = r*angles[1]
        z = r*angles[2]

        return  x,y,z

    def calculate_angle(self, fullbody):
        AngleList = np.zeros_like(fullbody)
        for i, frame in enumerate(fullbody):
            for joint in self.jointConnect:
                v = frame[joint[0]:joint[0]+3] - frame[joint[1]:joint[1]+3]
                AngleList[i][joint[0]:joint[0]+3] = list(self.get_angle(v))
        return AngleList


    def calculate_position(self, fullbody, TP):
        PosList = np.zeros_like(fullbody)
        for i, frame in enumerate(fullbody):
            for joint in self.jointConnect:
                v = TP[joint[0]:joint[0]+3] - TP[joint[1]:joint[1]+3]
                angles = frame[joint[0]:joint[0]+3]
                root = PosList[i][joint[1]:joint[1]+3]
                PosList[i][joint[0]:joint[0]+3] = np.array(list(self.get_position(v, angles)))+root

        return PosList
