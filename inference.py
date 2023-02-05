import os
import pickle
import numpy as np   
import torch
from processing import Processing
from visualize import AnimePlot
from model_joints import JointDefV3
from model_loader import ModelLoader
class Inference:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processing = Processing()
    joint_def = JointDefV3()
    part_list = joint_def.part_list
    args_type = 'infilling'
    inp_len = 20
    
    with open("./data/TPose/s_01_act_02_subact_01_ca_01.pickle", 'rb')as fpick:
        TPose = pickle.load(fpick)[0]

    def __init__(self):
        model_name = ['1011_V3_ChoreoMaster_Normal_train_angle_01_2010','1229_V3_ChoreoMaster_Normal_train_angle_01_2020',
                        '1011_V3_ChoreoMaster_Normal_train_angle_01_2030','1229_V3_ChoreoMaster_Normal_train_angle_01_2045', 
                            '1011_V3_ChoreoMaster_Normal_train_angle_01_2060']
        model_loader = ModelLoader(model_name, self.DEVICE, self.part_list)
        self.models, self.models_len = model_loader.load_model()

    def model_select(self, target_len):
        for model_len in self.models_len:
            target_model = model_len
            if int(model_len) > int(target_len):
                break
        return self.models[target_model], int(model_len)

    def infilling(self, part, dim, data, data_len, interpo_len):
        motion = data.to(self.DEVICE)
        motion = motion.view((1, -1, dim))
        ran = int(self.inp_len/2)
        cur_pos = data_len[0]
        result = motion[:, :cur_pos, :]
        for i, length in enumerate(data_len[1:]):
            model, out_len = self.model_select(interpo_len[i])
            model = self.models['20']
            missing_data = torch.ones_like(motion[:, 0:out_len, :])
            inp = torch.cat((result[:, -ran:, :], missing_data, motion[:, cur_pos: cur_pos+ran , :]), 1)
            out, _,_ = model[part](inp, self.inp_len+out_len, self.inp_len+out_len)
            out = out[:, ran:ran+interpo_len[i], :]
            result = torch.cat((result, out, motion[:, cur_pos: cur_pos+length , :] ), 1)
            cur_pos += length
        result = result.view((-1,dim))
        return result

    def get_result(self, data, part, data_len, interpo_len):
        dim = self.joint_def.n_joints_part[part]
        if self.args_type == 'infilling':
            result = self.infilling(part, dim, data, data_len, interpo_len)
        else:
            print('No this type!!')
        return result.detach().cpu().numpy()

    def main(self, data, data_len, interpo_len):
        part_datas = {}
        data = torch.tensor(data.astype("float32"))
        for part in self.part_list:
            part_data = self.joint_def.cat_torch(part, data)
            part_datas[part] = self.get_result(part_data, part, data_len, interpo_len)
            
        self.pred = self.joint_def.combine_numpy(part_datas)
        self.pred = self.processing.calculate_position(self.pred, self.TPose)
        self.gt = self.processing.calculate_position(data, self.TPose)
        # interpolate to target frames
        data_len = list(map(int,data_len))
        interpo_len = list(map(int,interpo_len))
        total_frame = sum(data_len) + sum(interpo_len)
        pred = np.zeros([total_frame,45])
        pred[:data_len[0]] = self.pred[:data_len[0]]
        ran = 0
        cur_pos = data_len[0]
        for i, length in enumerate(data_len[1:]):
            for model_len in self.models_len:
                if int(model_len) > int(interpo_len[i]):
                    break
            pred_interpo = self.processing.interp_motion_length(self.pred[cur_pos:cur_pos+int(model_len)], interpo_len[i])
            pred[cur_pos+ran:cur_pos+ran+interpo_len[i]] = pred_interpo
            ran += interpo_len[i]
            pred[cur_pos+ran:cur_pos+length+ran] = self.pred[cur_pos:cur_pos+length]
            cur_pos += length
        self.pred = pred
        
        gt = np.zeros_like(self.pred)
        ran = 0
        cur_pos = 0
        for i, length in enumerate(data_len):
            gt[cur_pos+ran:cur_pos+length+ran] = self.gt[cur_pos:cur_pos+length]
            ran += interpo_len[i]
            cur_pos += length
        self.gt = gt
    '''
    output .pkl and .gif
    '''
    def output(self, save_path, bpm, visual = True):
        if(bpm != 90):
            length = int(len(self.pred) * 90 / bpm)
            self.pred = self.processing.interp_motion_length(self.pred, length)
            self.gt = self.processing.interp_motion_length(self.gt, length)
        with open(f'{save_path}.pkl', 'wb') as fpick:
            pickle.dump(self.pred, fpick)
        if visual:
            figure = AnimePlot(10)
            labels = ['Predicted', 'Ground Truth']
            figure.set_fig(labels, save_path)
            figure.set_data([self.pred, self.gt], len(self.pred))
            figure.animate()
        
if __name__ == '__main__':
    visual = True
    inf = Inference()
    npy = np.load("test.npy")
    path = "data/result/test"

    inf.main(npy)
    inf.output(path)
    
    