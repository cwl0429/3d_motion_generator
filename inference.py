import pickle
import numpy as np   
import os
from processing import Processing
import torch
from visualize import AnimePlot

class Inference:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processing = Processing()
    partList = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso']
    args_type = 'infilling'
    args_model = '1011_ChoreoMaster_Normal_train_angle_01_2010'
    inp_len = int(args_model.split('_')[-1][:2])
    out_len = int(args_model.split('_')[-1][-2:])
    
    with open("./data/TPose/s_01_act_02_subact_01_ca_01.pickle", 'rb')as fpick:
        TPose = pickle.load(fpick)[0]

    def __init__(self):
        self.models = {}
        for part in self.partList:
            self.models[part] = self.load_model(part)

    def load_model(self, part):
        path = os.path.join("model", self.args_model, part)
        model = torch.load(path + "/best.pth",map_location = self.DEVICE).to(self.DEVICE)
        model.eval()
        return model

    def infilling(self, dim, model, data, data_len):
        motion = data.to(self.DEVICE)
        motion = motion.view((1, -1, dim))
        ran = int(self.inp_len/2)
        cur_pos = data_len[0]
        result = motion[:, :cur_pos, :]

        for len in data_len[1:]:
            missing_data = torch.ones_like(motion[:, 0:self.out_len, :])
            inp = torch.cat((result[:, -ran:, :], missing_data, motion[:, cur_pos: cur_pos+ran , :]), 1)
            out, _,_ = model(inp, self.inp_len+self.out_len, self.inp_len+self.out_len)
            result = torch.cat((result, out[:, ran:2 * ran, :], motion[:, cur_pos: cur_pos+len , :] ), 1)
            cur_pos += len

        result = result.view((-1,dim))
        return result

    def getResult(self, data, model, part, data_len):
        if part == 'torso':
            dim = 21
        else:
            dim = 18
        if self.args_type == 'infilling':
            result = self.infilling(dim, model, data, data_len)
        else:
            print('No this type!!')
        return result.detach().cpu().numpy()

    def combine(self, partDatas):
        torso = partDatas['torso']
        larm = partDatas['leftarm']
        lleg = partDatas['leftleg']
        rarm = partDatas['rightarm']
        rleg = partDatas['rightleg']
        result = np.concatenate((torso[:, 0:9], rarm[:, -6:], torso[:, 9:12], larm[:, -6:], torso[:, 12:18], rleg[:, -6:], torso[:, 18:21], lleg[:, -6:]), 1)
        return result

    def main(self, data, data_len):
        partDatas = {}
        data = torch.tensor(data.astype("float32"))
        for part in self.partList:
            model = self.load_model(part)
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
            partDatas[part] = self.getResult(part_data, model, part, data_len)
        self.pred = self.combine(partDatas)
        
        self.pred = self.processing.calculate_position(self.pred, self.TPose)
        self.gt = self.processing.calculate_position(data, self.TPose)
        if self.args_type == 'infilling':
            result = np.zeros_like(self.pred)
            ran = int(self.inp_len/2)
            cur_pos = 0
            for i, len in enumerate(data_len):
                result[cur_pos + i *ran :cur_pos+len + i *ran] = self.gt[cur_pos:cur_pos+len]
                cur_pos += len
            self.gt = result
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
    
    