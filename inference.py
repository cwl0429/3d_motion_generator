import os
import pickle
import torch
import numpy as np   
import time
from processing import Processing
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

    def infilling(self, dim, model, data):
        motion = data.to(self.DEVICE)
        motion = motion.view((1, -1, dim))
        result = motion[:, :int(self.inp_len/2), :]
        ran = int(self.inp_len/2)
        for j in range(0, len(data)-ran, ran):
            missing_data = torch.ones_like(motion[:, 0:self.out_len, :])
            inp = torch.cat((result[:, -ran:, :], missing_data, motion[:, j+ ran : j + ran * 2, :]), 1)
            out, _,_ = model(inp, self.inp_len+self.out_len, self.inp_len+self.out_len)
            result = torch.cat((result, out[:, ran:2 * ran + self.out_len, :]), 1)

        tail = len(data) - len(result.view((-1,dim)))
        if tail > 0:
            result = torch.cat((result, motion[:, -tail:, :]), 1)  
        result = result.view((-1,dim))
        return result

    def getResult(self, data, model, part):
        if part == 'torso':
            dim = 21
        else:
            dim = 18
        if self.args_type == 'infilling':
            result = self.infilling(dim, model, data)
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

    def main(self, data):
        partDatas = {}
        data = torch.tensor(data.astype("float32"))
        total_t = 0
        for part in self.partList:
            model = self.load_model(part)
            st = time.time()
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
            partDatas[part] = self.getResult(part_data, model, part)
            et = time.time()
            elapsed_time = et - st
            total_t += elapsed_time
        st = time.time()
        self.pred = self.combine(partDatas)
        self.pred = self.processing.calculate_position(self.pred, self.TPose)
        self.gt = self.processing.calculate_position(data, self.TPose)
        et = time.time()
        elapsed_time = et - st
        total_t += elapsed_time
        print('total execution time:', total_t, 'seconds')
        if self.args_type == 'infilling':
            result = np.zeros_like(self.pred)
            ran = int(self.inp_len/2)
            for j in range(0, len(self.gt)-ran+1, ran):
                step = int(j / ran)
                result[(ran + self.out_len) * step: (ran + self.out_len) * step + ran] = self.gt[j: j + ran]
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
            st = time.time()
            figure = AnimePlot(10)
            labels = ['Predicted', 'Ground Truth']
            figure.set_fig(labels, save_path)
            figure.set_data([self.pred, self.gt], 300)
            figure.animate()
            et = time.time()
            elapsed_time = et - st
            print('visualization execution time:', elapsed_time, 'seconds')
        
if __name__ == '__main__':
    visual = True
    inf = Inference()
    npy = np.load("test.npy")
    path = "data/result/test"

    inf.main(npy)
    inf.output(path)
    
    