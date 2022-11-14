import pickle
import numpy as np
import argparse    
import os
from processing import Processing
import torch
from visualize import AnimePlot

# python3 inference.py -t infilling -m new_Human3.6M_train_angle_01_1_1010 -d Human3.6M/test_angle -f s_09_act_02_subact_01_ca_01.pickle -o result/demo0923.pkl -v -p
# python3 inference.py -t infilling -m seiha_Human3.6M_train_angle_01_1_1010 -d ChoreoMaster_Normal/test_angle -f d_act_1_ca_01.pkl -o result/1011_demo_0.pkl -v -p
# python3 inference.py -t infilling -m 1011_ChoreoMaster_Normal_train_angle_01_2010 -d Tool/ -f NE6101076_1027_160_150_frames_angle.pkl -o result/tool_demo_1027.pkl -v -p

class Inference:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="Dataset Dir", required=True) # e.g. Human3.6M/test_angle
    parser.add_argument("-f", "--file", type=str, help="File name")                          
    parser.add_argument("-v", "--visual", help="Visualize", action="store_true")
    args = parser.parse_args()
    
    partList = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso']
    dataset = args.dataset.split('/')[0]
    dir = args.dataset.split('/')[1]
    file = args.file
    args_model = '1011_ChoreoMaster_Normal_train_angle_01_2010'
    inp_len = int(args_model.split('_')[-1][:2])
    out_len = int(args_model.split('_')[-1][-2:])
    args_type = 'infilling'
    

    with open("./data/TPose/s_01_act_02_subact_01_ca_01.pickle", 'rb')as fpick:
        TPose = pickle.load(fpick)[0]

    def __init__(self):
        self.models = {}
        for part in self.partList:
            self.models[part] = self.load_model(part)

    def load_model(self, part):
        # print(">>> Model loaded -->", part)
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
            # result = torch.cat((result, motion[:, j+ran+out_len:j+ran+out_len+ran, :]), 1)

        tail = len(data) - len(result.view((-1,dim)))
        if tail > 0:
            result = torch.cat((result, motion[:, -tail:, :]), 1)  
        result = result.view((-1,dim))
        return result

    def getResult(self, data, model, part):
        if part == 'torso':
            dim = 21
        elif part == 'entire':
            dim = 45
        else:
            dim = 18
        if self.args_type == 'infilling':
            result = self.infilling(dim, model, data)
        else:
            print('No this type!!')
        return result.detach().cpu().numpy()

    def combine(partDatas):
        torso = partDatas['torso']
        larm = partDatas['leftarm']
        lleg = partDatas['leftleg']
        rarm = partDatas['rightarm']
        rleg = partDatas['rightleg']
        result = np.concatenate((torso[:, 0:9], rarm[:, -6:], torso[:, 9:12], larm[:, -6:], torso[:, 12:18], rleg[:, -6:], torso[:, 18:21], lleg[:, -6:]), 1)
        return result

    def main(self, data):
        partDatas = {}
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
            partDatas[part] = self.getResult(part_data, model, part)
        pred = self.combine(partDatas)
        
        pred = Processing.calculate_position(pred, self.TPose)
        gt = Processing.calculate_position(data, self.TPose)
        if self.args_type == 'infilling':
            result = np.zeros_like(pred)
            ran = int(self.inp_len/2)
            for j in range(0, len(gt)-ran+1, ran):
                step = int(j / ran)
                result[(ran + self.out_len) * step: (ran + self.out_len) * step + ran] = gt[j: j + ran]
            gt = result
        return gt, pred
    '''
    output .pkl and .gif
    '''
    def output(self):
        pass
    # if __name__ == '__main__':
    #     gt, pred = main()
    #     path = args.out.split('.')
    #     with open(f'{path[0]}.pkl', 'wb') as fpick:
    #         pickle.dump(pred, fpick)
    #     with open(f'{path[0]}_ori.pkl', 'wb') as fpick:
    #         pickle.dump(gt, fpick)
    #     if args.visual:
    #         figure = AnimePlot()
    #         labels = ['Predicted', 'Ground Truth']
    #         figure.set_fig(labels, path[0])
    #         figure.set_data([pred, gt], 300)
    #         figure.animate()