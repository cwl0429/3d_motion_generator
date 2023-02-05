import os
import torch

class ModelLoader:
    def __init__(self, models_name, device, part_list) -> None:
        self.models_name = models_name
        self.DEVICE = device
        self.part_list = part_list

    def load_model(self):
        models = {}
        models_output_len = []
        for model_name in self.models_name:
            model_part = {}
            for part in self.part_list:
                model_part[part] = self._load_model(model_name, part)
                model_output_len = model_name.split('_')[-1][-2:]
            models[model_output_len] = model_part.copy()
            models_output_len.append(model_output_len)
        return models, models_output_len

    def _load_model(self, model_name, part):
        path = os.path.join("model", model_name, part)
        model = torch.load(path + "/best.pth",map_location = self.DEVICE).to(self.DEVICE)
        model.eval()
        return model

if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    part_list = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso']
    model_name = ['1011_V3_ChoreoMaster_Normal_train_angle_01_2010','1229_V3_ChoreoMaster_Normal_train_angle_01_2020',
                    '1011_V3_ChoreoMaster_Normal_train_angle_01_2030','1229_V3_ChoreoMaster_Normal_train_angle_01_2045', 
                        '1011_V3_ChoreoMaster_Normal_train_angle_01_2060']
    loader = ModelLoader(model_name, DEVICE, part_list)
    models = loader.load_model()

