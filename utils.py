import openpyxl
import numpy as np
import pickle
class Utils:
    dataset_path = './data/split_npy_with_sign/'
    def __init__(self, xlxs_path) -> None:
        self.xlxs_path = xlxs_path
    def load_xlsx(self):
        wb = openpyxl.load_workbook(self.xlxs_path)
        self.ws = wb.active
    def combine_selected_motion(self):
        is_first_data = True
        for i, row in enumerate(self.ws.iter_rows(values_only=True, min_row=2)):
            for anim in row:
                data = np.load(self.dataset_path + anim)
                if is_first_data != True:
                    result = np.concatenate((result,data))
                else:
                    result = data
                    is_first_data = False
        length = (i+1) * 40
        motion = np.array(result)
        file_name = self.xlxs_path.split('.')[-2]
        with open(file_name +'_{i}_{j}_frames.pkl'.format(i = length, j = length - 10),'wb')as fpick:
            pickle.dump(motion, fpick)



