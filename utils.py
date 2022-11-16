import openpyxl
import numpy as np
import pickle
from processing import Processing
class Utils:
    dataset_path = './data/split_npy_with_sign/'
    processing = Processing()
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
                    motion = np.concatenate((motion,data))
                else:
                    motion = data
                    is_first_data = False
        motion = self.processing.normalize(motion)
        motion = self.processing.calculate_angle(motion)
        return motion
    
if __name__ == "__main__":
    file_name = "excel_input_test.xlsx"
    utils = Utils(file_name)
    utils.load_xlsx()
    print(utils.combine_selected_motion())
