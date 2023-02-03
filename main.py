import argparse
from utils import Utils
from processing import Processing
from inference import Inference

class MotionGenerator:
    def __init__(self) -> None:
        self.processing = Processing()
        self.inference = Inference()

    def load_xlsx(self, xlxs_path) -> None:
        utils = Utils(xlxs_path)
        self.bpm = utils.load_xlsx()
        self.data, self.data_len ,self.data_len_interpo= utils.combine_selected_motion()
        
    def generate_motion(self, save_path):
        self.inference.main(self.data, self.data_len)
        self.inference.output(save_path, self.bpm)
        
if __name__ == "__main__":
    # python main.py -i data/xlsx/frames_160_bpm_144.xlsx -o data/result/demo
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--xlsx_path", type=str, help="excel file name", required=True) # eg. data/xlsx/xxxxx.xlsx
    parser.add_argument("-o", "--save_path", type=str, help="output file path", required=True, default = "result") # eg. data/result/test
    args = parser.parse_args()
    motion_generator = MotionGenerator()
    motion_generator.load_xlsx(args.xlsx_path)
    motion_generator.generate_motion(args.save_path)