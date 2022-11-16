from utils import Utils
from processing import Processing
from inference import Inference
class MotionGenerator:
    def __init__(self) -> None:
        self.processing = Processing()
        self.inference = Inference()
    def load_xlsx(self, xlxs_path) -> None:
        utils = Utils(xlxs_path)
        utils.load_xlsx()
        self.data = utils.combine_selected_motion()
    def generate_motion(self):
        pass
