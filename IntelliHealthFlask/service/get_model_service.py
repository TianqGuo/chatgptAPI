from utils.utils import ModelPrediction


class ModelService:
    def __init__(self):
        self.model_prediction = ModelPrediction()

    def predict(self, x_input):
        return self.model_prediction.predict(x_input)
