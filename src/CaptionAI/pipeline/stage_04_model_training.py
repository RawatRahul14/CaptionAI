from CaptionAI import logger
from CaptionAI.config.configuration import ConfigurationManager
from CaptionAI.components.model_trainer import ModelTrain

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_train = ModelTrain(model_trainer_config)
            model_train.train_model()
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} stated. <<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed. <<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e