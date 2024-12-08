from CaptionAI.config.configuration import ConfigurationManager
from CaptionAI.components.custom_dataset_create import DatasetCreation
from CaptionAI import logger

STAGE_NAME = "Custom Dataset Creation"

class DataCreationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            custom_dataset_config = config.get_dataset_config()
            create_dataset = DatasetCreation(config = custom_dataset_config)
            create_dataset.create_dataloader()
            create_dataset.save_dataset()
        except Exception as e:
            raise(e)
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} started. <<<<<<<<")
        obj = DataCreationPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed. <<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise(e)