from CaptionAI import logger
from CaptionAI.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from CaptionAI.pipeline.stage_02_tokenization import TokenizerPipeline
from CaptionAI.pipeline.stage_03_create_dataset import DataCreationPipeline
from CaptionAI.pipeline.stage_04_model_training import ModelTrainingPipeline

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started. <<<<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed. <<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Tokenization"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started. <<<<<<<<")
    obj = TokenizerPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed. <<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Custom Dataset Creation"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started. <<<<<<<<")
    obj = DataCreationPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed. <<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise(e)

STAGE_NAME = "Model Training"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} stated. <<<<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed. <<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e