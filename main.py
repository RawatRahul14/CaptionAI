from CaptionAI import logger
from CaptionAI.pipeline.stage_01_data_ingestion import DataIngestionPipeline

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e