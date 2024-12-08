from CaptionAI import logger
from CaptionAI.utils.dataset import FlickrDataset, generate_batch_captions
from CaptionAI.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from CaptionAI.pipeline.stage_02_tokenization import TokenizerPipeline

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Tokenization"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started.")
    obj = TokenizerPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed.")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Dataset Creation"
try:
    