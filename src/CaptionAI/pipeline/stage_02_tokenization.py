from CaptionAI.config.configuration import ConfigurationManager
from CaptionAI.components.tokenization import Tokenization
from CaptionAI import logger

STAGE_NAME = "Tokenization"

class TokenizerPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            tokenization_config = config.get_tokenization_config()
            tokenization = Tokenization(config = tokenization_config)
            tokenization.init_tokenizer()
            tokenization.build_vocab()
            tokenization.save_vocab_pickle()
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} started.")
        obj = TokenizerPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed.")
    
    except Exception as e:
        logger.exception(e)
        raise e