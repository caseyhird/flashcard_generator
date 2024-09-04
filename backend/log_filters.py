import logging

class HttpxFilter(logging.Filter):
    def filter(self, record):
        return not (record.name == 'httpx' and record.levelno == logging.INFO)
    
class OpenAIFilter(logging.Filter):
    def filter(self, record):
        return not ('openai._base_client' in record.name and record.levelno == logging.INFO)    
