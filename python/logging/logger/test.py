import sys
import logging

def test(logger : logging.Logger):
    print("First time set loglevel, setting to INFO")
    logger.setLevel(logging.INFO)
    print(f"{logger._cache=}")
    
    logger.info('info 111')
    logger.debug('debug 111')
    print(f"{logger._cache=}")
    
    print("set logger.level to DEBUG")
    logger.setLevel(logging.DEBUG)
    print(f"{logger._cache=}")
    
    logger.info('info 222')
    logger.debug('debug 222')
    
sys.path.insert(0, '.')
print("--------- CustomLoggerClass ----------")
from custom_logger_class import logger
test(logger)
print("\n\n")
print("--------- get_logger ----------")
from get_logger import logger
test(logger)




