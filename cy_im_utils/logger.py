import logging
import logging.config

def log_setup(  logging_f_name : str,
                level_console = logging.INFO,
                level_file = logging.INFO,
                mode = 'a') -> None:
    """
    Semi-Generic Logging Setup; 
    
    Mode is set to 'a' (append) so as long as you don't delete it you can have
    the entire history of operations.
    
    """
    logging_config = dict(
                            version = 1,
                            disable_existing_loggers = False,
    handlers = {
                'console':{
                            'class':'logging.StreamHandler',
                            'formatter':'f',
                            'level':level_console
                            },
                'file':{
                            'level':level_file,
                            'formatter':'f',
                            'class':'logging.handlers.RotatingFileHandler',
                            'filename':logging_f_name,
                            'mode':mode
                        }
                },
    formatters = {'f':{'format': "%(asctime)s [%(levelname)s] - %(message)s" }},

    loggers = {
                "__main__":{ 
                            'handlers':['console','file'],
                            'level':level_console,
                            'propagate':False
                            }
               },

    root = {'handlers':['console','file'],'level':level_console}

    )

    logging.config.dictConfig(logging_config)
