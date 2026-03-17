import sys
from networksecurity.logging.logger import logging

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys):  
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()

        self.line_number = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"error occured in python script name [{0}] at line number [{0}] error message [{2}]".format(self.file_name, self.line_number, self.error_message)

if __name__ == "__main__":
    try:
        logging.Logger.info("Entering the try block")
        a = 1/0
        print("This will not be printed because of the exception",a)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
