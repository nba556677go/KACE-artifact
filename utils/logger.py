import logging
import datetime
import os
import numpy as np
import sys


class MLLogger(logging.Logger):
    def __init__(self, log_dir, args, scriptname):
        
        log_dir = os.path.join(log_dir, str(args.device) ,args.model_name)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        log_file = os.path.join(log_dir, f"batch{args.batch_size}.log")
        self.log_file = log_file
        self.script = scriptname
        self.logger = self._create_logger()
        self.args = args

    def _create_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(filename)s:%(lineno)d] %(asctime)s - %(levelname)s - %(message)s')

        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"{message}")

    def writecsv(self, filename, data):
        #join log_dir with filename
        filename = os.path.join(self.log_dir, filename)
        #write data to csv file
        np.savetxt(filename, data, delimiter=",")

    def findCaller(self):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = logging.currentframe()
        #On some versions of IronPython, currentframe() returns None if
        #IronPython isn't run with -X:Frames.
        if f is not None:
            f = f.f_back
        rv = "(unknown file)", 0, "(unknown function)"
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == _srcfile:
                f = f.f_back
                continue
            rv = (co.co_filename, f.f_lineno, co.co_name)
            break
        return rv
    
    
    def countAverageProcessingTime(self, step_process_time):
        avg_processing_time = sum(step_process_time) / len(step_process_time)
        self.log(f"Average processing time: {avg_processing_time:.4f} seconds")
        self.log(f"Total processing time: {sum(step_process_time):.4f} seconds")
        #write req_processing_time to a csv file
        self.writecsv(f"req_device{self.args.device}batch{self.args.batch_size}_training_process_time.csv", step_process_time)
    
        
        
