import pandas as pd
from application_logging import logger

class Data_Getter_Pred:
    """
    This class shall  be used for obtaining the data from the source for prediction.

    """
    def __init__(self):
        self.prediction_file='Prediction_InputFileFromUser/SMSSpamCollection.csv'
        self.file_object=open("Training_Logs/Data_Ingestion_Log.txt", 'a+')
        self.logger_object=logger.App_Logger()

    def get_data(self):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception

        """
        self.logger_object.log(self.file_object,'Entered the get_data method of the Data_Getter class')
        try:
            self.data= pd.read_csv(self.prediction_file) # reading the data file
            self.logger_object.log(self.file_object,'Data Load Successful.Exited the get_data method of the Data_Getter class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            self.logger_object.log(self.file_object,'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()


