import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from application_logging import logger

class Cleansing():
    """
    This class shall  be used for cleansing data.

    """
    
    def __init__(self):
        self.file_object=open("Training_Logs/Preprocessing_Log.txt", 'a+')
        self.logger_object=logger.App_Logger()
        
    def Clean(self,df):
        
        """
        Method Name: Clean
        Description: This method perform the first step of data preproceesing by dropping unnecesssary columns.
        Output: A pandas DataFrame.
        On Failure: Raise Exception
        """
        
        self.logger_object.log(self.file_object,'Entered the Clean method of the Cleansing class')
        self.df = df
        
        try:
            columns_name = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
        
            self.df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace = True)
        
            self.df.rename(columns={"v1": "target", "v2": "text"}, inplace = True)
        
            encoder = LabelEncoder()
        
            self.df["target"] = encoder.fit_transform(self.df["target"])
        
            self.df = self.df.drop_duplicates(keep = "first")
        
            self.logger_object.log(self.file_object,'Data Cleansing done Successful.Exited the Clean method of the Cleansing class')

            return self.df
        
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in Clean method of the Cleansing class. Exception message: '+str(e))
            self.logger_object.log(self.file_object,'Data Cleansing Unsuccessful.Exited the Clean method of the Cleansing class')
            raise Exception()
        
        
    
    
    
    
        
        
        
        
        