import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from application_logging import logger

class splitData():
    """
    This class shall  be used for dividing the data into train and test.

    """
    
    def __init__(self):
        self.file_object=open("Training_Logs/Data_Division_Log.txt", 'a+')
        self.logger_object=logger.App_Logger()
        
        
    def TrainTest(self, df, testSize, method):
        """
        Method Name: TrainTest
        Description: This method divide the data.
        Output: X_train, X_test, y_train, y_test.
        Parameter: method: tfidf or cv, df: dataframe, testsize
        On Failure: Raise Exception
        
        """
        self.logger_object.log(self.file_object,'Entered the TrainTest method of the splitData class')
        self.df = df
        try:
        
            CV = CountVectorizer()
            tfidf = TfidfVectorizer(max_features=3000)
            
            if method == "tfidf":
                X = tfidf.fit_transform(self.df['transformed_text']).toarray()
            else:
                X = CV.fit_transform(self.df["transformed_text"]).toarray()
                
            y = self.df["target"].values
            
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=testSize,random_state=42)
            
            #saving Term-Frequency and Inverse Documnet Frequency object in working directory
            pickle.dump(tfidf,open('vectorizer.pkl','wb'))
            
            self.logger_object.log(self.file_object,'Train Test Successful.Exited the TrainTest method of the splitData class')
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in TrainTest method of the splitdata class. Exception message: '+str(e))
            self.logger_object.log(self.file_object,'Data Load Unsuccessful.Exited the TrainTest method of the splitData class')
            raise Exception()