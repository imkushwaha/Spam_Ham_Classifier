from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import pickle
from application_logging import logger

import warnings
warnings.filterwarnings("ignore")



class model():
    """
    This class shall  be used for model training.

    """
    
    def __init__(self):
        self.file_object=open("Training_Logs/ModelTraining_Log.txt", 'a+')
        self.logger_object=logger.App_Logger()
        
    def training(self, X_train,X_test,y_train,y_test):
        
        """
        Method Name: Training
        Description: This method used for training.
        Output: model.
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,'Entered the training method of the model class')
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        
        try:
            
            mnb = MultinomialNB()
    
            mnb.fit(self.X_train,self.y_train)
            
            y_pred = mnb.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test,y_pred)
            cmatrix = confusion_matrix(self.y_test,y_pred)
            precision = precision_score(self.y_test,y_pred)
           
            
            #saving MultiNomialNB model object in working directory
            pickle.dump(mnb,open('model.pkl','wb'))
            
            self.logger_object.log(self.file_object,f'Training Successful with Accuracy: {accuracy} and Precison: {precision}. Exited the training method of the Model class')
            
            return "Model Training Succesfull, Check your saved model in Work dir."
            
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in training method of the model class. Exception message: '+str(e))
            self.logger_object.log(self.file_object,'Model Training Unsuccessful.Exited the training method of the model class')
            raise Exception()
        
        
        
        
        
        
        
        
        
        
        
    
    
        
