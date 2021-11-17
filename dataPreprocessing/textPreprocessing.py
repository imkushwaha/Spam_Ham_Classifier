import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from application_logging import logger

def transform_text(text):
    
    ps = PorterStemmer()
            
    #converting text into lower case
    text = text.lower()
            
    #word tokenizing of text
    text = nltk.word_tokenize(text)
            
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
                    
    text = y[:]
    y.clear()
            
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
                    
    text = y[:]
    y.clear()
            
            
    for i in text:
        y.append(ps.stem(i)) 
                
    return " ".join(y)
    



class textprocessing():
    
    
    def __init__(self):
        self.file_object=open("Training_Logs/Preprocessing_Log.txt", 'a+')
        self.logger_object=logger.App_Logger()
        
        
        
    def text_preprocessor(self,df):
        """
        Method Name: textpreprocessor
        Description: This method perform text preproceesing using nltk library.
        Output: A pandas DataFrame.
        On Failure: Raise Exception
        """
        
        self.logger_object.log(self.file_object,'Entered the text_preprocessor method of the textprocessing class')
        self.df = df
        
        try:
            
            #adding number of characters column in dataframe for each record
            self.df["num_characters"] = self.df["text"].apply(len)
            
            #adding number of words column in dataframe for each record
            self.df["num_words"] = self.df["text"].apply(lambda x:len(nltk.word_tokenize(x)))
            
            #adding number of sentences column in dataframe for each record
            self.df["num_sentences"] = self.df["text"].apply(lambda x:len(nltk.sent_tokenize(x)))
            
            #
            self.df["transformed_text"] = self.df["text"].apply(transform_text)
            
            self.logger_object.log(self.file_object,'text preprocessing successful.Exited the text_preprocessor method of the textprocessing class')

            return self.df
            
        
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in text_preprocessor method of the textprocessing class. Exception message: '+str(e))
            self.logger_object.log(self.file_object,'text preprocessing Unsuccessful.Exited the text_preprocessor method of the textprocessing class')
            raise Exception()    
            
    """   
        def transform_text(self):
            
        Method Name: transform_text
        Description: This method transform text by lowering text and removing stopwords, punctuation and performing stemming.
        Output: A pandas DataFrame.
        On Failure: Raise Exception
      
        
        self.logger_object.log(self.file_object,'Entered the transform_text method of the textprocessing class')
                 
        try:
            ps = PorterStemmer()
            
            #converting text into lower case
            text = text.lower()
            
            #word tokenizing of text
            text = nltk.word_tokenize(text)
            
            y = []
            for i in text:
                if i.isalnum():
                    y.append(i)
                    
            text = y[:]
            y.clear()
            
            for i in text:
                if i not in stopwords.words('english') and i not in string.punctuation:
                    y.append(i)
                    
            text = y[:]
            y.clear()
            
            
            for i in text:
                y.append(ps.stem(i)) 
                
            return " ".join(y)
        
                    
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in transform_text method of the textprocessing class. Exception message: '+str(e))
            self.logger_object.log(self.file_object,'text preprocessing Unsuccessful.Exited the transform_text method of the textprocessing class')
            raise Exception() """
        
        
        