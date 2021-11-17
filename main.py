from dataPreprocessing import preprocessing
from dataPreprocessing import textPreprocessing
from data_division import train_test
from data_ingestion import data_loader
from model_building import model_training
#from application_logging import logger

#installizing Data_Getter class from data_loader module
get_data = data_loader.Data_Getter()

#data
df = get_data.get_data()

#installizing cleansing class from preprocessing module
preprocessor = preprocessing.Cleansing()

#preprocessed_data
preprocessed_df = preprocessor.Clean(df)

#installizing textprocessing class from textPreprocessing module
textProcessor = textPreprocessing.textprocessing()

#final__data
final_df = textProcessor.text_preprocessor(preprocessed_df)


#installizing spliData class from train_test module
data_split = train_test.splitData()

X_train,X_test,y_train,y_test = data_split.TrainTest(final_df, testSize= 0.2, method = "tfidf")


#installizing model class from model_training module
Model = model_training.model()

#strat of Training
Model.training(X_train, X_test, y_train, y_test)

print("All Module fetched perfectly, complete process done!")














