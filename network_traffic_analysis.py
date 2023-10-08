
!pip install tensorflow==1.1

# Commented out IPython magic to ensure Python compatibility.
# Importing useful libraries
import numpy as np
import pandas as pd
import pickle
import itertools
import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score, plot_confusion_matrix

# Defining util Functions
def conv_array(df):
    x,y=df.drop('Class',1),df['Class'].values
    x=x.values
    y0=np.ones(len(y),np.int8)
    y0[np.where(y=='normal')]=0
    y0[np.where(y=='dos')]=1
    y0[np.where(y=='r2l')]=2
    y0[np.where(y=='u2r')]=3
    y0[np.where(y=='probe')]=4
    return x,y,y0

# Function for saving trained models
def save_model(model, filename="model.sav"):
    pickle.dump(model, open(filename, 'wb'))
    print("Model has been saved at: ", filename)

"""### Loading and Cleaning Dataset"""

# Downloading training and test sets to local disk
!wget "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv" -O "KDDTrain.csv"
!wget 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv' -O 'KDDTest.csv'

# Reading the data from CSV files using Pandas

training_set_path = "KDDTrain.csv"
test_set_path = "KDDTest.csv"

training_df = pd.read_csv(training_set_path, header=None)
testing_df = pd.read_csv(test_set_path, header=None)

print("Training set has {} rows.".format(len(training_df)))
print("Testing set has {} rows.".format(len(testing_df)))

# Adding Column names to Dataset

columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome', 'difficulty']
training_df.columns = columns
testing_df.columns = columns

# A list ot attack names that belong to each general attack type
dos_attacks=["snmpgetattack","back","land","neptune","smurf","teardrop","pod","apache2","udpstorm","processtable","mailbomb"]
r2l_attacks=["snmpguess","worm","httptunnel","named","xlock","xsnoop","sendmail","ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"]
u2r_attacks=["sqlattack","buffer_overflow","loadmodule","perl","rootkit","xterm","ps"]
probe_attacks=["ipsweep","nmap","portsweep","satan","saint","mscan"]

# Helper function to label samples to 5 classes
def label_attack (row):
    if row["outcome"] in dos_attacks:
        return "dos"
    if row["outcome"] in r2l_attacks:
        return "r2l"
    if row["outcome"] in u2r_attacks:
        return "u2r"
    if row["outcome"] in probe_attacks:
        return "probe"                        
    return "normal"


# We combine the datasets temporarily to do the labeling 
test_samples_length = len(testing_df)
df=pd.concat([training_df,testing_df])
df["Class"]=df.apply(label_attack,axis=1)

# The old outcome field is dropped since it was replaced with the Class field, the difficulty field will be dropped as well.
df=df.drop("outcome",axis=1)
df=df.drop("difficulty",axis=1)

# We again split the data into training and test sets.
training_df= df.iloc[:-test_samples_length, :]
testing_df= df.iloc[-test_samples_length:,:]

# Training Dataset
training_df.head()

# Helper function for scaling continous values
def minmax_scale_values(training_df,testing_df, col_name):
    scaler = MinMaxScaler()
    scaler = scaler.fit(training_df[col_name].values.reshape(-1, 1))
    train_values_standardized = scaler.transform(training_df[col_name].values.reshape(-1, 1))
    training_df[col_name] = train_values_standardized
    test_values_standardized = scaler.transform(testing_df[col_name].values.reshape(-1, 1))
    testing_df[col_name] = test_values_standardized
    
    
#Helper function for one hot encoding
def encode_text(training_df,testing_df, name):
    training_set_dummies = pd.get_dummies(training_df[name])
    testing_set_dummies = pd.get_dummies(testing_df[name])
    for x in training_set_dummies.columns:
        dummy_name = "{}_{}".format(name, x)
        training_df[dummy_name] = training_set_dummies[x]
        if x in testing_set_dummies.columns :
            testing_df[dummy_name]=testing_set_dummies[x]
        else :
            testing_df[dummy_name]=np.zeros(len(testing_df))
    training_df.drop(name, axis=1, inplace=True)
    testing_df.drop(name, axis=1, inplace=True)
    
    
sympolic_columns=["protocol_type","service","flag"]
label_column="Class"
for column in df.columns :
    if column in sympolic_columns:
        encode_text(training_df,testing_df,column)
    elif not column == label_column:
        minmax_scale_values(training_df,testing_df, column)

# Training Dataset after one-hot encoding
training_df.head()

testing_df.to_pickle("./testing_df.pkl")

unpickled_df = pd.read_pickle("./testing_df.pkl")

training_df.Class.value_counts()

# Creating final dataset

x_train, y_train, y0_train = conv_array(training_df)
print(y_train[0], y0_train[0])

x_test,y_test,y0_test = conv_array(testing_df)
print(y_test[0], y0_test[0])

"""## Training Part starts from here

### Random Forest Model
"""

# Loading the model
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators=100)

# Training the model
random_forest_model.fit(x_train, y0_train)
print("Model has been trained.")

y0_test[10]

random_forest_model.predict_proba(x_test[10].reshape(1,122))

# Using model for predictions

dict = {0:"Normal    ", 1:"dos ", 2:"u2r", 3:"r2l", 4:"probe"}

y_pred = random_forest_model.predict(x_test)
print("Prediction | Expected")
print("----------------------")
for i in range(10):
    print(dict[y_pred[i]],"|",y_test[i])

# Analysing the model's predictions
result = random_forest_model.score(x_test, y0_test)
print(result)

accuracy=accuracy_score(y0_test,y_pred)
recall=recall_score(y0_test,y_pred,average='micro')
precision=precision_score(y0_test,y_pred,average='micro')
f1=f1_score(y0_test,y_pred,average='micro')
print("Performance over the testing data set \n")
print("Accuracy : {} , Recall : {} , Precision : {} , F1 : {}\n".format(accuracy,recall,precision,f1 ))

#confusion matrix
plot_confusion_matrix(random_forest_model, x_test, y0_test)

save_model(random_forest_model, "random_forest_model.sav")

