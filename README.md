<H3>ENTER YOUR NAME       : T.RUCHITRA D</H3>
<H3>ENTER YOUR REGISTER NO:230003110043</H3>
<H3>EX.NO:1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


```
## OUTPUT:
### Dataset:
![data](https://github.com/user-attachments/assets/d7e34885-dfd0-449a-9fc2-9c08280b555d)

### X Values:
![x_values](https://github.com/user-attachments/assets/4dd01ff8-ca95-4543-afd0-12a43364edfc)

### Y Values:
![y_values](https://github.com/user-attachments/assets/cafc1b51-0c29-4a6e-9706-cf71f93887a4)

### Null Values:
![null_values](https://github.com/user-attachments/assets/9bb42d50-afff-40ec-a915-20f4c1b9a551)

### Duplicated Values:
![duplicated_values](https://github.com/user-attachments/assets/f91d2b25-689a-4072-b2fb-50aa6461ef73)

### Description:
![Uploading describe.png…]()

### Normalized Dataset:
![normalized](https://github.com/user-attachments/assets/b9add1bf-bfab-42b9-bc93-a7da2a121590)

### Training Data:
![training ](https://github.com/user-attachments/assets/b0afccc9-7f6a-4794-8154-2d5e73c45c2f)

### Testing Data:
![test](https://github.com/user-attachments/assets/e795fcea-32a2-4cae-8ecf-03f4290510e2)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


