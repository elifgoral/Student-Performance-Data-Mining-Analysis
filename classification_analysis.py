import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold,cross_val_score, RepeatedStratifiedKFold,StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold, cross_val_predict, train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score,classification_report


#importing plotly and cufflinks in offline mode
import plotly.offline
import plotly 
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings("ignore")


pd.set_option('max_columns',100)
pd.set_option('max_rows',900)
pd.set_option('max_colwidth',200)


# sütunların kaçar tane missing cell içerdiğini buluyor.
def missing (df):
    missing_number = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    return missing_values

def get_numerical_columns(df):
    numerical= df.drop(['sex'], axis=1).select_dtypes('number').columns
    return df[numerical].columns

def get_categorical_columns(df):    
    categorical = df.select_dtypes('object').columns
    return df[categorical].columns

def get_categorical_number_of_unique(df):
    categorical = df.select_dtypes('object').columns
    return df[categorical].nunique()

#   cinsiyet yüzdeliği
def sex_percentage(df):
    y = df['sex']
    print(f'Percentage of female students:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of male students: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')

#   öğrencinin romantik olup olmama durumu
def romantic_percentage(df):
    y = df['romantic']
    print(f'Percentage of romantic students:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of not romantic students: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')

#   derslere ekstra para verip vermeme durumu
def paid_percentage(df):
    y = df['paid']
    print(f'Percentage of students who extra paid classes within the course subject:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who not extra paid classes within the course subject: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')

#   evde internete erişiminin olup olmama durumu   
def internet_access_percentage(df):
    y = df['internet']
    print(f'Percentage of students who have access internet access at home:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who have not access internet access at home: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')

#   yüksek öğrenim isteyip istememe durumu
def wish_higher_education_percentage(df):
    y = df['higher']
    print(f'Percentage of students who want to take higher education:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who do not want to take higher education: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')

#   anaokuluna gidip gitmeme durumu
def nursery_school_history_percentage(df):
    y = df['nursery']
    print(f'Percentage of students who attended nursery school:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who did not attend nursery school: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')

#   ekstra müfredat dışı aktivitelere katılım
def extra_curricular_activities_percentage(df):
    y = df['activities']
    print(f'Percentage of students who attend extra-curricular activities:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who do not attend extra-curricular activities: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')

#   aile desteğine sahip olup olmama durumu
def family_support_percentage(df):
    y = df['famsup']
    print(f'Percentage of students who has family educational support :  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who has not family educational support : {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')

#   okul desteğine sahip olup olmama durumu
def school_support_percentage(df):
    y = df['schoolsup']
    print(f'Percentage of students who has school educational support:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who has not school educational support: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')

#   her x sınavındaki(G1,G2,G3) not sayısına karşılık annesi y(at_home,service,other,teacher) olan öğrenci sayısı
#   örneğin G1'den 10 alan ve annesi ev hanımı olan 11 öğrenci var.
#   data: dataset
#   faimly_mem: mother or father
#   grade_num:  hangi sınav sonucu olduğu. G1,G2,G3 gibi.
#   mother_job:  teacher,health,services,at_home,other
def grade_family_relation(data, family_mem, grade_num, family_mem_job):
    # soldaki (G1=x) durumu, sağdaki ise Mjob=at home olma sayısı
    try:
        family_mem_dict = {"mother":"Mjob","father":"Fjob"}
        isValidFamilyMem = family_mem in family_mem_dict
        
        family_mem_job_list = ["teacher","health","services","at_home","other"]
        isValidFamilyMemJob = family_mem_job in family_mem_job_list

        if isValidFamilyMem == False:
            raise ValueError("Invalid Family Member. It should be 'mother' or 'father'!")
        if isValidFamilyMemJob == False:
            raise ValueError("Invalid Family Member Job. It should be 'teacher' or 'health' or 'services' or 'at_home' or 'other'!")

        memberJob = family_mem_dict[family_mem]
        series = data[data[memberJob] == family_mem_job][grade_num].value_counts()
        values = series.tolist()
        keys= series.keys().tolist()
        for i in range(len(values)):
            print(f'{grade_num}={keys[i]} \t\t{memberJob}={family_mem_job} num: {values[i]}')

    except ValueError as exp:
        print ("Error", exp) 


#   education values:
#   0 - none
#   1 - primary education (4th grade), 
#   2 - 5th to 9th grade
#   3 - secondary education
#   4 - higher education
def grade_family_education_relation(data,family_mem, grade_num, family_mem_edu):
    try:
        family_education_dict = {0:"none",1:"primary education (4th grade)",2:"5th to 9th grade",3:"secondary education",4:"higher education"}
        isValidFamilyEdu = family_mem_edu in family_education_dict

        family_mem_dict = {"mother":"Medu","father":"Fedu"}
        isValidFamilyMem = family_mem in family_mem_dict

        if isValidFamilyMem == False:
            raise ValueError("Invalid Family Member. It should be 'mother' or 'father'!")

        if isValidFamilyEdu == False:
            raise ValueError("Invalid Family Member Education. It should be 0,1,2,3,4 !")

        memberEdu = family_mem_dict[family_mem]
        series = data[data[memberEdu] == family_mem_edu][grade_num].value_counts()
        values = series.tolist()
        keys= series.keys().tolist()
        for i in range(len(values)):
            print(f'{grade_num}={keys[i]} \t\t{memberEdu}={family_education_dict[family_mem_edu]} num: {values[i]}')

    except ValueError as exp:
        print ("Error", exp) 


def family_size_gender_relation():
    probability_LE3_female = round(df[df["famsize"]=="LE3"]["sex"].value_counts(normalize=True)[0], 2)
    probability_LE3_male = round(df[df["famsize"]=="LE3"]["sex"].value_counts(normalize=True)[1], 2)
    probability_GT3_female = round(df[df["famsize"]=="GT3"]["sex"].value_counts(normalize=True)[0], 2)
    probability_GT3_male = round(df[df["famsize"]=="GT3"]["sex"].value_counts(normalize=True)[1], 2)
    print (f'A respondent, whose famsize is LE3, has a probability of {probability_LE3_female} of being female and {probability_LE3_male} of being male.')
    print (f'A respondent, whose famsize is LE3, has a probability of {probability_GT3_female} of being female and {probability_GT3_male} of being male.')
    fig = px.histogram(data_frame=data, x="famsize", color="sex", width=400, height=400)
    fig.show()

def family_size_school_relation():
    probability_LE3_GP = round(df[df["famsize"]=="LE3"]["school"].value_counts(normalize=True)["GP"], 2)
    probability_LE3_MS = round(df[df["famsize"]=="LE3"]["school"].value_counts(normalize=True)["MS"], 2)
    probability_GT3_GP = round(df[df["famsize"]=="GT3"]["school"].value_counts(normalize=True)["GP"], 2)
    probability_GT3_MS = round(df[df["famsize"]=="GT3"]["school"].value_counts(normalize=True)["MS"], 2)
    print (f'A respondent, whose famsize is LE3, has a probability of {probability_LE3_GP} of being GP and {probability_LE3_MS} of being MS.')
    print (f'A respondent, whose famsize is LE3, has a probability of {probability_GT3_GP} of being GP and {probability_GT3_MS} of being MS.')
    fig = px.histogram(data_frame=data, x="famsize", color="school", width=400, height=400)
    fig.show()


if __name__ == "__main__":

    # dataset'i csv dosyasından okuyor.
    data = pd.read_csv("dataset/student performance/student/student-mat.csv",sep=";")

    # data'nın ilk 5 satırını getiriyor.
    print(data.head())


    # data'daki sex string olduğu için ve kıyaslanabilir bir şey olsun diye
    # F: 0 , M:1 yapıyoruz
    data_1 = data.copy()
    data_1["sex"] = data_1["sex"].map(lambda x: 0 if x=="F" else 1)
    df = data_1.copy()

    print(df.head())

    #(395,33) = (row,column)
    print(df.shape)

    # column isimlerini, kaç satırda null olmadığını ve data tipini gösteriyor.
    print(df.info())

    # kaç tane aynı satırdan var bunu buluyor.
    print(df.duplicated().sum())


    print(missing(df))
    print(f'Numerical Columns:  {get_numerical_columns(df)}')
    print(f'Categorical Columns:  {get_categorical_columns(df)}')
    print(f'Categorical unique:  {get_categorical_number_of_unique(df)}')

    sex_percentage(df)
    romantic_percentage(df)
    paid_percentage(df)
    internet_access_percentage(df)
    wish_higher_education_percentage(df)
    nursery_school_history_percentage(df)
    extra_curricular_activities_percentage(df)
    family_support_percentage(df)
    school_support_percentage(df)

    print(df.head())

    numerical= data.select_dtypes('number').columns
    categorical = data.select_dtypes('object').columns
    print(data[categorical])
    print(data[numerical])


    # romantik olmayan öğrencilerin cinsiyete göre ayrımı (yüzdelik)
    print(data[data["romantic"] == "no"]["sex"].value_counts(normalize=True))
    # romantik olmayan öğrencilerin cinsiyete göre ayrımı (sayı)
    print(data[data["romantic"] == "no"]["sex"].value_counts())
    
    print("grade and family job relation")
    grade_family_relation(data,"father","G1","at_home")
    print()

    print("grade and family education relation")
    grade_family_education_relation(data,"mother","G1",2)
    print()

    print("gender and family size relation")
    family_size_gender_relation()
    print()
    
    print("school and family size relation")
    family_size_school_relation()
    print()
    

