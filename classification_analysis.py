import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score,roc_curve
import plotly 
import plotly.express as px
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

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
    fig = px.histogram(data_frame=df, x="romantic", color="sex", width=800, height=800, title = "Romatic students according to gender <br><sup>Sex: 0 -> Male  1 -> Female</sup>")
    fig.show()

#   derslere ekstra para verip vermeme durumu
def paid_percentage(df):
    y = df['paid']
    print(f'Percentage of students who extra paid classes within the course subject:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who not extra paid classes within the course subject: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')
    fig = px.histogram(data_frame=df, x="paid", color="sex", width=800, height=800,
         title = "number of students who extra paid classes within the course subject according to gender <br><sup>Sex: 0 -> Male  1 -> Female</sup>")
    fig.show()


#   evde internete erişiminin olup olmama durumu   
def internet_access_percentage(df):
    y = df['internet']
    print(f'Percentage of students who have access internet access at home:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who have not access internet access at home: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')
    fig = px.histogram(data_frame=df, x="internet", color="sex", width=800, height=800,
        title = "number of students who have access internet access at home according to gender <br><sup>Sex: 0 -> Male  1 -> Female</sup>")
    fig.show()


#   yüksek öğrenim isteyip istememe durumu
def wish_higher_education_percentage(df):
    y = df['higher']
    print(f'Percentage of students who want to take higher education:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who do not want to take higher education: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')
    fig = px.histogram(data_frame=df, x="higher", color="sex", width=800, height=800,
        title = "number of students who do want to take higher education according to gender <br><sup>Sex: 0 -> Male  1 -> Female</sup>")
    fig.show()


#   anaokuluna gidip gitmeme durumu
def nursery_school_history_percentage(df):
    y = df['nursery']
    print(f'Percentage of students who attended nursery school:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who did not attend nursery school: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')
    fig = px.histogram(data_frame=df, x="nursery", color="sex", width=600, height=600,
        title = "number of students who attended nursery school according to gender <br><sup>Sex: 0 -> Male  1 -> Female</sup>")
    fig.show()

#   ekstra müfredat dışı aktivitelere katılım
def extra_curricular_activities_percentage(df):
    y = df['activities']
    print(f'Percentage of students who attend extra-curricular activities:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who do not attend extra-curricular activities: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')
    fig = px.histogram(data_frame=df, x="activities", color="sex", width=800, height=800,
        title = "number of students who attend extra-curricular activities according to gender <br><sup>Sex: 0 -> Male  1 -> Female</sup>")
    fig.show()

#   aile desteğine sahip olup olmama durumu
def family_support_percentage(df):
    y = df['famsup']
    print(f'Percentage of students who has family educational support :  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who has not family educational support : {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')
    fig = px.histogram(data_frame=df, x="famsup", color="sex", width=800, height=800,
        title = "number of students who has family educational support according to gender <br><sup>Sex: 0 -> Male  1 -> Female</sup>")
    fig.show()

#   okul desteğine sahip olup olmama durumu
def school_support_percentage(df):
    y = df['schoolsup']
    print(f'Percentage of students who has school educational support:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} student)')
    print(f'Percentage of students who has not school educational support: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} student)\n')
    fig = px.histogram(data_frame=df, x="schoolsup", color="sex", width=800, height=800,
        title = "number of students who has school educational support according to gender <br><sup>Sex: 0 -> Male  1 -> Female</sup>")
    fig.show()

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
        
        # Fjob yada Mjob'ı family_mem_job'a eşit olan öğrencilerin G1 dağılım grafiği
        data = data[data[memberJob] == family_mem_job]
        fig = px.histogram(data_frame=data, x="G1", color=memberJob, width=400, height=400)
        fig.show()
        
        data = data[data[memberJob] == family_mem_job]
        fig = px.histogram(data_frame=data, x="G2", color=memberJob, width=400, height=400)
        fig.show()
        
        data = data[data[memberJob] == family_mem_job]
        fig = px.histogram(data_frame=data, x="G3", color=memberJob, width=400, height=400)
        fig.show()

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

        # Fjob yada Medu değeri family_mem_edu'a eşit olan öğrencilerin G_x dağılım grafiği
        data = data[data[memberEdu] == family_mem_edu]
        fig = px.histogram(data_frame=data, x="G1", color=memberEdu, width=400, height=400)
        fig.show()
        
        data = data[data[memberEdu] == family_mem_edu]
        fig = px.histogram(data_frame=data, x="G2", color=memberEdu, width=400, height=400)
        fig.show()
        
        data = data[data[memberEdu] == family_mem_edu]
        fig = px.histogram(data_frame=data, x="G3", color=memberEdu, width=400, height=400)
        fig.show()

    except ValueError as exp:
        print ("Error", exp) 

def family_size_gender_relation(data):
    probability_LE3_female = round(df[df["famsize"]=="LE3"]["sex"].value_counts(normalize=True)[0], 2)
    probability_LE3_male = round(df[df["famsize"]=="LE3"]["sex"].value_counts(normalize=True)[1], 2)
    probability_GT3_female = round(df[df["famsize"]=="GT3"]["sex"].value_counts(normalize=True)[0], 2)
    probability_GT3_male = round(df[df["famsize"]=="GT3"]["sex"].value_counts(normalize=True)[1], 2)
    print (f'A student, whose famsize is LE3, has a probability of {probability_LE3_female} of being female and {probability_LE3_male} of being male.')
    print (f'A student, whose famsize is LE3, has a probability of {probability_GT3_female} of being female and {probability_GT3_male} of being male.')
    fig = px.histogram(data_frame=data, x="famsize", color="sex", width=400, height=400)
    fig.show()

def family_size_school_relation(data):
    probability_LE3_GP = round(data[data["famsize"]=="LE3"]["school"].value_counts(normalize=True)["GP"], 2)
    probability_LE3_MS = round(data[data["famsize"]=="LE3"]["school"].value_counts(normalize=True)["MS"], 2)
    probability_GT3_GP = round(data[data["famsize"]=="GT3"]["school"].value_counts(normalize=True)["GP"], 2)
    probability_GT3_MS = round(data[data["famsize"]=="GT3"]["school"].value_counts(normalize=True)["MS"], 2)
    print (f'A student, whose famsize is LE3, has a probability of {probability_LE3_GP} of being GP and {probability_LE3_MS} of being MS.')
    print (f'A student, whose famsize is LE3, has a probability of {probability_GT3_GP} of being GP and {probability_GT3_MS} of being MS.')
    fig = px.histogram(data_frame=data, x="famsize", color="school", width=400, height=400)
    fig.show()

#   absences - number of school absences (numeric: from 0 to 93)
def absences_gender_relation(data):
    fig = px.histogram(data_frame=data, x="absences", color="sex", width=400, height=400)
    fig.show()

"""
features:
famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
freetime - free time after school (numeric: from 1 - very low to 5 - very high)
goout - going out with friends (numeric: from 1 - very low to 5 - very high)
Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
health - current health status (numeric: from 1 - very bad to 5 - very good)   
data[sex] == gender olanları family quality yüzdeliklerini veriyor.
örneğin yüzde ikisinin quality'si 1 gibi. (very_low=0,02)
"""
def gender_family_quality_relation(data, sex, feature):
    try:
        features = {"famrel":"quality family relationship",
                    "freetime": "free time after school",
                    "goout":"going out with friends",
                    "Dalc":"workday alcohol consumption",
                    "Walc":"weekend alcohol consumption",
                    "health":"current health status"
                    }
        
        if sex == "female":
            gender = "F"
        elif sex == "male":
            gender = "M"
        else:
            raise ValueError("Invalid Gender Type. It should be 'female' or 'male'!")

        isFeatureValid = feature in features
        if isFeatureValid == False: 
            raise ValueError("Invalid Feature Type. It can be:\n-famrel\n-goout\n-Dalc\n-Walc\n-health\n")

        very_low = round(data[data["sex"]==gender][feature].value_counts(normalize=True)[1], 2)
        low = round(data[data["sex"]==gender][feature].value_counts(normalize=True)[2], 2)
        medium = round(data[data["sex"]==gender][feature].value_counts(normalize=True)[3], 2)
        high = round(data[data["sex"]==gender][feature].value_counts(normalize=True)[4], 2)
        very_high = round(data[data["sex"]==gender][feature].value_counts(normalize=True)[5], 2)
        print(f'A student, whose gender is {sex}, has very low {features[feature]}: {very_low}')
        print(f'A student, whose gender is {sex}, has low {features[feature]}: {low}')
        print(f'A student, whose gender is {sex}, has medium {features[feature]}: {medium}')
        print(f'A student, whose gender is {sex}, has high {features[feature]}: {high}')
        print(f'A student, whose gender is {sex}, has very high {features[feature]}: {very_high}')    
        fig = px.histogram(data_frame=data, x=feature, color="sex", width=400, height=400)
        fig.show()

    except ValueError as exp:
        print ("Error", exp) 

def convert_categorical_to_binary(data):
    # verisetindeki yes leri 1, no'ları 0 yapıyorum.
    yes_no_column_names = ["schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"]
    for i in range(len(yes_no_column_names)):    
        data[yes_no_column_names[i]] = np.where(data[yes_no_column_names[i]] =="no", 0, data[yes_no_column_names[i]])
        data[yes_no_column_names[i]] = np.where(data[yes_no_column_names[i]] =="yes", 1, data[yes_no_column_names[i]])
        data[yes_no_column_names[i]] = pd.to_numeric(data[yes_no_column_names[i]])

    #   school: GP(Gabriel Pereira):1   MS(Mousinho da Silveira):0
    data["school"] = np.where(data["school"] =="GP", 0, data["school"])
    data["school"] = np.where(data["school"] =="MS", 1, data["school"])
    data["school"] = pd.to_numeric(data["school"])

    #   sex: F=1, M=0
    data["sex"] = np.where(data["sex"] =="F", 0, data["sex"])
    data["sex"] = np.where(data["sex"] =="M", 1, data["sex"])
    data["sex"] = pd.to_numeric(data["sex"])

    #   address: U(urban):1  R(rural):0
    data["address"] = np.where(data["address"] =="U", 0, data["address"])
    data["address"] = np.where(data["address"] =="R", 1, data["address"])
    data["address"] = pd.to_numeric(data["address"])

    #   famsize: LE3(less or equal to 3 ):1  GT3(greater than 3):0
    data["famsize"] = np.where(data["famsize"] =="LE3", 0, data["famsize"])
    data["famsize"] = np.where(data["famsize"] =="GT3", 1, data["famsize"])
    data["famsize"] = pd.to_numeric(data["famsize"])

    #   Pstatus: T(living together):0  A(apart):1
    data["Pstatus"] = np.where(data["Pstatus"] =="T", 0, data["Pstatus"])
    data["Pstatus"] = np.where(data["Pstatus"] =="A", 1, data["Pstatus"])
    data["Pstatus"] = pd.to_numeric(data["Pstatus"])

    
    #   Mjob(mother's job): teacher:0 , health:1, services:2, at_home:3, other:4
    #   Fjob(father's job): teacher:0 , health:1, services:2, at_home:3, other:4    
    job = ["Mjob","Fjob"]
    for i in range(2):
        data[job[i]] = np.where(data[job[i]] =="teacher", 0, data[job[i]])
        data[job[i]] = np.where(data[job[i]] =="health", 1, data[job[i]])
        data[job[i]] = np.where(data[job[i]] =="services", 2, data[job[i]])
        data[job[i]] = np.where(data[job[i]] =="at_home", 3, data[job[i]])
        data[job[i]] = np.where(data[job[i]] =="other", 4, data[job[i]])
        data[job[i]] = pd.to_numeric(data[job[i]])

    #   reason: home:0  reputation:1    course:2    other:3
    data["reason"] = np.where(data["reason"] =="home", 0, data["reason"])
    data["reason"] = np.where(data["reason"] =="reputation", 1, data["reason"])
    data["reason"] = np.where(data["reason"] =="course", 2, data["reason"])
    data["reason"] = np.where(data["reason"] =="other", 3, data["reason"])
    data["reason"] = pd.to_numeric(data["reason"])

    #   guardian:   mother:0    father:1    other:2
    data["guardian"] = np.where(data["guardian"] =="mother", 0, data["guardian"])
    data["guardian"] = np.where(data["guardian"] =="father", 1, data["guardian"])
    data["guardian"] = np.where(data["guardian"] =="other", 2, data["guardian"])
    data["guardian"] = pd.to_numeric(data["guardian"])

    return data


def decisionTree(X_train,y_train):
    print("Decision Tree with max depth=5")
    decision_tree = tree.DecisionTreeClassifier(max_depth=5)
    decision_tree.fit(X_train,y_train)
    score = decision_tree.score(X_test,y_test)
    y_pred = decision_tree.predict(X_test)
    return score, y_pred

def randomForest(X_train,y_train):
    print("random_forest with n_estimators=5")
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train,y_train)
    score = random_forest.score(X_test,y_test)
    y_pred = random_forest.predict(X_test)
    return score, y_pred

def GradientBoosting(X_train,y_train):
    print("Gradient Boosting Classifier")
    gradient_boosting = GradientBoostingClassifier()
    gradient_boosting.fit(X_train,y_train)
    score = gradient_boosting.score(X_test,y_test)
    y_pred = gradient_boosting.predict(X_test)
    return score, y_pred

def GradientBoostingWithEstimator(X_train, y_train, n_estimator):
    #   Gradient Boosting Classifier with  n_esitmators
    gradient_boosting = GradientBoostingClassifier(n_estimators=n_estimator)
    gradient_boosting.fit(X_train,y_train)
    score = gradient_boosting.score(X_test,y_test)
    y_pred = gradient_boosting.predict(X_test)
    return score, y_pred

def NaiveBayes(X_train, y_train):
    print("Naive Bayes Classifier")
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train,y_train)
    score = naive_bayes_classifier.score(X_test,y_test)
    y_pred = naive_bayes_classifier.predict(X_test)
    return score, y_pred

def logisticRegression(X_train, y_train):
    print("Logistic Regression Classifier")
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train,y_train)
    score = logistic_regression.score(X_test,y_test)
    y_pred = logistic_regression.predict(X_test)
    return score, y_pred

def knn(X_train, y_train):
    print("K-Nearest Neighbor Classifier")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    score = knn.score(X_test,y_test)
    y_pred = knn.predict(X_test)
    return score, y_pred

def svm(X_train, y_train):
    print("SVM Classifier")
    svm = SVC(probability=True)
    svm.fit(X_train,y_train)
    score = svm.score(X_test,y_test)
    y_pred = svm.predict(X_test)
    return score, y_pred


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
    print(df.dtypes)
    # kaç tane aynı satırdan var bunu buluyor.
    print(df.duplicated().sum())


    print(missing(df))
    print(f'Numerical Columns:  {get_numerical_columns(df)}')
    print(f'Categorical Columns:  {get_categorical_columns(df)}')
    print(f'Categorical unique:  {get_categorical_number_of_unique(df)}')

    # sex_percentage(df)
    # romantic_percentage(df)
    # paid_percentage(df)
    # internet_access_percentage(df)
    # wish_higher_education_percentage(df)
    # nursery_school_history_percentage(df)
    # extra_curricular_activities_percentage(df)
    # family_support_percentage(df)
    # school_support_percentage(df)

    # # print(df.head())

    # numerical= data.select_dtypes('number').columns
    # categorical = data.select_dtypes('object').columns
    # # print(data[categorical])
    # # print(data[numerical])


    # # romantik olmayan öğrencilerin cinsiyete göre ayrımı (yüzdelik)
    # print(data[data["romantic"] == "no"]["sex"].value_counts(normalize=True))
    # # romantik olmayan öğrencilerin cinsiyete göre ayrımı (sayı)
    # print(data[data["romantic"] == "no"]["sex"].value_counts())
    
    # print("grade and family job relation")
    # grade_family_relation(data,"father","G1","at_home")
    # print()

    # print("grade and family education relation")
    # grade_family_education_relation(data,"mother","G1",2)
    # print()

    # print("gender and family size relation")
    # family_size_gender_relation(data)
    # print()
    
    # print("school and family size relation")
    # family_size_school_relation(data)
    # print()
    
    # gender_family_quality_relation(data,"male","famrel")

    # absences_gender_relation(data)

    data_converted = convert_categorical_to_binary(data)
    data_converted = data_converted.drop(["G2","G3"],axis=1)
    print(f'Categorical Columns:  {get_categorical_columns(data_converted)}')
    print(data_converted.dtypes)
    X = data_converted.values
    y = data_converted["G1"].values
    
    # 30. column(yani G1'i siliyorum.)
    X = np.delete(X,[30],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    

    
    # Prediction Probabilities:
    # random_forest_probs = random_forest.predict_proba(X_test)[:,1]
    # naive_bayes_probs = naive_bayes_classifier.predict_proba(X_test)[:,1]
    # decision_tree_probs = decision_tree.predict_proba(X_test)[:,1]
    # gradient_boosting_probs = gradient_boosting.predict_proba(X_test)[:,1]
    # gradient_boosting_estimators_40_probs = gradient_boosting_estimators_40.predict_proba(X_test)[:,1]
    # knn_probs = knn.predict_proba(X_test)[:,1]
    # logistic_regression_probs = logistic_regression.predict_proba(X_test)[:,1]
    # svm_probs = svm.predict_proba(X_test)[:,1]
    
    
    # fpr_random_forest, tpr_random_forest, threshold_random_forest = roc_curve(y_test, random_forest_probs)
    # fpr_naive_bayes, tpr_naive_bayes, threshold_naive_bayes = roc_curve(y_test, naive_bayes_probs)
    # print('roc_auc_score for random forest: ', roc_auc_score(y_test, random_forest_probs))
    # print('roc_auc_score for naive bayes: ', roc_auc_score(y_test, naive_bayes_probs))
    
    # plt.subplots(1, figsize=(10,10))
    # plt.title('Receiver Operating Characteristic - random forest')
    # plt.plot(fpr_random_forest, tpr_random_forest)
    # plt.plot([0, 1], ls="--")
    # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

    # plt.subplots(1, figsize=(10,10))
    # plt.title('Receiver Operating Characteristic -naive bayes')
    # plt.plot(fpr_naive_bayes, tpr_naive_bayes)
    # plt.plot([0, 1], ls="--")
    # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()



