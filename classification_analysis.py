import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
import plotly.express as px
import warnings
from sklearn.metrics import classification_report
import seaborn as sns
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
#.\myenv\Scripts\activate

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
    fig = px.histogram(data_frame=df, x="romantic", color="sex", width=400, height=400, title = "Romatic students according to gender <br><sup>Sex: 0 -> Male  1 -> Female</sup>")
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
    fig = px.histogram(data_frame=df, x="internet", color="sex", width=400, height=400,
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
    fig = px.histogram(data_frame=df, x="nursery", color="sex", width=400, height=400,
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
        
        # # Fjob yada Mjob'ı family_mem_job'a eşit olan öğrencilerin G1 dağılım grafiği
        # data = data[data[memberJob] == family_mem_job]
        # fig = px.histogram(data_frame=data, x="G1", color=memberJob, width=400, height=400)
        # fig.show()
        
        # data = data[data[memberJob] == family_mem_job]
        # fig = px.histogram(data_frame=data, x="G2", color=memberJob, width=400, height=400)
        # fig.show()
        
        # data = data[data[memberJob] == family_mem_job]
        # fig = px.histogram(data_frame=data, x="G3", color=memberJob, width=400, height=400)
        # fig.show()

        data = data[data[memberJob] == family_mem_job]
        fig = px.histogram(data_frame=data, x="final_grade", color=memberJob, width=400, height=400)
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
        # data = data[data[memberEdu] == family_mem_edu]
        # fig = px.histogram(data_frame=data, x="G1", color=memberEdu, width=400, height=400)
        # fig.show()
        
        # data = data[data[memberEdu] == family_mem_edu]
        # fig = px.histogram(data_frame=data, x="G2", color=memberEdu, width=400, height=400)
        # fig.show()
        
        # data = data[data[memberEdu] == family_mem_edu]
        # fig = px.histogram(data_frame=data, x="G3", color=memberEdu, width=400, height=400)
        # fig.show()
          
        data = data[data[memberEdu] == family_mem_edu]
        fig = px.histogram(data_frame=data, x="final_grade", color=memberEdu, width=400, height=400)
        fig.show()

    except ValueError as exp:
        print ("Error", exp) 

def family_size_gender_relation(data):
    probability_LE3_female = round(data[data["famsize"]=="LE3"]["sex"].value_counts(normalize=True)[0], 2)
    probability_LE3_male = round(data[data["famsize"]=="LE3"]["sex"].value_counts(normalize=True)[1], 2)
    probability_GT3_female = round(data[data["famsize"]=="GT3"]["sex"].value_counts(normalize=True)[0], 2)
    probability_GT3_male = round(data[data["famsize"]=="GT3"]["sex"].value_counts(normalize=True)[1], 2)
    print (f'A student, whose famsize is LE3, has a probability of {probability_LE3_female} of being female and {probability_LE3_male} of being male.')
    print (f'A student, whose famsize is GT3, has a probability of {probability_GT3_female} of being female and {probability_GT3_male} of being male.')
    fig = px.histogram(data_frame=data, x="famsize", color="sex", width=400, height=400)
    fig.show()

def family_size_school_relation(data):
    probability_LE3_GP = round(data[data["famsize"]=="LE3"]["school"].value_counts(normalize=True)["GP"], 2)
    probability_LE3_MS = round(data[data["famsize"]=="LE3"]["school"].value_counts(normalize=True)["MS"], 2)
    probability_GT3_GP = round(data[data["famsize"]=="GT3"]["school"].value_counts(normalize=True)["GP"], 2)
    probability_GT3_MS = round(data[data["famsize"]=="GT3"]["school"].value_counts(normalize=True)["MS"], 2)
    print (f'A student, whose famsize is LE3, has a probability of {probability_LE3_GP} of being GP and {probability_LE3_MS} of being MS.')
    print (f'A student, whose famsize is GT3, has a probability of {probability_GT3_GP} of being GP and {probability_GT3_MS} of being MS.')
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


    data["final_grade"] = np.where(data["final_grade"] =="AA", 0, data["final_grade"])
    data["final_grade"] = np.where(data["final_grade"] =="BA", 1, data["final_grade"])
    data["final_grade"] = np.where(data["final_grade"] =="BB", 2, data["final_grade"])
    data["final_grade"] = np.where(data["final_grade"] =="CB", 3, data["final_grade"])
    data["final_grade"] = np.where(data["final_grade"] =="CC", 4, data["final_grade"])
    data["final_grade"] = np.where(data["final_grade"] =="DC", 5, data["final_grade"])
    data["final_grade"] = np.where(data["final_grade"] =="DD", 6, data["final_grade"])
    data["final_grade"] = np.where(data["final_grade"] =="FF", 7, data["final_grade"])
    data["final_grade"] = pd.to_numeric(data["final_grade"])

    return data

def decisionTree(X_train,y_train,X_test,y_test):
    decision_tree = tree.DecisionTreeClassifier(max_depth=5)
    decision_tree.fit(X_train,y_train)
    score = decision_tree.score(X_test,y_test)
    y_pred = decision_tree.predict(X_test)
    y_pred_prob = decision_tree.predict_proba(X_test)
    
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - decisionTree', fontsize=18)
    plt.show()

    print()
    print("Classification Report of decisionTree")
    print(classification_report(y_test, y_pred, digits=3))
    print()

    return score,y_pred, y_pred_prob

def randomForest(X_train,y_train,X_test,y_test):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train,y_train)
    score = random_forest.score(X_test,y_test)
    y_pred = random_forest.predict(X_test)
    y_pred_prob = random_forest.predict_proba(X_test)
    
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - randomForest', fontsize=18)
    plt.show()

    print()
    print("Classification Report of randomForest")
    print(classification_report(y_test, y_pred, digits=3))
    print()

    return score,y_pred, y_pred_prob

def GradientBoosting(X_train,y_train,X_test,y_test):
    gradient_boosting = GradientBoostingClassifier()
    gradient_boosting.fit(X_train,y_train)
    score = gradient_boosting.score(X_test,y_test)
    y_pred = gradient_boosting.predict(X_test)
    y_pred_prob = gradient_boosting.predict_proba(X_test)

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - GradientBoosting', fontsize=18)
    plt.show()

    print()
    print("Classification Report of GradientBoosting")
    print(classification_report(y_test, y_pred, digits=3))
    print()

    return score,y_pred, y_pred_prob

def GradientBoostingWithEstimator(X_train, y_train,X_test,y_test, n_estimator):
    gradient_boosting = GradientBoostingClassifier(n_estimators=n_estimator)
    gradient_boosting.fit(X_train,y_train)
    score = gradient_boosting.score(X_test,y_test)
    y_pred = gradient_boosting.predict(X_test)
    y_pred_prob = gradient_boosting.predict_proba(X_test)
  
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - GradientBoostingWithEstimator', fontsize=18)
    plt.show()
    
    print()
    print("Classification Report of GradientBoosting with ",n_estimator, " estimator")
    print(classification_report(y_test, y_pred, digits=3))
    print()

    return score,y_pred, y_pred_prob

def NaiveBayes(X_train, y_train,X_test,y_test):
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train,y_train)
    score = naive_bayes_classifier.score(X_test,y_test)
    y_pred = naive_bayes_classifier.predict(X_test)
    y_pred_prob = naive_bayes_classifier.predict_proba(X_test)
    
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - NaiveBayes', fontsize=18)
    plt.show()
    
    print()
    print("Classification Report of NaiveBayes")
    print(classification_report(y_test, y_pred, digits=3))
    print()

    return score,y_pred, y_pred_prob

def logisticRegression(X_train, y_train,X_test,y_test):
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train,y_train)
    score = logistic_regression.score(X_test,y_test)
    y_pred = logistic_regression.predict(X_test)
    y_pred_prob = logistic_regression.predict_proba(X_test)
    
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - Logistic Regression', fontsize=18)
    plt.show()

    print()
    print("Classification Report of Logistic Regression")
    print(classification_report(y_test, y_pred, digits=3))
    print()
    return score,y_pred, y_pred_prob

def knn(X_train, y_train,X_test,y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    score = knn.score(X_test,y_test)
    y_pred = knn.predict(X_test)
    y_pred_prob = knn.predict_proba(X_test)
    
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - KNN', fontsize=18)
    plt.show()

    print()
    print("Classification Report of KNN")
    print(classification_report(y_test, y_pred, digits=3))
    print()
    return score,y_pred, y_pred_prob

def svm(X_train, y_train,X_test,y_test):
    svm = SVC(probability=True)
    svm.fit(X_train,y_train)
    score = svm.score(X_test,y_test)
    y_pred = svm.predict(X_test)
    y_pred_prob = svm.predict_proba(X_test)
    
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix - SVM', fontsize=18)
    plt.show()

    # df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
    # print(df)

    print()
    print("Classification Report of SVM")
    print(classification_report(y_test, y_pred, digits=3))
    print()
    return score,y_pred, y_pred_prob


if __name__ == "__main__":

    # dataset'i csv dosyasından okuyor.
    data = pd.read_csv("dataset/student performance/student/student-mat.csv",sep=";")

    #   ilk önce grade'lar 0-20 arasında olduğu için hepsini 5 ile çarpıyorum.
    data["G1"] = data["G1"].apply(lambda x: x * 5)
    data["G2"] = data["G2"].apply(lambda x: x * 5)
    data["G3"] = data["G3"].apply(lambda x: x * 5)

    data['final_grade'] = 'NA'
    data.loc[((data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) >= 90) & ( (data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) <= 100), 'final_grade'] = 'AA' 
    data.loc[((data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) >= 80) & ( (data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) <= 89), 'final_grade'] = 'BA' 
    data.loc[((data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) >= 70) & ( (data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) <= 79), 'final_grade'] = 'BB' 
    data.loc[((data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) >= 60) & ( (data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) <= 69), 'final_grade'] = 'CB' 
    data.loc[((data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) >= 50) & ( (data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) <= 59), 'final_grade'] = 'CC' 
    data.loc[((data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) >= 45) & ( (data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) <= 49), 'final_grade'] = 'DC' 
    data.loc[((data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) >= 40) & ( (data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) <= 45), 'final_grade'] = 'DD' 
    data.loc[((data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) >= 0) & ( (data.G1 * 0.25 ) + (data.G2 * 0.25) + (data.G3 * 0.5) <= 40), 'final_grade'] = 'FF' 

    
    # fig = px.histogram(data_frame=data, x="final_grade", color="sex", width=400, height=400)
    # fig.show()

    # fig = px.histogram(data_frame=data, x="final_grade", color="schoolsup", width=400, height=400)
    # fig.show()
    
    # fig = px.histogram(data_frame=data, x="final_grade", color="famsup", width=400, height=400)
    # fig.show()

    # fig = px.histogram(data_frame=data, x="final_grade", color="paid", width=400, height=400)
    # fig.show()

    # fig = px.histogram(data_frame=data, x="final_grade", color="romantic", width=400, height=400)
    # fig.show()
    
    # fig = px.histogram(data_frame=data, x="final_grade", color="address", width=400, height=400)
    # fig.show()

    # fig = px.histogram(data_frame=data, x="final_grade", color="nursery", width=400, height=400)
    # fig.show()
    
    # fig = px.histogram(data_frame=data, x="final_grade", color="higher", width=400, height=400)
    # fig.show()

    # fig = px.histogram(data_frame=data, x="final_grade", color="activities", width=400, height=400)
    # fig.show()

    # fig = px.histogram(data_frame=data, x="final_grade", color="internet", width=400, height=400)
    # fig.show()

    # fig = px.histogram(data_frame=data, x="final_grade", color="famsize", width=400, height=400)
    # fig.show()
    
    data_famrel = data[data["sex"] == "male"]
    fig = px.histogram(data_frame=data_famrel, x="final_grade", color="famrel", width=400, height=400)
    fig.show()
    
    data_famrel = data[data["sex"] == "female"]
    fig = px.histogram(data_frame=data_famrel, x="final_grade", color="famrel", width=400, height=400)
    fig.show()


    # data'daki sex string olduğu için ve kıyaslanabilir bir şey olsun diye
    # F: 0 , M:1 yapıyoruz
    data_1 = data.copy()
    data_1["sex"] = data_1["sex"].map(lambda x: 0 if x=="F" else 1)
    df = data_1.copy()

    # #(395,34) = (row,column)
    # print(df.shape)

    # # column isimlerini, kaç satırda null olmadığını ve data tipini gösteriyor.
    # print(df.info())
    # # kaç tane aynı satırdan var bunu buluyor.
    # print(df.duplicated().sum())

    # print(missing(df))
    # print(f'Numerical Columns:  {get_numerical_columns(df)}')
    # print(f'Categorical Columns:  {get_categorical_columns(df)}')
    # print(f'Categorical unique:  {get_categorical_number_of_unique(df)}')

    # sex_percentage(df)
    # romantic_percentage(df)
    # paid_percentage(df)
    # internet_access_percentage(df)
    # wish_higher_education_percentage(df)
    # nursery_school_history_percentage(df)
    # extra_curricular_activities_percentage(df)
    # family_support_percentage(df)
    # school_support_percentage(df)

    # # # print(df.head())

    # numerical= data.select_dtypes('number').columns
    # categorical = data.select_dtypes('object').columns
    # # print(data[categorical])
    # # print(data[numerical])


    # # romantik olmayan öğrencilerin cinsiyete göre ayrımı (yüzdelik)
    # print(data[data["romantic"] == "no"]["sex"].value_counts(normalize=True))
    # # romantik olmayan öğrencilerin cinsiyete göre ayrımı (sayı)
    # print(data[data["romantic"] == "no"]["sex"].value_counts())
    
    # print("grade and family job relation")
    # grade_family_relation(data,"mother","final_grade","other")
    # print()
    # print("grade and family job relation")
    # grade_family_relation(data,"father","final_grade","other")
    # print()

    # print("grade and family education relation")
    # grade_family_education_relation(data,"mother","final_grade",4)
    # print()
    # print("grade and family education relation")
    # grade_family_education_relation(data,"father","final_grade",4)
    # print()

    # print("gender and family size relation")
    # family_size_gender_relation(data)
    # print()
    
    # print("school and family size relation")
    # family_size_school_relation(data)
    # print()
    
    gender_family_quality_relation(data,"male","famrel")
    print()
    gender_family_quality_relation(data,"female","famrel")

    # absences_gender_relation(data)

    # data_converted = convert_categorical_to_binary(data)
    # X = data_converted.values
    # y = data_converted["final_grade"].values
    
    # # 33. column(yani final_Grade'i siliyorum.)
    # X = np.delete(X,[33],axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

    # decisionTreeScore,decisionTreePred, decisionTreePredProb= decisionTree(X_train,y_train,X_test,y_test)    
    # randomForestScore,randomForestPred,randomForestPredProb = randomForest(X_train,y_train,X_test,y_test)
    # naiveBayesScore,naiveBayesPred,naiveBayesPredProb = NaiveBayes(X_train,y_train,X_test,y_test)    
    # gradientBoostingScore,gradientBoostingPred,gradientBoostingPredProb = GradientBoosting(X_train,y_train,X_test,y_test)
    # gradientBoostingEstimatorScore,gradientBoostingEstimatorPred,gradientBoostingEstimatorPredProb = GradientBoostingWithEstimator(X_train,y_train,X_test,y_test,3)
    # knnScore,knnPred,knnPredProb = knn(X_train,y_train,X_test,y_test)
    # svmScore,svmPred,svmPredProb = svm(X_train,y_train,X_test,y_test)
    # logisticRegressionScore,logisticRegressionPred,logisticRegressionPredProb = logisticRegression(X_train,y_train,X_test,y_test)
    
    # decision_tree_auc_score = roc_auc_score(y_test, decisionTreePredProb,multi_class="ovo")
    # random_forest_auc_score = roc_auc_score(y_test, randomForestPredProb,multi_class="ovo")
    # naiveBayes_auc_score = roc_auc_score(y_test, naiveBayesPredProb,multi_class="ovo")
    # gradientBoosting_auc_score = roc_auc_score(y_test, gradientBoostingPredProb,multi_class="ovo")
    # gradientBoostingEstimator_auc_score = roc_auc_score(y_test, gradientBoostingEstimatorPredProb,multi_class="ovo")
    # knn_auc_score = roc_auc_score(y_test, knnPredProb,multi_class="ovo")
    # svm_auc_score = roc_auc_score(y_test, svmPredProb,multi_class="ovo")
    # logisticRegression_auc_score = roc_auc_score(y_test, logisticRegressionPredProb,multi_class="ovo")
    
    # x = PrettyTable()
    # x.field_names = ["Algorithm", "Accuracy Score","AUC Score"]
    # x.add_row(["decisionTree", decisionTreeScore,decision_tree_auc_score])
    # x.add_row(["randomForest", randomForestScore,random_forest_auc_score])
    # x.add_row(["NaiveBayes", naiveBayesScore,naiveBayes_auc_score])
    # x.add_row(["GradientBoosting", gradientBoostingScore,gradientBoosting_auc_score])
    # x.add_row(["GradientBoostingWithEstimator", gradientBoostingEstimatorScore,gradientBoostingEstimator_auc_score])
    # x.add_row(["KNN", knnScore,knn_auc_score])
    # x.add_row(["SVM", svmScore,svm_auc_score])
    # x.add_row(["logisticRegression", logisticRegressionScore,logisticRegression_auc_score])

    # print(x)




