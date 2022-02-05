import os
import sys
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd

ENDPOINT = "http://localhost:9200/"

es = Elasticsearch([{"host":"localhost","port":9200}])

def check_es_connection():
    """
        Function checks elasticsearch connection is established or not.
    """
    try:
        if es.ping():
            print("Connection established")
        else:
            print("There is elasticsearch connection problem!")
    except:
        raise Exception("Elasticsearch check connection error")

def create_es_index(index_name):
    """
        That function creates a elasticsearch index.
    """
    try:
        es.indices.create(index=index_name,ignore=400)
    except:
        raise Exception("Elasticsearch create index error")

def get_all_indexes():
    """
        That function gets all indexes.
    """
    try:
        res = es.indices.get_alias("*")
        for Name in res:
            print(Name)
    except:
        raise Exception("Elasticsearch get all indexes error")

def delete_es_index(index_name):
    """
        That function deletes a elasticsearch index.
    """
    try:
        es.indices.delete(index=index_name,ignore=400)
    except:
        raise Exception("Elasticsearch delete index error")

def upload_json_doc(index_name, json_doc, id):
    """
        That function uploads a json document to the index.
    """
    try:
        res = es.index(index=index_name, doc_type=index_name, body=json_doc, id=id)
    except:
        raise Exception("Elasticsearch upload json document error")

def upload_csv_file():
    # dataset'i csv dosyasından okuyor.
    data = pd.read_csv("student-mat.csv",sep=";")

    data['id'] = 'NA'
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
  
    count = 1
    for index in data.index:
        data.loc[index,'id'] = str(count)
        count += 1

    print(data.head())

    data_dict = data.to_dict("records")
    for i in range(len(data_dict)):
        json_doc = {
            "school":data_dict[i]["school"],
            "sex":data_dict[i]["sex"],
            "age":data_dict[i]["age"],
            "romantic":data_dict[i]["romantic"],
            "final_grade":data_dict[i]["final_grade"]
        }
        es.index(index="student", doc_type="student", body=json_doc, id=data_dict[i]["id"])


if __name__=="__main__":
    check_es_connection()
    create_es_index("index1")
    get_all_indexes()
    print("--------")
    delete_es_index("index1")
    get_all_indexes()

    json_doc_1 = {
        "name":"Elif",
        "surname":"Goral",
        "age":"23",
        "interests":["piano","sudoku"]
    }
    json_doc_2 = {
        "name":"Levent",
        "surname":"Goral",
        "age":"27",
        "interests":["movie","music"]
    }
    # upload_json_doc("person",json_doc_1,1)
    create_es_index("student")
    # upload_json_doc("student",json_doc_2,2)
    upload_csv_file()
