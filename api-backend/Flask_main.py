from flask import Flask,request
import pandas as pd
from Data_Preprocessing import Data_Preprocessing
from flask_cors import CORS
import json
import pandas as pd
from Model_explanation import Model_Explanation
from  Model_Train import Model_Train

app = Flask(__name__)
CORS(app, origins='http://localhost:3000/*')

@app.route('/getdata', methods=['GET'])
def get_data():
    data = Data_Preprocessing.getTransformedData()
    data_dict = data.to_dict(orient='records')
    # Convertir le dictionnaire Python en une chaîne JSON
    json_data = json.dumps(data_dict)
    print("data from api flask are :",json_data[2])
    return json_data

@app.route('/getDataPerClass', methods=['GET'])
def get_data_per_class() :
    data = Data_Preprocessing.getTransformedData()
    print(data)
    res_dict = {
        "anomalies" : len(data[data['anomaly_if'] == -1]),
        "normales" : len(data[data['anomaly_if'] == 1])
        }
    #data_dict = res_dict.to_dict(orient='records')
    # Convertir le dictionnaire Python en une chaîne JSON
    json_data = json.dumps(res_dict)
    #print(json_data[0])
    return json_data


@app.route('/getSamplaExplanation', methods=['POST'])
def get_sample_explanation():
    
    json_data = request.data
    sample = json.loads(json_data)
    df = pd.DataFrame([sample])
    #data = Data_Preprocessing.getTransformedData()
    print("observation after convernitn to df from explanation :",df)
    explanation = Model_Explanation.explain_observation(df)
    res_dict = {
        "explanation" : explanation}
    print("explanation : ",explanation)
    json_data = json.dumps(res_dict)
    return json_data

@app.route('/inference', methods=['POST'])
def get_inference() :
    json_data = request.data
    print("json data:", json_data)
    sample = json.loads(json_data)
    df = pd.DataFrame([sample])
    #data = Data_Preprocessing.getTransformedData()
    print(df)
    inferecne = Model_Train().inference(df)
    data_dict = inferecne.to_dict(orient='records')
    
    final_data = json.dumps(data_dict)
    print('inference returned by flask :',final_data)
    return final_data
    
if __name__ == '__main__':
    app.run(debug=True)
