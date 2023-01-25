import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle
# from flask_mysqldb import MySQL
# import mysql.connector

# load model
modelCured = pickle.load(open('cured-adaboost-model.pkl', 'rb'))
modelDeceased = pickle.load(open('deceased-adaboost-model.pkl', 'rb'))

# app
app = Flask(__name__)

# #dbconnection
# connect = mysql.connector.connect('localhost','root','1','heart')
#
# # connect to database
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = '1'
# app.config['MYSQL_DB'] = 'heart'
#
# # mysql = MySQL(app)
#
# # create cursor object
# cursor = connect.cursor()
#
# # display all records
# table = cursor.fetchall()
#
# # assign data query
# query2 = "select * from deceased"
#
# # executing cursor
# cursor.execute(query2)
#
# # display all records
# table = cursor.fetchall()
#
# # fetch all columns
# print('\n Table Data:')
# for row in table:
#     print(row[0], end=" ")
#     print(row[1], end=" ")
#     print(row[2], end=" ")
#     print(row[3], end="\n")
#
# # closing cursor connection
# cursor.close()
#
# # closing connection object
# connect.close()

# routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recovery')
def homeRecovery():
    return render_template('index-cured.html')


@app.route('/predict/recovery', methods=['POST'])
def predictRecovery():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = modelCured.predict(final_features)

    output = round(prediction[0], 2)

    if (output == 1):
        prediction_text = "This individual has high recovery rate when it comes to atherosclerotic heart disease"
    else:
        prediction_text = "This individual has low recovery rate when it comes to atherosclerotic heart disease"

    return render_template('index-cured.html', prediction_text=prediction_text)


@app.route('/predict/mortality', methods=['POST'])
def predictMortality():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features, dtype=object)]
    prediction = modelDeceased.predict(final_features)

    output = round(prediction[0], 2)

    if (output == 1):
        prediction_text = "This individual has high mortality rate when it comes to atherosclerotic heart disease"
    else:
        prediction_text = "This individual has low mortality rate when it comes to atherosclerotic heart disease"

    # # Store to database
    # if request.method == "POST":
    #     details = request.form
    #     gender = details['gender']
    #     age = details['age']
    #     eritrosit = details['eritrosit']
    #     hematokrit = details['hematokrit']
    #     hemoglobin = details['hemoglobin']
    #     hermch = details['hermch']
    #     khermchc = details['khermchc']
    #     leukosit = details['leukosit']
    #     trombosit = details['trombosit']
    #     deceased = details['deceased']
    #
    #     cur = mysql.connection.cursor()
    #     cur.execute(
    #         "INSERT INTO deceased(gender, age, eritrosit, hematokrit, hemoglobin, hermch, khermchc, leukosit, trombosit, deceased) VALUES (%d, %d, %d, %d, %d, %d, %d, %d, %d, %d)",
    #         (gender, age, eritrosit, hematokrit, hemoglobin, hermch, khermchc, leukosit, trombosit, deceased))
    #     mysql.connection.commit()
    #     cur.close()
    #
    # return render_template('index.html', prediction_text=prediction_text)
    #

if __name__ == '__main__':
    app.run(port=5000, debug=True)