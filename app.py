from flask import Flask, render_template, request, redirect, flash
import os, logging, socket
from datetime import datetime
# from gevent.pywsgi import WSGIServer
# from flask_appl_utils import get_flask_out_schema

app = Flask(__name__)



@app.route("/")
def Home():
    return render_template("result.html")

@app.route("/Analysis", methods=['GET','POST'])
def enter_sentiment():
    if request.method == 'POST':
        if request.form == 'click_btn':
            return render_template("sentiment.html")


@app.route("/x/", methods=['GET', 'POST'])
def result():

    return render_template("result.html", test_name='Sentiment test', title = '1223')




if __name__ == '__main__':
    app.run(debug=True)
