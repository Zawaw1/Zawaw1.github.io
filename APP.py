# app.py
from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate',methods=['POST'])

def translate():
    try:
        subprocess.run(['python','/Test_mouse.py'])
        return render_template('index.html')

    except Exception as e:
        return render_template('404.html', error_message=str(e))
    
if __name__ == '__main__':
    app.run(debug=True)
