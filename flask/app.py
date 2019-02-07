import os
from flask import Flask, render_template

# PORT = int(os.environ.get('PORT', 5000)
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)