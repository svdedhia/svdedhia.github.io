'''
# Note
app -> web app itself
__name__ -> parameter
'''
from flask import Flask
app = Flask(__name__)

@app.route('/WheatDisease')

def WheatDisease():
    return {
        'status' : 'ok'
    }

if __name__ == '__main__':
    app.run(debug=True)