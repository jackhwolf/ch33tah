from flask import Flask
import time

app = Flask(__name__)

@app.route('/v1/time')
def t():
    return {"time": str(int(time.time()*1000))}


if __name__ == "__main__":
    app.run(port=5000, debug=True)