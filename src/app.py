"""
Flask web server exposing EdgePredict state and history as JSON.
Also serves the dashboard page.
"""

import os
from flask import Flask, jsonify, render_template

from state import SharedState
from inference_thread import start_inference_thread


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
state = SharedState(history_size=100)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(state.get_latest())


@app.route("/api/history")
def api_history():
    return jsonify(state.get_history())


def main():
    start_inference_thread(state, source_file="105.mat")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
