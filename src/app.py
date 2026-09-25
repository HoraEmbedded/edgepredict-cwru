"""
Flask web server exposing EdgePredict state and history as JSON.
Serves the dashboard page and provides model metadata.
"""

import os
import json
from flask import Flask, jsonify, render_template

from state import SharedState
from inference_thread import start_inference_thread


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
META_FILE = os.path.join(BASE_DIR, "models", "cwru_model_tuned_meta.json")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
state = SharedState(history_size=100)


def load_meta():
    try:
        with open(META_FILE, "r") as fp:
            return json.load(fp)
    except Exception:
        return {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(state.get_latest())


@app.route("/api/history")
def api_history():
    return jsonify(state.get_history())


@app.route("/api/model")
def api_model():
    return jsonify(load_meta())


def main():
    start_inference_thread(state, source_file="105.mat")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
