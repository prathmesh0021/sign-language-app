from flask import Flask, render_template, request, jsonify, send_file
import os
from model.inference import predict
from utils.report_generator import generate_report

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    graphs = [
        "graphs/confusion_matrix.png",
        "graphs/metrics.png",
        "graphs/test_accuracy.png"
    ]
    return render_template("index.html", graphs=graphs)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["video"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    prediction, confidence, top5 = predict(path)

    return jsonify({
        "prediction": prediction,
        "confidence": f"{confidence:.2f}%",
        "top5": top5
    })

@app.route("/download_report", methods=["POST"])
def download_report():

    data = request.json

    file_path = generate_report(
        data["prediction"],
        data["confidence"],
        data["top5"]
    )

    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)