from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
import shutil
from PIL import Image
import numpy as np

UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ["Airplane", "Automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
n_class = len(labels)
img_size = 32
n_result = 3  # Top 3

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        if "file" not in request.files:
            print("File doesn't exist!")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ": File not allowed!")
            return redirect(url_for("index"))

        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        image = Image.open(filepath)
        image = image.convert("RGB")
        image = image.resize((img_size, img_size))

        normalize = transforms.Normalize(
            (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) 
        to_tensor = transforms.ToTensor()
        transform = transforms.Compose([to_tensor, normalize])

        x = transform(image)
        x = x.reshape(1, 3, img_size, img_size)

        net = Net()
        net.load_state_dict(torch.load(
            "model_cnn.pth", map_location=torch.device("cpu")))
        net.eval() 

        y = net(x)
        y = F.softmax(y, dim=1)[0]
        sorted_idx = torch.argsort(-y) 
        result = ""
        for i in range(n_result):
            idx = sorted_idx[i].item()
            ratio = y[idx].item()
            label = labels[idx]
            result += "<p>" + str(round(ratio*100, 1)) + \
                "%chance of being a " + label + ".</p>"
                
        return render_template("result.html", result=Markup(result), filepath=filepath)
    else:
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
