import flask
from flask import Flask, request, jsonify, render_template, Blueprint, render_template, redirect, url_for, request, flash
import model
from model import VIT, PatchEmbedding, multiHeadAttention, residual, mlp, TransformerBlock, Transformer, \
    Classification
import main
from main import *

app = Flask(__name__)

from main import Jedi
app.register_blueprint(Jedi, url_prefix='/')


if __name__ == '__main__':
    app.run(port=5000, debug=True)
