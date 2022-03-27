from flask import Flask, send_file,flash, redirect, render_template, \
     request, url_for
import start
import glob
app = Flask(__name__)


@app.route('/')
def hello():
    calculations=glob.glob("/home/pokemonchecker/api/images/*.png")
    for i in range(len(calculations)):
        calculations[i]=calculations[i].split("/")[-1].split(".")[0]
    return render_template('analyze.html', calculations=calculations)
@app.route("/test" , methods=['GET', 'POST'])
def test():
    select = request.form.get('comp_select')
    start.all(select+".png")
    return send_file("/home/pokemonchecker/api/foo2.png", mimetype='image/png') # just to see what select is
