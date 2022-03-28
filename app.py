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
    start.all(select+".png",'color','Hellinger','Manhattan')
    files=glob.glob("/home/pokemonchecker/public_html/pokemonapi/*.png")
    links=[]
    for file in files:
        links.append("https://pokemonchecker.site/pokemonapi/"+file.split("/")[-1])
    return render_template('allres.html', links=links) # just to see what select is
