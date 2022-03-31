from flask import Flask, send_file,flash, redirect, render_template, \
     request, url_for
import start
import glob
import csv
from PIL import Image
from PIL.ExifTags import TAGS
import os
app = Flask(__name__)

pokemons=[]
fileslocations="/home/pokemonchecker/public_html/pokemonapi/"
with open('/home/pokemonchecker/api/pokemon_infos.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        pokemons.append(row)

@app.route('/')
def hello():
    calculations=glob.glob("/home/pokemonchecker/api/images/*.png")
    for i in range(len(calculations)):
        calculations[i]=calculations[i].split("/")[-1].split(".")[0]
    return render_template('analyze.html', calculations=calculations)
@app.route("/pokemondet" , methods=['GET', 'POST'])
def pokemondet():
    filelist = [ f for f in os.listdir(fileslocations) if f.endswith(".png") ]
    for f in filelist:
        os.remove(os.path.join(fileslocations, f))
    select = request.form.get('comp_select')
    image = Image.open("/home/pokemonchecker/api/images/"+select+".png")
    info_dict = [image.filename, image.size,image.height,image.width, image.format,image.mode,getattr(image, "is_animated", False),getattr(image, "n_frames", 1)]
    start.all(select+".png",'color','Hellinger','Manhattan')
    start.all(select+".png",'grey','Hellinger','Manhattan')
    out=start.find_pokemon("/home/pokemonchecker/api/greyscaled/"+select+".png", "/home/pokemonchecker/api/index.cpickle")
    #start.all(select+".png",'gray','Chi-Squared','Euclidean')
    data=[]
    for pok in pokemons:
        if select==pok[0]:
            data=pok
    files=glob.glob("/home/pokemonchecker/public_html/pokemonapi/*.png")
    links=[]
    for file in files:
        links.append("https://pokemonchecker.site/pokemonapi/"+file.split("/")[-1])
    urlfor="https://pokemonchecker.site/images/"+select+".png"
    return render_template('allres.html',selected=select.capitalize(), links=links,data=data,data3=info_dict,urlfor=urlfor,out1=round(out[0][0],4),out1s=out[0][1].split("\\")[1].split(".")[0],out2=round(out[1][0],4),out2s=out[1][1].split("\\")[1].split(".")[0],out3=round(out[2][0],4),out3s=out[2][1].split("\\")[1].split(".")[0],out4=round(out[3][0],4),out4s=out[3][1].split("\\")[1].split(".")[0]) # just to see what select is
