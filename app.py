import pickle
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


# Ajuste das pastas de template e assets

POKEMON_FOLDER = os.path.join('template/assets', 'pokemons')

app = Flask(__name__, template_folder='template',
            static_folder='template/assets')
app.config['UPLOAD_FOLDER'] = POKEMON_FOLDER


# Import do modelo já treinado e salvo (essa parte foi feita no jupyter notebook)
modelo_pipeline = load_model('./models/my_model.h5')

# Pagina principal


@app.route('/')
def home():
    return render_template('form_result.html', tables=[],
                           result="", imagem="")


Dataset_Labels = ['Porygon', 'Goldeen', 'Hitmonlee', 'Hitmonchan', 'Gloom', 'Aerodactyl', 'Mankey', 'Seadra', 'Gengar', 'Venonat', 'Articuno', 'Seaking', 'Dugtrio', 'Machop', 'Jynx', 'Oddish', 'Dodrio', 'Dragonair', 'Weedle', 'Golduck', 'Flareon', 'Krabby', 'Parasect', 'Ninetales', 'Nidoqueen', 'Kabutops', 'Drowzee', 'Caterpie', 'Jigglypuff', 'Machamp', 'Clefairy', 'Kangaskhan', 'Dragonite', 'Weepinbell', 'Fearow', 'Bellsprout', 'Grimer', 'Nidorina', 'Staryu', 'Horsea', 'Electabuzz', 'Dratini', 'Machoke', 'Magnemite', 'Squirtle', 'Gyarados', 'Pidgeot', 'Bulbasaur', 'Nidoking', 'Golem', 'Dewgong', 'Moltres', 'Zapdos', 'Poliwrath', 'Vulpix', 'Beedrill', 'Charmander', 'Abra', 'Zubat', 'Golbat', 'Wigglytuff', 'Charizard', 'Slowpoke', 'Poliwag', 'Tentacruel', 'Rhyhorn', 'Onix', 'Butterfree', 'Exeggcute', 'Sandslash', 'Pinsir', 'Rattata', 'Growlithe',
                  'Haunter', 'Pidgey', 'Ditto', 'Farfetchd', 'Pikachu', 'Raticate', 'Wartortle', 'Vaporeon', 'Cloyster', 'Hypno', 'Arbok', 'Metapod', 'Tangela', 'Kingler', 'Exeggutor', 'Kadabra', 'Seel', 'Voltorb', 'Chansey', 'Venomoth', 'Ponyta', 'Vileplume', 'Koffing', 'Blastoise', 'Tentacool', 'Lickitung', 'Paras', 'Clefable', 'Cubone', 'Marowak', 'Nidorino', 'Jolteon', 'Muk', 'Magikarp', 'Slowbro', 'Tauros', 'Kabuto', 'Spearow', 'Sandshrew', 'Eevee', 'Kakuna', 'Omastar', 'Ekans', 'Geodude', 'Magmar', 'Snorlax', 'Meowth', 'Pidgeotto', 'Venusaur', 'Persian', 'Rhydon', 'Starmie', 'Charmeleon', 'Lapras', 'Alakazam', 'Graveler', 'Psyduck', 'Rapidash', 'Doduo', 'Magneton', 'Arcanine', 'Electrode', 'Omanyte', 'Poliwhirl', 'Mew', 'Alolan Sandslash', 'Mewtwo', 'Weezing', 'Gastly', 'Victreebel', 'Ivysaur', 'MrMime', 'Shellder', 'Scyther', 'Diglett', 'Primeape', 'Raichu']

# Renderiza o resultado predito pelo modelo ML na Webpage


@app.route('/send', methods=['POST'])
def show_data():

    try:
        f = request.files['img_pokemon']
        filename = secure_filename(f.filename)
        fullfilename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(fullfilename)

        imagem = filename
        img = image.load_img(fullfilename, target_size=(224, 224))

        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        prediction = modelo_pipeline.predict(img_tensor)
        predicted_class = np.argmax(prediction)
        outcome = Dataset_Labels[predicted_class]

    except ValueError as e:
        outcome = 'OPAAAA você enviou coisa errada! ' + \
            str(e).split('\n')[-1].strip()
        imagem = 'pokemon.png'

    return render_template('form_result.html', tables=[],
                           result=outcome, imagem=imagem)


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
