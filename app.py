import pickle
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image


# Ajuste das pastas de template e assets

POKEMON_FOLDER = os.path.join('template/assets', 'pokemons')

app = Flask(__name__, template_folder='template', static_folder='template/assets')
app.config['UPLOAD_FOLDER'] = POKEMON_FOLDER



# Import do modelo já treinado e salvo (essa parte foi feita no jupyter notebook)
modelo_pipeline = pickle.load(open('./models/models.pkl', 'rb'))


# Pagina principal
@app.route('/')
def home():
    return render_template("homepage.html")

# Pagina Forms que é preenchido pelo usuario
@app.route('/dados_pokemon')
def dados_pokemon():
    return render_template("form.html")


def get_data():
    img_pokemon = request.form.get('img_pokemon')


    d_dict = {'img_pokemon': [img_pokemon]}

    return pd.DataFrame.from_dict(d_dict, orient='columns')

    

## Renderiza o resultado predito pelo modelo ML na Webpage
@app.route('/send', methods=['POST'])
def show_data():

    try:
        f = request.files['img_pokemon']
        print(f.filename)
        filename = secure_filename(f.filename)
        fullfilename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(fullfilename)

        # Try to open the file as an image. If it's not an image, an exception will be thrown.
        try:
            Image.open(fullfilename)
        except IOError:
            return 'Erro: arquivo não é uma imagem válida', 400

        outcome = filename
        imagem = filename
        # class_name = model.classify(os.path.join('uploads', filename))
        # df = get_data()
        # df = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

        # Faz a predição com os dados digitados pelo usuario
        # prediction = modelo_pipeline.predict(df)

        # if prediction == 'Iris-virginica':
        #     outcome = 'OPAAAA é uma Iris-virginica!'
        #     imagem = 'Iris_virginica.jpg'
        # elif prediction == 'Iris-setosa':
        #     outcome = 'Quem diria, é uma Iris-setosa!'
        #     imagem = 'Iris_setosa.jpg'
        # else:
        #     outcome = 'Eu jurava que não era uma Iris-versicolor!'
        #     imagem = 'Iris_versicolor.jpg'

    except ValueError as e:
        outcome = 'OPAAAA você enviou coisa errada! '+str(e).split('\n')[-1].strip()
        imagem = 'pokemon.png'
    
    return render_template('result.html', tables=[],
                           result=outcome, imagem=imagem)
    # return render_template('result.html', tables=[df.to_html(classes='data', header=True, col_space=10)],
    #                        result=outcome, imagem=imagem)

# retorna o a predição formatada em JSON para uma solicitação HTTP
@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    print(data)

    try:
         prediction = modelo_pipeline.predict([np.array(list(data.values()))])
         output = {
        'status': 200,
        'prediction': prediction[0]
        }
         
    except ValueError as e:
        output = {
        'status': 500,
        'prediction': str(e).split('\n')[-1].strip()
        }
   
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
