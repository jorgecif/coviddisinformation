from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import os, joblib
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS, preprocess_string, strip_punctuation, strip_numeric
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import tensorflow as tf
from tensorflow import keras

max_len = 300


# Convertir palabras plural en singular
stemmer = SnowballStemmer("english")
original_words = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 
                'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 
                'traditional', 'reference', 'colonizer','plotted']
singles = [stemmer.stem(plural) for plural in original_words]

#app = Flask(__name__, template_folder='templates')
app = Flask(__name__)

def get_keys(val,my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


# Función para extraer subtemáticas
def topics_lda(documento): 
    unseen_document=documento
    # Preprocesamiento
    bow_vector = lda_dictionay.doc2bow(preprocess(unseen_document))
    # Aplico modelo
    prediction_lda=lda_model[bow_vector]

    probs=[]
    for i in range(0, len(prediction_lda)):
        probs.append(prediction_lda[i][1])
    max_probs=max(probs)

    position=[]
    for i in range(0,len(prediction_lda)):
        if max_probs==prediction_lda[i][1]:
            position=i
            break
    result=prediction_lda[position][0]
    return result

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# Función para modelo unificado de 2 entradas 
def aplicar_modelo_unif_2input(in1, in2, modelo_probar, tokenizer):
    clf=modelo_probar
    tok=tokenizer
    # Tokenizacion
    corpus_1=[]
    corpus_1.append(in1)
    corpus_2=pd.Series(corpus_1)
    sequences_reserva = tok.texts_to_sequences(corpus_2.values)
    in2_arr=np.array(in2) # cambio a array para pasar al modelo
    transform_vect_reserva= keras.preprocessing.sequence.pad_sequences(sequences_reserva, maxlen=max_len)    
    prediccion=clf.predict({'nlp_input': transform_vect_reserva, 'meta_input': in2_arr})    
    if prediccion > 0.5:
      label= "NO"
    else:
      label = "SI"
    alerta=[prediccion,label]
    return alerta


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/clasificar', methods=['GET','POST'])
def clasificar():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
   
        # 0. Carga de modelos
        # Modelo clasificación del tema
        news_vectorizer = open(os.path.join("static/modelos/PrediccionTema/Tema_ML_vectorizer.pkl"),"rb")
        news_cv = joblib.load(news_vectorizer)
        news_log_model = open(os.path.join("static/modelos/PrediccionTema/Tema_ML_LOG_model.pkl"),"rb") 
        news_clf = joblib.load(news_log_model)

        # Modelo LDA subtema y palabras relacionadas
        lda_dictionay_path = open(os.path.join("static/modelos/PrediccionSubtema/dictionary.pkl"),"rb") 
        lda_model_path = open(os.path.join("static/modelos/PrediccionSubtema/model_LDA.pkl"),"rb") 
        lda_dictionay = joblib.load(lda_dictionay_path)
        lda_model= joblib.load(lda_model_path)

        # Modelo final de alerta
        news_tokenizer = open(os.path.join("static/modelos/PrediccionAlerta/tokenizer.pkl"),"rb") 
        news_tk = joblib.load(news_tokenizer)
        model_alert_2input = keras.models.load_model('static/modelos/PrediccionAlerta/modelLSTM_2inputs.h5')
  
        # 1. Predicción de la temática general
        vectorized_text=news_cv.transform([rawtext]).toarray()
        prediction_tema_labels = {"Health":0, "Enviroment":1, "Lifestyle":2, "Finance":3, "Sports":4, "Politics":5, "Technology":6, "Science":7}
        prediction_tema = news_clf.predict(vectorized_text)
        prediction_tema_label = get_keys(prediction_tema,prediction_tema_labels)

        # 2. Predicción de subtemática y palabras relacionadas
        # Recuento de palabras de topics
        num_words=10
        lda_topics_cargados = lda_model.show_topics(num_words=num_words)
        topics_cargados = []
        filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]

        for topic in lda_topics_cargados:
            topics_cargados.append(preprocess_string(topic[1], filters))
        
        unseen_document = rawtext
        pred=topics_lda(unseen_document)
        words_topics_lda=str(topics_cargados[pred])
        
        # 3. Predicción de alerta con modelo unificado

        # Datos de texto = INPUT 1
        datos_analizar_texto=words_topics_lda+rawtext
        input1=datos_analizar_texto

        # Datos numéricos = INPUT 2
        metadatos=[np.asarray(prediction_tema[0]).astype(np.int32),np.asarray(pred).astype(np.int32)]
        metadatos=np.asarray(metadatos).astype(np.int32)
        input2=[]
        input2.append(metadatos)  
        in1=input1
        in2=input2
        resultado_prediccion_2=aplicar_modelo_unif_2input(in1, in2, model_alert_2input, news_tk)
        label_alerta=resultado_prediccion_2[1]
        prob_alerta=1-resultado_prediccion_2[0][0]
    
    return render_template("index.html", rawtext= rawtext.upper(), prediction_general_topic=prediction_tema_label, label_alerta=label_alerta, prob_alerta=prob_alerta, words_topics=words_topics_lda)

if __name__ == '__main__':
    app.run(debug=False)
