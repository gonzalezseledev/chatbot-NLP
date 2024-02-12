from flask import Flask, render_template, request, jsonify

# Esta línea de código importa las funciones y clases necesarias del módulo Flask para construir una aplicación web utilizando el framework Flask, permitiendo manejar solicitudes del usuario, renderizar plantillas HTML y enviar datos en formato JSON como respuesta.


from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, pipeline

# - BlenderbotTokenizer: Esta clase se encarga de tokenizar el texto, es decir, de convertirlo en una secuencia de tokens que el modelo pueda entender. Este modelo divide las palabras en unidades más pequeñas y las combina con un algoritmo de "wordpieces" (divide las palabras en unidades más pequeñas para mejorar la eficiencia y la precisión del procesamiento del lenguaje natural).

# - BlenderbotForConditionalGeneration: Esta clase representa el modelo de lenguaje en sí. Es un modelo transformador pre-entrenado que puede generar texto, traducir idiomas, escribir diferentes tipos de contenido creativo y responder a preguntas de forma informativa.

# - AutoTokenizer: Esta clase genérica se puede usar para cargar un tokenizador utilizando cualquier modelo de Transformers.

# - pipeline: Esta clase se utiliza para crear un "pipeline" (secuencia de pasos o etapas que se ejecutan en orden para procesar datos o realizar una tarea compleja) de procesamiento del lenguaje natural que puede realizar una secuencia de tareas.

# En resumen, estas clases trabajan juntas para permitirte interactuar con el modelo de lenguaje Blenderbot y realizar diferentes tareas de procesamiento del lenguaje natural.


translate_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
translate_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

# Estas líneas de código utilizan la biblioteca Hugging Face Transformers para crear dos pipelines de traducción utilizando modelos pre-entrenados disponibles en el hub de modelos de Hugging Face

# translate_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es"), crea un pipeline de traducción que traduce texto del inglés al español utilizando el modelo pre-entrenado "Helsinki-NLP/opus-mt-en-es". Este modelo utiliza la arquitectura Helsinki-NLP y está entrenado específicamente para traducir del inglés al español.

# La segunda línea de código funciona igual que la línea anterior, pero usa el modelo opus-mt-es-en que está entrenado para traducir del español al inglés.


tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# Estas líneas de código preparan un tokenizador y un modelo preentrenado de Blenderbot, lo que permite generar texto condicionalmente en función de un contexto dado.

# BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill"), inicializa un tokenizador para el modelo Blenderbot. El tokenizador se utiliza para dividir texto en tokens o unidades más pequeñas, como palabras o subpalabras, que el modelo puede entender y procesar. El método from_pretrained carga un tokenizador preentrenado específico, en este caso, el modelo facebook/blenderbot-400M-distill.

# BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill"), inicializa el modelo Blenderbot para generación condicional de texto. El modelo preentrenado se carga utilizando el método from_pretrained, con el identificador del modelo "facebook/blenderbot-400M-distill". Este modelo está listo para generar texto en función de un contexto dado y está configurado para utilizar el tokenizador previamente definido.

 
app = Flask(__name__)

# Esta línea de código crea una instancia de la aplicación Flask y la asigna a la variable app, lo que permite que la aplicación se ejecute y maneje las solicitudes web entrantes.


@app.route("/")
# Esta línea crea una ruta en la aplicación Flask. Cuando un usuario accede a la URL raíz ("/") en el navegador web, Flask ejecutará la función asociada a esta ruta.

def index():
    # Esta es la función asociada a la ruta definida en la línea anterior. 
    # Su propósito es renderizar (generar) la plantilla HTML 'chat.html' y devolverla como respuesta al cliente que realizó la solicitud.

    return render_template('chat.html')
    # En resumen, cuando un usuario accede a la URL raíz ("/"), Flask ejecutará la función index() y devolverá el contenido de 'chat.html' al navegador del usuario.


@app.route("/get", methods=["GET", "POST"])
# Esta línea define la ruta en la aplicación Flask usando el decorador @app.route().
# Especifica que chat() llamará a la función cuando se realice una solicitud a la URL /get.
# El argumento indica que la ruta acepta tanto solicitudes GET como POST.

def chat():
    # Esta linea define una función denominada chat(), que maneja la lógica para procesar las solicitudes entrantes a la ruta /get.

    msg = request.form["msg"]
    # Esta línea recupera el valor del campo "msg" de los datos del formulario de solicitud.
    input = msg
    # Esta línea asigna el mensaje recuperado ( msg) a una variable denominada input.
    return generate_response(input)
    # Esta línea llama a la función generate_response(), pasando input (el mensaje del usuario) como argumento.


def generate_response(prompt):
    input_en = translate_es_en(prompt)[0]["translation_text"]
    inputs = tokenizer([input_en], return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150, num_beams=5, length_penalty=0.6, no_repeat_ngram_size=2)
    response_bot = translate_en_es(tokenizer.batch_decode(outputs, skip_special_tokens=True))[0]["translation_text"]
    return response_bot

# La función generate_response(), actúa como un intermediario entre el usuario y el modelo de lenguaje conversacional en inglés. Permite hacer preguntas en español y obtener respuestas coherentes, a pesar de que el modelo no habla español directamente.

# 1. **prompt:** Es la pregunta o frase en español que se le da a la función.

# 2. **input_en = traslate_es_en(prompt)[0]["translation_text"]:** Traduce la pregunta usando el pipeline de traducción *traslate_es_en*.

# 3. **inputs = tokenizer([input_en], return_tensors="pt"):** El tokenizador convierte la pregunta en inglés a una secuencia de números que el modelo puede entender.
#   - tokenizer se refiere a una herramienta que convierte texto en tokens. Los tokens son como las piezas más pequeñas de una oración o texto. Pueden ser palabras individuales o partes más pequeñas de palabras, dependiendo del tipo de tokenizador que estés utilizando.
#   - [input_en] es una lista que contiene el texto de entrada que queremos tokenizar. Este texto puede ser una oración o un párrafo en un determinado idioma, en este caso, probablemente en inglés.

#   - return_tensors="pt" le dice al tokenizador que queremos que nos devuelva los tokens en forma de tensores de PyTorch. PyTorch es una biblioteca de aprendizaje automático en Python, y trabajar con tensores de PyTorch facilita la manipulación y el procesamiento posterior del texto.

# 4. **outputs = model.generate(...):** El modelo Blenderbot procesa la pregunta y genera una respuesta en inglés, aplicando estrategias para evitar repeticiones y controlar la longitud.

#   - **model.generate** se refiere al modelo que estamos utilizando para generar texto basado en ciertos parámetros.

#   - **inputs** inputs es una forma de pasar un diccionario como argumentos de palabra clave a una función. En este caso, inputs contiene la información necesaria sobre el texto de entrada tokenizado, que el modelo utilizará para generar la respuesta.

#   - **max_length=150** establece la longitud máxima que puede tener la respuesta generada, que no será más larga que 150 tokens.

#   - **num_beams=5** está configurando el número de "beams" que el modelo utilizará durante la generación. Los beams son diferentes trayectorias que el modelo puede tomar durante la generación de texto. Un número mayor de beams puede permitir que el modelo explore más opciones y genere respuestas potencialmente mejores.

#   - **length_penalty=0.6** es una penalización aplicada a las respuestas más largas. Un valor más bajo de length_penalty significa que el modelo prefiere generar respuestas más cortas.

#   - **no_repeat_ngram_size=2** ayuda a evitar que el modelo genere secuencias de palabras que se repitan en la respuesta. En este caso, el modelo intentará evitar repeticiones de secuencias de 2 palabras.

# 5. **response_bot = traslate_en_es(...):** La respuesta del modelo se traduce de vuelta al español usando el pipeline *traslate_en_es*.

#   - tokenizer.batch_decode(outputs, skip_special_tokens=True): Aquí, tokenizer.batch_decode() toma los tokens generados por el modelo y los convierte de nuevo en texto legible. La opción skip_special_tokens=True le dice al tokenizador que ignore los tokens especiales que el modelo puede haber agregado, como los tokens de inicio y fin de secuencia.

#   - traslate_en_es() llama a una función de traducción, que toma el texto en inglés como entrada y devuelve su traducción al español.

#   - [0]["translation_text"] se utiliza para obtener la traducción de la primera opción devuelta por la función de traducción.

# 6. **return respuesta_bot:** La función devuelve la respuesta traducida al español, lista para ser mostrada.


if __name__ == '__main__':
    app.run()

    # Esta línea de código se utiliza para ejecutar la aplicación Flask. El bloque de código dentro de la instrucción if solo se ejecutará cuando el archivo se ejecute como un programa principal, y no cuando se importe como un módulo.
