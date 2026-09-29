from flask import Flask, render_template, request, jsonify
from pipelineRag import rag_pipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message.strip():
        return jsonify({'response': 'Por favor, escribe una pregunta.'}), 400
    
    try:
        # Llama a tu IA con la pregunta del usuario
        bot_response = rag_pipeline(user_message, 5)
        return jsonify({'response': bot_response})
        
    except Exception as e:
        print(f"Error procesando la pregunta: {e}")
        return jsonify({'response': 'Lo siento, tuve un problema consultando el reglamento.'}), 500

if __name__ == '__main__':
    app.run(debug=True)