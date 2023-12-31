from flask import Flask, request, jsonify, render_template
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)

# Chargez le modèle et le tokenizer
model = BartForConditionalGeneration.from_pretrained('C:/Users/user/PycharmProjects/text_mining/model_bart')
tokenizer = BartTokenizer.from_pretrained('C:/Users/user/PycharmProjects/text_mining/tkenizer_bart')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({"error": "No text provided for summarization."}), 400

    inputs = tokenizer([text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify(summary=summary)

@app.route('/summarizer')
def summarizer():
    return render_template('index.html')  # Assurez-vous que votre fichier HTML est dans le dossier 'templates'.



if __name__ == '__main__':
    app.run(debug=True)
