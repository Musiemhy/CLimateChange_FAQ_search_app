from flask import Flask, request, jsonify, render_template
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load the preprocessed data and the model
df = pd.read_csv('climate_change_faqs_cleaned.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight semantic search model

# Precompute embeddings for faster search
df['faq_embeddings'] = df['faq'].apply(lambda x: model.encode(x, convert_to_tensor=True))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarity for FAQs
    df['similarity'] = df['faq_embeddings'].apply(lambda x: util.cos_sim(query_embedding, x).item())
    top_results = df.sort_values('similarity', ascending=False).head(5)  # Top 5 results

    results = []
    for _, row in top_results.iterrows():
        # Split answer into sentences for detailed similarity analysis
        sentences = row['faq'].split('. ')
        sentence_embeddings = [model.encode(sentence, convert_to_tensor=True) for sentence in sentences]
        sentence_scores = [util.cos_sim(query_embedding, emb).item() for emb in sentence_embeddings]

        # Find the most relevant sentence
        max_score_index = sentence_scores.index(max(sentence_scores))
        most_relevant_sentence = sentences[max_score_index]
        
        # Append only the relevant sentence
        results.append({
            'answer': most_relevant_sentence,  # Only show the relevant sentence
            'text_type': row['text_type'],
            'source': row['source'],
            'similarity': row['similarity']
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
