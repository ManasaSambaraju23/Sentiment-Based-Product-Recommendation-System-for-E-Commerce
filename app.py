# app.py
from flask import Flask, render_template, request
from model import recommend_products, predict_sentiment

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Render the input form (templates/index.html must exist)
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    username = request.form.get('username', '').strip()
    if not username:
        return render_template('index.html', error="Please enter a username.")

    try:
        # Get top-5 product NAMES (list of strings)
        top5 = recommend_products(username, top_k=5)
    except Exception as e:
        return render_template('index.html', error=f"Error when generating recommendations: {e}")

    if not top5:
        return render_template('index.html', error=f"No recommendations found for username: {username}")

    # Optionally, you can also show a sample predicted sentiment for a sample text:
    # sample_sent = predict_sentiment("This product is great!")  # not used, but available

    return render_template('index.html', username=username, recommendations=top5, success=True)

if __name__ == '__main__':
    # Use port 5000 
    app.run(debug=True, port=5000)

