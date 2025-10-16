Sentiment-Based Product Recommender (Flask Version)
------------------------------------------------------

📂 Folder Structure
sentiment_flask_app/
│
├── app.py
├── templates/
│   ├── index.html
│   └── result.html
│
└── sentiment_recommender_demo/   <-- place or extract your existing data folder here

🚀 How to Run in PyCharm
1. Extract this zip file.
2. Ensure your folder structure matches the above (the 'sentiment_recommender_demo' folder should be next to 'app.py').
3. In PyCharm, open the folder 'sentiment_flask_app' as a project.
4. Run the app:
   ```bash
   python app.py
   ```
5. Open your browser and visit:
   http://127.0.0.1:5000

🧠 Notes
- Type any username from your dataset (e.g., user_1 to user_100).
- The model will show the top-5 products predicted for that user.
- You can replace the data in 'sentiment_recommender_demo/' anytime (same file names).

Enjoy experimenting! 🚀
