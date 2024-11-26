from flask import Flask, request, jsonify
from flask_cors import CORS

from data_creation import *

from main import *

# Assuming `func` and other dependencies are already imported from your existing code
# Also, make sure the required modules are installed using pip if needed

app = Flask(__name__)
CORS(app)  # Enables Cross-Origin Resource Sharing

# Define your retriever (e.g., the data you want to work with)
retriever = db.as_retriever()
  # Replace with your retriever object

@app.route('/', methods=['GET'])
def query_handler():
    try:
        # Retrieve the user query from the URL parameter
        user_query = request.args.get('query')
        if not user_query:
            return jsonify({"error": "Query parameter is missing!"}), 400
        
        # Call the func method with the user query and retriever
        response = func(user_query, retriever)
        
        # Return the response in JSON format
        return jsonify({"question": user_query, "response": response['text']}), 200
    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
