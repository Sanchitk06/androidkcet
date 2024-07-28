import logging
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load data
try:
    predicted_ranks_path = 'predicted_ranks.csv'
    college_names_path = 'Collegewithcode.csv'

    data = pd.read_csv(predicted_ranks_path)
    college_names = pd.read_csv(college_names_path)

    # Replace "--" with NaN and convert to numeric
    data.replace("--", np.nan, inplace=True)
    for col in data.columns[2:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    app.logger.info("Data loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading data: {e}")

@app.route('/get_college_recommendations', methods=['POST'])
def get_college_recommendations():
    try:
        rank = int(request.json['rank'])
        category = request.json['category']
        department = request.json['department']

        app.logger.debug(f"Received input - Rank: {rank}, Category: {category}, Department: {department}")

        # Filter data for the given department
        dept_data = data[data['Dept'] == department]

        if dept_data.empty:
            app.logger.warning(f"No data found for department: {department}")
            return jsonify({"error": f"No data found for department: {department}"}), 400

        # Drop rows with NaN values in the given category column
        dept_data = dept_data.dropna(subset=[category])

        if dept_data.empty:
            app.logger.warning(f"No data found for category: {category} in department: {department}")
            return jsonify({"error": f"No data found for category: {category} in department: {department}"}), 400

        # Prepare the data for k-NN
        X = dept_data[[category]].values

        if X.shape[0] == 0:
            app.logger.warning("No samples available for k-NN model")
            return jsonify({"error": "No samples available for k-NN model"}), 400

        # Fit the k-NN model
        knn = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(X)

        # Find the nearest neighbors
        distances, indices = knn.kneighbors([[rank]])

        # Get the closest colleges
        closest_colleges = dept_data.iloc[indices[0]]

        # Sort the colleges by their cutoff ranks
        sorted_colleges = closest_colleges.sort_values(by=category)

        # Split into two groups: less likely and more likely
        less_likely = sorted_colleges[sorted_colleges[category] < rank].tail(10)
        more_likely = sorted_colleges[sorted_colleges[category] >= rank].head(10)

        # Merge with college names
        less_likely = less_likely.merge(college_names, on='College_code', how='left')
        more_likely = more_likely.merge(college_names, on='College_code', how='left')

        result = {
            "less_likely": less_likely[['College_code', 'College_name']].values.tolist(),
            "more_likely": more_likely[['College_code', 'College_name']].values.tolist()
        }

        app.logger.debug("Recommendations generated successfully")
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in /get_college_recommendations: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
