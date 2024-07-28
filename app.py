# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load data
predicted_ranks_path = 'predicted_ranks.csv'
college_names_path = 'Collegewithcode.csv'

data = pd.read_csv(predicted_ranks_path)
college_names = pd.read_csv(college_names_path)

# Replace "--" with NaN and convert to numeric
data.replace("--", np.nan, inplace=True)
for col in data.columns[2:]:
    data[col] = pd.to_numeric(col, errors='coerce')

@app.route('/get_college_recommendations', methods=['POST'])
def get_college_recommendations():
    rank = int(request.json['rank'])
    category = request.json['category']
    department = request.json['department']

    # Filter data for the given department
    dept_data = data[data['Dept'] == department]

    # Drop rows with NaN values in the given category column
    dept_data = dept_data.dropna(subset=[category])

    # Prepare the data for k-NN
    X = dept_data[[category]].values

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

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
