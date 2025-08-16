from flask import Flask, request, jsonify
import util

app = Flask(__name__)

util.load_saved_artifacts()

@app.route('/get_loc_names')
def get_location_names():
    response = jsonify({
        'locations' : util.get_location_names()
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict', methods=['POST'])
def predict_home_prices():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bath = float(request.form['bath'])
    bhk = float(request.form['bhk'])
    
    response = jsonify({
        'estimated_price' : util.get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == '__main__':
    app.run(debug=True)