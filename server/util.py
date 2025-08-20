import json
import pickle
import numpy as np

locations = None
data_col = None
model = None

def get_location_names():
    return locations

def load_saved_artifacts():
    print("Loading Saved Artifacts...")
    global locations
    global data_col
    global model
    with open("C:/Users/NEW/OneDrive/Desktop/Property_price_predictor/server/artifacts/columns.json", 'r') as f:
        data_col = json.load(f)['data_columns']
        locations = data_col[3:]

    with open("C:/Users/NEW/OneDrive/Desktop/Property_price_predictor/server/artifacts/Model_Selection.pickle", 'rb') as f:
        model = pickle.load(f)
    print("Loading Done.")

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = data_col.index(location.lower())
    except:
        loc_index = -1     
    x = np.zeros(len(data_col))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(model.predict([x])[0], 2)
    
def get_meta_data(data, column):
    return data[column].min(), data[column].max()


if __name__ == '__main__':
    load_saved_artifacts()
    #print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))