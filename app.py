from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your DataFrame 'df_selected' and target variable 'Profit' here
df_selected = pd.read_csv("/Users/dhruv/Desktop/healthcare project/healthcare_df_selected.csv")

# Assuming 'int64_df' is your DataFrame containing only int64 columns
X = df_selected.drop(columns=['Life_expectancy'])
y = df_selected['Life_expectancy']
# Assuming your DataFrame is loaded and processed here

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the Random Forest Regressor
regressor = ExtraTreesRegressor()

# Fit the model on the training data
regressor.fit(X_train, y_train)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = [float(x) for x in request.form.values()]
        input_data_numpyarray = np.asarray(input_data)
        input_reshape = input_data_numpyarray.reshape(1, -1)
        prediction = regressor.predict(input_reshape)

        # if prediction[0] == 0:
        #     result = "No Profit made"
        # else:
        #     result = "Profit Made: ${}".format(prediction[0])

        if all(x >= y for x, y in zip(input_data, (1999,-1, 0, 0, 0))):
            result = "At {} Country's Expected Human Life Expectancy : {} Years".format(int(input_data[0]),int(prediction[0]))
        else:
            result = "The prediction suggests no life expectancy"

        return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
