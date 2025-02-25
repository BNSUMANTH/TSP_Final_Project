from flask import Flask, request, render_template, send_file
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize LabelEncoder for categorical columns
label_encoders = {}

def preprocess_data(df):
    # List of categorical columns
    categorical_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 
                        'Outlet_Identifier', 'Outlet_Size', 
                        'Outlet_Location_Type', 'Outlet_Type']
    
    # Encode categorical columns
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le  # Save the encoder for potential inverse transform
    return df

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file uploaded!", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected!", 400
        if file:
            try:
                # Read the uploaded file
                df = pd.read_csv(file)
                # Ensure the dataset has the expected columns
                expected_columns = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 
                                    'Item_Visibility', 'Item_Type', 'Item_MRP', 
                                    'Outlet_Identifier', 'Outlet_Establishment_Year', 
                                    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
                
                # Check if the uploaded dataset has the expected columns
                if not all(col in df.columns for col in expected_columns):
                    return "Uploaded dataset does not have the expected columns!", 400
                
                # Drop any extra columns (e.g., 'Item_Outlet_Sales')
                df = df[expected_columns]
                
                # Preprocess the data (encode categorical variables)
                df = preprocess_data(df)
                
                # Make predictions
                predictions = model.predict(df)
                
                # Add predictions to the DataFrame
                df['Predicted_Sales'] = predictions
                
                # Save the DataFrame with predictions to a new CSV file
                output_filename = 'predictions.csv'
                df.to_csv(output_filename, index=False)
                
                return render_template('result.html', predictions=predictions.tolist(), filename=output_filename)
            except Exception as e:
                return f"Error processing file: {str(e)}", 500
    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    # Use Gunicorn in production
    app.run(debug=False)  # Set debug=False for production
