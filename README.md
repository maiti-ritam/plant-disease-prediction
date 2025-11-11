Plant Disease Prediction (CNN + Streamlit)

This project trains a TensorFlow/Keras CNN model in a Jupyter Notebook (Plant_Disease_Prediction_CNN_Image_Classifier.ipynb) to classify 38 plant diseases from the PlantVillage dataset. The trained model (plant_disease_prediction_model.keras) and class names (class_indices.json) are then used by a Streamlit web app (app.py) to provide real-time predictions on user-uploaded images.

How to Run

1. Train Model (One-time setup)

Install: pip install tensorflow numpy pillow matplotlib jupyterlab kaggle streamlit

Kaggle API: Place your kaggle.json in the location specified in the notebook.

Run Notebook: Execute all cells in Plant_Disease_Prediction_CNN_Image_Classifier.ipynb to generate the .keras and .json files.

2. Run the Web App

Check Files: Ensure app.py, plant_disease_prediction_model.keras, and class_indices.json are in the same directory.

Start Server:

streamlit run app.py


View: Open http://localhost:8501 in your browser to use the app.
