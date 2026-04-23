import streamlit as st
import pickle
import pandas as pd
import os

# Select model dynamically
MODEL_PATH = "model"

model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".pkl") and f != "pipe.pkl"]

selected_model = st.sidebar.selectbox("Choose Model", model_files)

# Load model
with open(f"{MODEL_PATH}/{selected_model}", "rb") as file:
    model = pickle.load(file)

# Load preprocessor
with open('model/pipe.pkl','rb') as file:
    preprocessor = pickle.load(file)


def main():
    st.title('Cancer Survival Prediction')

    Age = st.number_input("Age")
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Protein1 = st.number_input('Protein1')
    Protein2 = st.number_input('Protein2')
    Protein3 = st.number_input('Protein3')
    Protein4 = st.number_input('Protein4')
    Tumour_Stage = st.selectbox('Tumor_stage', ['II', 'I', 'III'])
    ER_status = st.selectbox('ER_status', ['Positive', 'Negative'])
    PR_status = st.selectbox('PR_status', ['Positive', 'Negative'])
    Histology = st.selectbox('Histology', [
        'Infiltrating Ductal Carcinoma',
        'Infiltrating Lobular Carcinoma',
        'Mucinous Carcinoma'
    ])
    HER2_status = st.selectbox('HER2_status', ['Negative', 'Positive'])
    Surgery_type = st.selectbox('Surgery_type', [
        'Other', 'Lumpectomy', 'Modified Radical Mastectomy', 'Simple Mastectomy'
    ])

    survival_labels = {0: 'negative', 1: 'positive'}

    if st.button("Predict"):
        data = pd.DataFrame({
            'Age': [Age],
            'Gender': [Gender],
            'Protein1': [Protein1],
            'Protein2': [Protein2],
            'Protein3': [Protein3],
            'Protein4': [Protein4],
            'Tumour_Stage': [Tumour_Stage],
            'Histology': [Histology],
            'HER2 status': [HER2_status],
            'Surgery_type': [Surgery_type],
            'ER status': [ER_status],
            'PR status': [PR_status]
        })

        processed_data = preprocessor.transform(data)

        # Universal prediction
        prediction = model.predict(processed_data)

        # Handle different output formats safely
        if hasattr(prediction, "__len__"):
            prediction = prediction[0]

        prediction = int(prediction)

        st.success(f"Model: {selected_model}")
        st.success(f"Survival Prediction: {survival_labels[prediction]}")


if __name__ == "__main__":
    main()