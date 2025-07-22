import gradio as gr
import joblib
import pandas as pd

# Load the saved pipeline and label encoder
pipeline = joblib.load('model_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Feature columns
numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols = ['workclass', 'marital-status', 'relationship', 'race', 'gender']

# Prediction function
def predict_income(age, fnlwgt, educational_num, capital_gain, capital_loss, hours_per_week,
                   workclass, marital_status, relationship, race, gender):
    try:
        input_data = pd.DataFrame([{
            'age': age,
            'fnlwgt': fnlwgt,
            'educational-num': educational_num,
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week,
            'workclass': workclass,
            'marital-status': marital_status,
            'relationship': relationship,
            'race': race,
            'gender': gender
        }])
        
        print("Input DataFrame:")
        print(input_data)

        pred_encoded = pipeline.predict(input_data)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        return f"Predicted Income Group: {pred_label}"

    except Exception as e:
        # Print full error to Hugging Face logs
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


# Gradio interface with sliders and dropdowns
demo = gr.Interface(
    fn=predict_income,
    inputs=[
        gr.Slider(minimum=17, maximum=90, step=1, label="Age"),
        gr.Slider(minimum=10000, maximum=150000, step=1000, label="Final Weight (fnlwgt)"),
        gr.Slider(minimum=1, maximum=16, step=1, label="Educational Number"),
        gr.Slider(minimum=0, maximum=100000, step=500, label="Capital Gain"),
        gr.Slider(minimum=0, maximum=5000, step=100, label="Capital Loss"),
        gr.Slider(minimum=1, maximum=99, step=1, label="Hours per Week"),
        gr.Dropdown(choices=[
            'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
            'State-gov', 'Without-pay', 'Never-worked'
        ], label="Workclass"),
        gr.Dropdown(choices=[
            'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'
        ], label="Marital Status"),
        gr.Dropdown(choices=[
            'Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'
        ], label="Relationship"),
        gr.Dropdown(choices=[
            'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
        ], label="Race"),
        gr.Dropdown(choices=['Male', 'Female'], label="Gender")
    ],
    outputs="text",
    title="Income Group Prediction",
    description="Adjust the sliders and dropdowns to predict whether income is >50K or <=50K."
)

if __name__ == "__main__":
    demo.launch()
