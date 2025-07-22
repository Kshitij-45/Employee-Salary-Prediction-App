import gradio as gr
import joblib
import pandas as pd

# Load the saved pipeline and label encoder
pipeline = joblib.load('model_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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

        pred_encoded = pipeline.predict(input_data)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        return f"🔎 **Predicted Income Group:** `{pred_label}`"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


# App layout using Blocks
with gr.Blocks(theme=gr.themes.Default(primary_hue="cyan")) as demo:

    gr.Image(value="logo employee.png", show_label=False, interactive=False, height=100, width=100)


    gr.Markdown("""
    # 🧠 Employee Salary Prediction  
    Predict whether an employee earns **>50K** or **<=50K** using an XGBoost model trained on census features.
    """)

    with gr.Row():
        with gr.Column():
            age = gr.Slider(17, 90, step=1, label="Age")
            fnlwgt = gr.Slider(10000, 150000, step=1000, label="Final Weight (fnlwgt)")
            educational_num = gr.Slider(1, 16, step=1, label="Educational Number")
            capital_gain = gr.Slider(0, 100000, step=500, label="Capital Gain")
            capital_loss = gr.Slider(0, 5000, step=100, label="Capital Loss")
            hours_per_week = gr.Slider(1, 99, step=1, label="Hours per Week")

        with gr.Column():
            workclass = gr.Dropdown(
                ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
                 'State-gov', 'Without-pay', 'Never-worked'], label="Workclass")
            marital_status = gr.Dropdown(
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'],
                label="Marital Status")
            relationship = gr.Dropdown(
                ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'], label="Relationship")
            race = gr.Dropdown(
                ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], label="Race")
            gender = gr.Dropdown(['Male', 'Female'], label="Gender")

    predict_btn = gr.Button("🔍 Predict")
    output_text = gr.Markdown()

    predict_btn.click(fn=predict_income,
                      inputs=[age, fnlwgt, educational_num, capital_gain, capital_loss, hours_per_week,
                              workclass, marital_status, relationship, race, gender],
                      outputs=output_text)

if __name__ == "__main__":
    demo.launch()

