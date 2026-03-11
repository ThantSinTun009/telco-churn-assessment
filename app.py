from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load pipeline
try:
    final_pipeline = joblib.load('final_tuned_lgbm_pipeline.pkl')
except Exception as e:
    print("Error loading pipeline:", e)
    final_pipeline = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Telecom Churn Prediction - GlobalNet</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #f0f4f8; margin:0; }
        .header {
            background-color: #0052cc; /* GlobalNet blue */
            color:white;
            text-align:center;
            padding:20px;
            display:flex;
            align-items:center;
            justify-content:center;
            gap:10px;
        }
        .header img { height:40px; }
        .header h1 { font-size:1.5rem; margin:0; }

        .container { display:flex; flex-wrap:wrap; gap:20px; padding:20px; }

        /* Sidebar for company info */
        .sidebar { flex:1 1 250px; background:white; padding:20px; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,0.1); }
        .sidebar h3 { margin-bottom:10px; color:#0052cc; }
        .sidebar p { margin-bottom:10px; color:#333; font-size:0.95rem; }

        /* Main content */
        .main { flex:3 1 600px; background:white; padding:20px; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,0.1); }

        h2 { text-align:center; color:#0052cc; margin-bottom:20px; }

        /* Collapsible sections */
        .collapsible {
            background-color: #0052cc;
            color: white;
            cursor: pointer;
            padding: 12px;
            width: 100%;
            border: none;
            text-align: left;
            font-size: 1.1rem;
            border-radius: 8px;
            margin-bottom:5px;
            transition:0.3s;
        }
        .collapsible:hover { background-color: #0041a8; }
        .content {
            padding: 0 15px;
            display: none;
            overflow: hidden;
            background-color: #e6f0ff;
            border-radius: 8px;
            margin-bottom: 15px;
            border:1px solid #cce0ff;
        }

        label { display:block; margin:8px 0 2px; font-weight:600; color:#333; }
        input[type=number], select { width:100%; padding:8px; margin-bottom:10px; border-radius:5px; border:1px solid #ccc; }

        input[type=submit] {
            background-color:#d32f2f; /* Red accent from logo */
            color:white;
            padding:12px;
            border:none;
            border-radius:8px;
            width:100%;
            font-size:1rem;
            cursor:pointer;
            transition:0.3s;
        }
        input[type=submit]:hover { background-color:#b71c1c; }

        .result { padding:20px; margin-top:20px; border-radius:10px; font-size:1.1rem; font-weight:bold; text-align:center; }
        .churn { background-color: #ff4d4d; color:white; }
        .safe { background-color: #0052cc; color:white; }

        @media(max-width:900px){ .container{ flex-direction:column; } }
    </style>
</head>
<body>
    <div class="header">
        <img src="../static/globalnet.jpg" alt="GlobalNet Logo">
        <h1>Telecom Churn Prediction Dashboard</h1>
    </div>

    <div class="container">
        <div class="sidebar">
            <h3>About GlobalNet</h3>
            <p>GlobalNet is a leading technology company providing innovative telecom solutions worldwide. We focus on delivering high-quality services and cutting-edge technology to enhance connectivity for our customers.</p>
            <h3>Our Mission</h3>
            <p>Empower individuals and businesses with reliable and advanced communication tools for seamless connectivity.</p>
            <h3>Contact</h3>
            <p>Email: info-marketing@globalnetmm.com</p>
            <p>Phone: +95 9 788 500 072</p>
        </div>

        <div class="main">
            <h2>Enter Customer Details</h2>
            <form method="post" action="/predict_form">
                <!-- Personal Info -->
                <button type="button" class="collapsible">Personal Info</button>
                <div class="content">
                    <label>Senior Citizen:</label>
                    <input type="number" name="SeniorCitizen" min="0" max="1" value="0">
                    <label>Gender:</label>
                    <select name="gender"><option value="1">Male</option><option value="0">Female</option></select>
                    <label>Partner:</label>
                    <select name="Partner"><option value="1">Yes</option><option value="0">No</option></select>
                    <label>Dependents:</label>
                    <select name="Dependents"><option value="1">Yes</option><option value="0">No</option></select>
                </div>

                <!-- Subscription Details -->
                <button type="button" class="collapsible">Subscription Details</button>
                <div class="content">
                    <label>Internet Service:</label>
                    <select name="InternetService"><option value="DSL">DSL</option><option value="Fiber optic">Fiber optic</option><option value="No">No</option></select>
                    <label>Contract:</label>
                    <select name="Contract"><option value="Month-to-month">Month-to-month</option><option value="One year">One year</option><option value="Two year">Two year</option></select>
                    <label>Tenure Group:</label>
                    <select name="tenure_group"><option value="0-1yr">0-1yr</option><option value="1-2yr">1-2yr</option><option value="2-4yr">2-4yr</option><option value="4-6yr">4-6yr</option></select>
                    <label>Payment Method:</label>
                    <select name="PaymentMethod"><option value="Electronic check">Electronic check</option><option value="Mailed check">Mailed check</option><option value="Credit card (automatic)">Credit card (automatic)</option><option value="Bank transfer">Bank transfer</option></select>
                    <label>Tenure (months):</label>
                    <input type="number" name="tenure" min="0" value="12">
                </div>

                <!-- Services & Billing -->
                <button type="button" class="collapsible">Services & Billing</button>
                <div class="content">
                    <label>Phone Service:</label>
                    <select name="PhoneService"><option value="1">Yes</option><option value="0">No</option></select>
                    <label>Multiple Lines:</label>
                    <select name="MultipleLines"><option value="1">Yes</option><option value="0">No</option></select>
                    <label>Online Security:</label>
                    <select name="OnlineSecurity"><option value="1">Yes</option><option value="0">No</option></select>
                    <label>Online Backup:</label>
                    <select name="OnlineBackup"><option value="1">Yes</option><option value="0">No</option></select>
                    <label>Device Protection:</label>
                    <select name="DeviceProtection"><option value="1">Yes</option><option value="0">No</option></select>
                    <label>Tech Support:</label>
                    <select name="TechSupport"><option value="1">Yes</option><option value="0">No</option></select>
                    <label>Streaming TV:</label>
                    <select name="StreamingTV"><option value="1">Yes</option><option value="0">No</option></select>
                    <label>Streaming Movies:</label>
                    <select name="StreamingMovies"><option value="1">Yes</option><option value="0">No</option></select>
                    <label>Paperless Billing:</label>
                    <select name="PaperlessBilling"><option value="1">Yes</option><option value="0">No</option></select>
                    <label>Monthly Charges:</label>
                    <input type="number" name="MonthlyCharges" step="0.01" value="50">
                    <label>Total Charges:</label>
                    <input type="number" name="TotalCharges" step="0.01" value="600">
                    <label>Charge per tenure:</label>
                    <input type="number" name="charge_per_tenure" step="0.01" value="50">
                    <label>Avg monthly spend:</label>
                    <input type="number" name="avg_monthly_spend" step="0.01" value="50">
                    <label>Number of services:</label>
                    <input type="number" name="num_services" min="1" value="3">
                </div>

                <input type="submit" value="Predict">
            </form>

            {% if prediction %}
                <div class="result {% if churn %}churn{% else %}safe{% endif %}">
                    <h2>Prediction Result</h2>
                    <p>{{ prediction }}</p>
                </div>
            {% endif %}
        </div>
    </div>

    <script>
        var coll = document.getElementsByClassName("collapsible");
        for(var i=0;i<coll.length;i++){
            coll[i].addEventListener("click",function(){
                this.classList.toggle("active");
                var content=this.nextElementSibling;
                content.style.display = content.style.display==="block"?"none":"block";
            });
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict_form', methods=['POST'])
def predict_form():
    if final_pipeline is None:
        return "Pipeline not loaded."

    try:
        form_data = request.form.to_dict()
        numeric_fields = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges',
                          'charge_per_tenure','avg_monthly_spend','num_services']
        binary_fields = ['gender','Partner','Dependents','PhoneService','MultipleLines',
                         'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                         'StreamingTV','StreamingMovies','PaperlessBilling']

        for field in numeric_fields + binary_fields:
            form_data[field] = float(form_data[field])

        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        tenure_group_map = {"0-1yr": 0, "1-2yr": 1, "2-4yr": 2, "4-6yr": 3}

        form_data['Contract'] = contract_map.get(form_data['Contract'], 0)
        form_data['tenure_group'] = tenure_group_map.get(form_data['tenure_group'], 0)

        user_df = pd.DataFrame([form_data])
        y_prob = final_pipeline.predict_proba(user_df)[:,1]
        y_pred = (y_prob > 0.35).astype(int)
        message = "⚠️ User likely to churn." if y_pred[0]==1 else "✅ User likely safe."
        return render_template_string(HTML_TEMPLATE, prediction=message, churn=(y_pred[0]==1))
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {e}", churn=False)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    if final_pipeline is None:
        return jsonify({"error": "Pipeline not loaded"}), 500

    try:
        data = request.get_json()

        numeric_fields = [
            'SeniorCitizen','tenure','MonthlyCharges','TotalCharges',
            'charge_per_tenure','avg_monthly_spend','num_services'
        ]

        binary_fields = [
            'gender','Partner','Dependents','PhoneService','MultipleLines',
            'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
            'StreamingTV','StreamingMovies','PaperlessBilling'
        ]

        for field in numeric_fields + binary_fields:
            data[field] = float(data[field])

        contract_map = {"Month-to-month":0,"One year":1,"Two year":2}
        tenure_group_map = {"0-1yr":0,"1-2yr":1,"2-4yr":2,"4-6yr":3}

        data['Contract'] = contract_map.get(data['Contract'],0)
        data['tenure_group'] = tenure_group_map.get(data['tenure_group'],0)

        user_df = pd.DataFrame([data])

        y_prob = final_pipeline.predict_proba(user_df)[:,1]
        y_pred = (y_prob > 0.35).astype(int)

        result = {
            "churn_probability": float(y_prob[0]),
            "prediction": int(y_pred[0]),
            "message": "User likely to churn" if y_pred[0]==1 else "User likely safe"
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)