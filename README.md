#  RiskGuard: Credit Default Predictor

## Overview
**RiskGuard** is a web-based decision support tool designed to help banks and financial institutions predict **credit card default risk**.  
The application uses a **tuned Random Forest classifier** trained on historical customer data to provide **real-time risk assessments**.

The model analyzes **six months of payment history** and customer profile information to estimate the probability of a customer defaulting on their next credit card payment.


##  Technical Details

- **Model**  
  - Scikit-learn Random Forest Classifier  
  - Hyperparameters optimized using `RandomizedSearchCV`

- **Preprocessing**  
  - `StandardScaler` applied to numerical features  
  - `OneHotEncoder` applied to categorical variables  

- **Feature Engineering**  
  - Six-month average bill amount  
  - Six-month average payment amount  
  - Credit utilisation ratio  
    ```
    Utilisation Ratio = Total Bill Amount / Credit Limit
    ```

- **Evaluation Metric**  
  - Optimized for **F1-score** to balance:
    - False negatives (missed defaulters leading to financial loss)
    - False positives (incorrectly flagged customers leading to lost opportunities)

---

##  File Structure

```text
.
├── app.py                    # Streamlit dashboard application
├── final_rf_model.pkl        # Trained Random Forest model
├── preprocessor.pkl          # Data preprocessing pipeline
├── top_15.pkl                # Selected feature names
├── top_15_indices.pkl        # Feature selection indices
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
```
 ## Business Logic

- **Threshold Slider**
Allows users to adjust the system’s risk appetite, controlling the trade-off between identifying more defaulters and minimizing false alarms.

- **Risk Classification Output**

   - Low Risk

   - High Risk

- **Key Risk Drivers**
The application highlights influential features (e.g., PAY_0, utilisation ratio) that contribute most to each individual prediction, improving transparency and interpretability.

## Business Impact

RiskGuard demonstrates how machine learning can support credit risk management by:

- Identifying high-risk customers early

- Reducing potential financial losses

- Enabling flexible, data-driven credit decision
