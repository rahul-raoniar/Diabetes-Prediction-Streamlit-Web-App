

import streamlit as st

# Load ML packages
import joblib
import os

# Load EDA Packages
import numpy as np

attrib_info = """
#### Attribute Information:
    - Age 1.20-65
    - Sex 1. Male, 2.Female
    - Polyuria 1.Yes, 2.No.
    - Polydipsia 1.Yes, 2.No.
    - sudden weight loss 1.Yes, 2.No.
    - weakness 1.Yes, 2.No.
    - Polyphagia 1.Yes, 2.No.
    - Genital thrush 1.Yes, 2.No.
    - visual blurring 1.Yes, 2.No.
    - Itching 1.Yes, 2.No.
    - Irritability 1.Yes, 2.No.
    - delayed healing 1.Yes, 2.No.
    - partial paresis 1.Yes, 2.No.
    - muscle stiffness 1.Yes, 2.No.
    - Alopecia 1.Yes, 2.No.
    - Obesity 1.Yes, 2.No.
    - Class 1.Positive, 2.Negative.

"""

label_dict = {"No": 0, "Yes": 1}
gender_map = {"Female": 0, "Male": 1}
target_label_map = {"Negative":0, "Positive": 1}

def get_fvalue(val):
    feature_dict = {"No": 0, "Yes": 1}
    for key, value in feature_dict.items():
        if val == key:
            return value
    
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value
        
@st.cache
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


def run_ml_app():
    st.subheader("Machine Learning Prediction")
    # st.write("It is working")
    # st.success("It is cool")
    
    with st.beta_expander("Attribute Info."):
        st.markdown(attrib_info)
        
    # Layout
    col1, col2 = st.beta_columns(2)
    
    with col1:
       age = st.number_input("Age",10,100)
       gender = st.radio("Gender",("Female","Male"))
       polyuria = st.radio("Polyuria",["No","Yes"])
       polydipsia = st.radio("Polydipsia",["No","Yes"]) 
       sudden_weight_loss = st.radio("Sudden_weight_loss",["No","Yes"])
       weakness = st.radio("weakness",["No","Yes"]) 
       polyphagia = st.radio("polyphagia",["No","Yes"]) 
       genital_thrush = st.radio("Genital_thrush",["No","Yes"]) 
		
	
    with col2:
       visual_blurring = st.radio("Visual_blurring",["No","Yes"])
       itching = st.radio("itching",["No","Yes"]) 
       irritability = st.radio("irritability",["No","Yes"]) 
       delayed_healing = st.radio("delayed_healing",["No","Yes"]) 
       partial_paresis = st.radio("Partial_paresis",["No","Yes"])
       muscle_stiffness = st.radio("muscle_stiffness",["No","Yes"]) 
       alopecia = st.radio("alopecia",["No","Yes"]) 
       obesity = st.radio("obesity",["No","Yes"])
    
    
    with st.beta_expander("Your Selected Options"):
        result = {'age':age,
		'gender':gender,
		'polyuria':polyuria,
		'polydipsia':polydipsia,
		'sudden_weight_loss':sudden_weight_loss,
		'weakness':weakness,
		'polyphagia':polyphagia,
		'genital_thrush':genital_thrush,
		'visual_blurring':visual_blurring,
		'itching':itching,
		'irritability':irritability,
		'delayed_healing':delayed_healing,
		'partial_paresis':partial_paresis,
		'muscle_stiffness':muscle_stiffness,
		'alopecia':alopecia,
		'obesity':obesity}
        st.write(result)
        
        encoded_result = []
        for i in result.values():
            if type(i) == int:
                encoded_result.append(i)
            elif i in ["Female", "Male"]:
                res = get_value(i, gender_map)
                encoded_result.append(res)
            else:
                encoded_result.append(get_fvalue(i))

        st.write(encoded_result)
        
    with st.beta_expander("Model Prediction"):
        single_sample = np.array(encoded_result).reshape(1, -1)
        st.write(single_sample)
        
        model = load_model("models/logistic_regression_model_diabetes_21_oct_2020.pkl")
        prediction = model.predict(single_sample)
        pred_prob = model.predict_proba(single_sample)
        
        if prediction == 1:
            st.warning("Positive Risk: {}".format(prediction[0]))
            pred_probability_score = {"Negative DM Risk (%)": round(pred_prob[0][0]*100, 2), 
                                      "Positive DM Risk (%)": round(pred_prob[0][1]*100, 2)}
            st.write(pred_probability_score)
            
        else:
            st.success("Negative Risk: {}".format(prediction[0]))
            pred_probability_score = {"Negative DM Risk (%)": round(pred_prob[0][0]*100, 2), 
                                      "Positive DM Risk (%)": round(pred_prob[0][1]*100, 2)}
            st.write(pred_probability_score)
        
    
    
    
    