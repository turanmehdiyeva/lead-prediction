import pickle
import numpy as np

def predict_single(costumer, dv, model):
    X = dv.transform([costumer])
    y_pred = model.predict_proba(X)[:,1]
    return y_pred[0]

costumer = {'what_is_your_current_occupation': 'Unemployed',
            'last_activity': 'Email Link Clicked',
            'lead_profile': 'Potential Lead',
            'last_notable_activity': 'Modified',
            'lead_source': 'Reference',
            'what_matters_most_to_you_in_choosing_a_course': 'Better Career Prospects',
            'lead_origin': 'Lead Add Form',
            'specialization': 'Services Excellence',
            'do_not_email': 'No',
            'totalvisits': 0.0,
            'total_time_spent_on_website': 0,
            'page_views_per_visit': 0.0}

with open('converting-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

prediction = predict_single(costumer, dv, model)

print('Prediction: ', prediction)
