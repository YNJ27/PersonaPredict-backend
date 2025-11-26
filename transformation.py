from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Time_spent_Alone: Hours spent alone daily (0–11).
# Stage_fear: Presence of stage fright (Yes/No).
# Social_event_attendance: Frequency of social events (0–10).
# Going_outside: Frequency of going outside (0–7).
# Drained_after_socializing: Feeling drained after socializing (Yes/No).
# Friends_circle_size: Number of close friends (0–15).
# Post_frequency: Social media post frequency (0–10).
# Personality: Target variable (Extrovert/Introvert).*

trf = ColumnTransformer([
    ('imputer1', SimpleImputer(strategy = 'constant', fill_value = 2), [1, 4]),           #Always use column indices (best practice), 
                                                                                          #especially if you are using pipelines having many column transformers as steps 
    ('imputer2', SimpleImputer(strategy = 'constant', fill_value = 100), [0, 2, 3, 5, 6])
], remainder = 'passthrough')

# y = y.replace({'Extrovert': 0, 'Introvert': 1})

threshold = 0.6070285052627968