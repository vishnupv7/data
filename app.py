
# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Step 2: Load CSV files from local paths
# Ensure these CSV files are located in the same directory as your script, or adjust the paths accordingly.
confusion_matrix_data = pd.read_csv('confusion_matrix.csv')
feature_importance_data = pd.read_csv('feature_importance.csv')
model_performance_data = pd.read_csv('model_performance_metrics.csv')

# Step 3: Clean up the confusion matrix data
confusion_matrix_data_cleaned = confusion_matrix_data.set_index('Unnamed: 0')
confusion_matrix_data_cleaned.columns = ['No Failure', 'Failure']

# Step 4: Clean up the model performance data
model_performance_data_cleaned = model_performance_data.rename(columns={'Unnamed: 0': 'Class'}).set_index('Class')

# Step 5: Plot Confusion Matrix
st.title("Machine Failure Prediction Dashboard")
st.header("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(confusion_matrix_data_cleaned, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Step 6: Plot Feature Importance
st.header("Feature Importance")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_data, palette='viridis', ax=ax)
st.pyplot(fig)

# Step 7: Plot Model Performance Metrics (Precision, Recall, F1-Score)
st.header("Model Performance Metrics")
fig, ax = plt.subplots(figsize=(8, 5))
model_performance_data_cleaned[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax)
ax.set_title('Precision, Recall, F1-Score by Class')
ax.set_xticklabels(['No Failure (0)', 'Failure (1)'], rotation=0)
st.pyplot(fig)

# Step 8: Display Model Accuracy (hardcoded for now, replace with your actual accuracy metric)
accuracy = 88  # Example accuracy
st.metric(label="Model Accuracy", value=f"{accuracy}%")
