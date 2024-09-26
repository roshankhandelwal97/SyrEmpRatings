#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
data = pd.read_csv('syracuse_requests.csv')
print("Dataset loaded successfully.")

# Convert 'Created_at_local' and 'Closed_at_local' into datetime objects
data['Created_at_local'] = pd.to_datetime(data['Created_at_local'], errors='coerce')
data['Closed_at_local'] = pd.to_datetime(data['Closed_at_local'], errors='coerce')
print("Date columns converted to datetime format.")

# Check for any conversion issues and basic dataframe info
print(data.info())

# Display the first few rows to confirm changes
print(data.head())


# In[2]:


import matplotlib.pyplot as plt

# Group by month and count the number of issues reported and resolved
monthly_reported = data['Created_at_local'].dt.to_period('M').value_counts().sort_index()
monthly_closed = data['Closed_at_local'].dt.to_period('M').value_counts().sort_index()

# Plotting the data
plt.figure(figsize=(14, 7))
plt.plot(monthly_reported.index.astype(str), monthly_reported.values, label='Reported Issues', marker='o')
plt.plot(monthly_closed.index.astype(str), monthly_closed.values, label='Resolved Issues', marker='x')
plt.title('Monthly Trend of Issues Reported and Resolved')
plt.xlabel('Month')
plt.ylabel('Number of Issues')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print out the first few entries of each series to check
print("Monthly Reported Issues:\n", monthly_reported.head())
print("Monthly Closed Issues:\n", monthly_closed.head())


# It appears there are seasonal patterns or perhaps specific events influencing the spikes 
# in both reported and resolved issues.

# In[3]:


# Calculate the monthly average time taken to resolve issues
data['Resolution_Time'] = (data['Closed_at_local'] - data['Created_at_local']).dt.total_seconds() / 3600  # in hours
monthly_resolution_time = data.groupby(data['Created_at_local'].dt.to_period('M'))['Resolution_Time'].mean()

# Plot the average resolution time by month
plt.figure(figsize=(14, 7))
plt.plot(monthly_resolution_time.index.astype(str), monthly_resolution_time.values, label='Average Resolution Time (Hours)', marker='o', color='green')
plt.title('Average Monthly Resolution Time for Issues')
plt.xlabel('Month')
plt.ylabel('Average Resolution Time (Hours)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print out the first few entries to check the calculation
print("Average Monthly Resolution Time:\n", monthly_resolution_time.head())


# Observations:
# There's a peak in resolution time around mid-2021, particularly from July to September, indicating that issues took longer to resolve during this period.
# The resolution time steadily decreases afterwards, which could suggest process improvements, seasonal variation in issue types, or perhaps a decrease in the complexity or severity of issues.

# In[4]:


import numpy as np
import seaborn as sns
# Combine the data into a single DataFrame for analysis
analysis_df = pd.DataFrame({
    'Reported_Issues': monthly_reported,
    'Resolved_Issues': monthly_closed,
    'Average_Resolution_Time': monthly_resolution_time
})

# Drop any months with missing data to ensure accurate correlation calculation
analysis_df.dropna(inplace=True)

# Calculate and print the correlation matrix
correlation_matrix = analysis_df.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Reported and Resolved Issues with Resolution Time')
plt.show()


# Key Observations:
# High Correlation Between Reported and Resolved Issues: A correlation of 0.95 indicates that there is a very strong relationship between the number of issues reported and the number resolved. This suggests that as more issues are reported, a similar increase in issue resolutions follows.
# 
# Reported Issues and Resolution Time: The correlation of -0.44 suggests that as the number of reported issues increases, the average resolution time slightly decreases. This could imply that with higher volumes, processes might be optimized or prioritized to handle the increase.
# 
# Resolved Issues and Resolution Time: The stronger negative correlation of -0.56 indicates a more pronounced effect of increased resolutions on reducing the average resolution time. This could be due to streamlined processes or more resources being allocated as the volume of resolutions increases.

# In[ ]:





# In[5]:


# Calculate the resolution time for each issue in hours
data['Resolution_Time'] = (data['Closed_at_local'] - data['Created_at_local']).dt.total_seconds() / 3600

# Group the data by 'Agency_Name' and calculate the average resolution time
agency_performance = data.groupby('Agency_Name')['Resolution_Time'].mean().sort_values()

# Display the average resolution time per agency
print("Average Resolution Time by Agency (in hours):\n", agency_performance)


# In[6]:


import matplotlib.pyplot as plt

# Plotting
plt.figure(figsize=(12, 8))
agency_performance.plot(kind='barh', color='skyblue')
plt.xlabel('Average Resolution Time (Hours)')
plt.ylabel('Agency')
plt.title('Average Resolution Time by Agency')
plt.grid(True)
plt.show()


# In[ ]:





# The bar chart and the results clearly show significant differences in the average resolution time among the various agencies, with "Green Spaces, Trees & Public Utilities" taking the longest and "Water & Sewage" the shortest. 
# 
# Insights and Recommendations:
# Significant Variation in Resolution Times: Agencies like "Green Spaces, Trees & Public Utilities" and "Feedback to the City" exhibit notably higher resolution times compared to others. This could be due to the nature of the issues they deal with, which may be more complex or less prioritized.
# 
# Potential Areas for Improvement: Agencies with longer average resolution times might benefit from process audits and efficiency improvements. It may also be helpful to investigate the types of issues these agencies are addressing to see if additional resources or specific changes in process are necessary.
# 
# Resource Allocation: Agencies with faster resolution times might serve as models for best practices. Their strategies could potentially be adapted to improve performance in slower agencies.

# In[7]:


# Select top 3 agencies with the highest resolution times for detailed analysis
top_agencies = agency_performance.tail(3).index.tolist()

# Filter the data for these agencies
top_agency_data = data[data['Agency_Name'].isin(top_agencies)]

# Print the selected agencies
print("Selected Agencies for Detailed Analysis:", top_agencies)


# In[8]:


# Group by agency and issue category, then calculate the average resolution time
issue_type_performance = top_agency_data.groupby(['Agency_Name', 'Category'])['Resolution_Time'].mean().sort_values()

# Display the average resolution time by issue type within these agencies
print("Average Resolution Time by Issue Type within Selected Agencies:\n", issue_type_performance)


# In[9]:


import matplotlib.pyplot as plt

# Plotting
plt.figure(figsize=(12, 8))
issue_type_performance.unstack().plot(kind='barh', figsize=(14, 7))
plt.xlabel('Average Resolution Time (Hours)')
plt.ylabel('Agency')
plt.title('Average Resolution Time by Issue Type within Selected Agencies')
plt.legend(title='Issue Type')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[10]:


import matplotlib.pyplot as plt

# Filter data for each agency and prepare for plotting
data_feedback = top_agency_data[top_agency_data['Agency_Name'] == 'Feedback to the City'].groupby('Category')['Resolution_Time'].mean()
data_streets = top_agency_data[top_agency_data['Agency_Name'] == 'Streets, Sidewalks & Transportation'].groupby('Category')['Resolution_Time'].mean()
data_green_spaces = top_agency_data[top_agency_data['Agency_Name'] == 'Green Spaces, Trees & Public Utilities'].groupby('Category')['Resolution_Time'].mean()

# Create subplots
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 18), sharex=True)

# Plot for Feedback to the City
ax[0].barh(data_feedback.index, data_feedback.values, color='skyblue')
ax[0].set_title('Feedback to the City - Issue Resolution Times')
ax[0].set_xlabel('Average Resolution Time (Hours)')
ax[0].set_ylabel('Issue Type')

# Plot for Streets, Sidewalks & Transportation
ax[1].barh(data_streets.index, data_streets.values, color='lightgreen')
ax[1].set_title('Streets, Sidewalks & Transportation - Issue Resolution Times')
ax[1].set_xlabel('Average Resolution Time (Hours)')
ax[1].set_ylabel('Issue Type')

# Plot for Green Spaces, Trees & Public Utilities
ax[2].barh(data_green_spaces.index, data_green_spaces.values, color='salmon')
ax[2].set_title('Green Spaces, Trees & Public Utilities - Issue Resolution Times')
ax[2].set_xlabel('Average Resolution Time (Hours)')
ax[2].set_ylabel('Issue Type')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:





# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('syracuse_requests.csv')
data.info()



# In[12]:


missing_data = data.isnull().sum().sort_values(ascending=False)
missing_percentage = (missing_data / len(data)) * 100
missing_report = pd.DataFrame({'Missing Count': missing_data, 'Missing Percentage': missing_percentage})

print(missing_report)


# In[13]:


import pandas as pd

# Function to clean up the address
def clean_address(address):
    if not isinstance(address, str):
        return None
    address = address.lower()  # Convert to lower case
    address = ''.join([i for i in address if not i.isdigit()])  # Remove digits
    address = address.replace('syracuse', '')  # Remove the word 'syracuse'
    address = address.replace(',', '')  # Remove commas
    address = address.replace('new york', '')  # Remove "new york"
    address = address.replace('ny', '')  # Remove "ny"
    address = address.replace('usa', '')  # Remove "usa"
    address = address.replace('united states', '')  # Remove "united states"
    address = address.replace('united states of america', '')  # Remove "united states of america"
    address = address.replace('-', '')  # Remove hyphens
    address = ' '.join(address.split())  # Remove extra spaces and reformat
    return address.strip()

# Apply the function to your DataFrame
data['Street_Name'] = data['Address'].apply(clean_address)


data['Street_Name'].head(50)


# In[14]:


# Dropping rows where Minutes_to_closed is missing
data_cleaned = data.dropna(subset=['Minutes_to_closed'])

# Dropping columns with a high proportion of missing values
data_cleaned = data_cleaned.drop(['Summary','X', 'Y','Minutes_to_acknowledged', 'Id', 'URL', 'ObjectId', 'Closed_at_local','Acknowledged_at_local', 'Address', 'Description', 'Export_tagged_places'], axis=1)

# Verifying the shape of the cleaned dataset
print(data_cleaned.shape)


# In[15]:


data_cleaned['Minutes_to_closed'].isnull().sum()
data_cleaned.head(10)


# In[16]:


# Converting 'Created_at_local' to datetime format
data_cleaned['Created_at_local'] = pd.to_datetime(data_cleaned['Created_at_local'])

# Verifying the changes
print(data_cleaned.head())


# In[17]:


# Converting 'Created_at_local' to datetime format
data_cleaned['Created_at_local'] = pd.to_datetime(data_cleaned['Created_at_local'], format='%m/%d/%Y - %I:%M%p')

# Feature engineering: Extracting date and time features
data_cleaned['Hour_of_day'] = data_cleaned['Created_at_local'].dt.hour
data_cleaned['Day_of_week'] = data_cleaned['Created_at_local'].dt.dayofweek
data_cleaned['Month'] = data_cleaned['Created_at_local'].dt.month

# Dropping the original 'Created_at_local' column
data_cleaned = data_cleaned.drop('Created_at_local', axis=1)

# Verifying the changes
data_cleaned.head()


# In[18]:


# Dropping rows where any of the important columns contain missing values
columns_with_missing = [ 'Lat', 'Lng', 'Sla_in_hours', 'Assignee_name']
data_cleaned = data_cleaned.dropna(subset=columns_with_missing)

# Verifying that no missing values remain
missing_data = data_cleaned.isnull().sum().sort_values(ascending=False)
missing_percentage = (missing_data / len(data_cleaned)) * 100
missing_report = pd.DataFrame({'Missing Count': missing_data, 'Missing Percentage': missing_percentage})

print(missing_report)


# In[19]:


# Group the data by 'Category' and calculate mean, min, and max for 'Minutes_to_closed'
category_stats = data_cleaned.groupby('Category')['Minutes_to_closed'].agg(['mean', 'min', 'max'])

# Resetting the index to make 'Category' a column again for better display
category_stats.reset_index(inplace=True)

# Renaming columns for clarity
category_stats.columns = ['Category', 'Average Minutes to Closed', 'Minimum Minutes to Closed', 'Maximum Minutes to Closed']

# Display the DataFrame
print("Statistics for Minutes to Closed for Each Category:")
category_stats


# In[20]:


# import pandas as pd
# import numpy as np

# # Calculate the first quartile and third quartile
# Q1 = data_cleaned['Minutes_to_closed'].quantile(0.25)
# Q3 = data_cleaned['Minutes_to_closed'].quantile(0.75)
# IQR = Q3 - Q1

# # Define bounds for outliers
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Identifying outliers
# outliers = data_cleaned[(data_cleaned['Minutes_to_closed'] < lower_bound) | (data_cleaned['Minutes_to_closed'] > upper_bound)]

# # Display outliers
# print(f"Lower Bound for Outliers: {lower_bound}")
# print(f"Upper Bound for Outliers: {upper_bound}")
# print(f"Number of Outliers: {outliers.shape[0]}")
# print("Outliers in 'Minutes_to_closed':")
# outliers


# In[21]:


data_cleaned.info()


# In[22]:


# Count total rows in the dataset
total_rows = data_cleaned.shape[0]

# Count rows where 'Minutes_to_closed' is negative
negative_minutes_count = (data_cleaned['Minutes_to_closed'] < 0).sum()

# Display the counts
print(f"Total number of rows in the dataset: {total_rows}")
print(f"Number of rows with negative 'Minutes to Closed': {negative_minutes_count}")


# In[23]:


# Extract unique values from categorical columns

# Assuming `data_cleaned` contains your preprocessed dataset before one-hot encoding
agencies = data_cleaned['Agency_Name'].unique().tolist()
assignees = data_cleaned['Assignee_name'].unique().tolist()
categories = data_cleaned['Category'].unique().tolist()
report_sources = data_cleaned['Report_Source'].unique().tolist()

# Display the unique values for each categorical variable

print("Agencies:", agencies)
print("Assignees:", assignees)
print("Categories:", categories)
print("Report Sources:", report_sources)


# In[24]:


import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# One-hot encoding for categorical columns
categorical_columns = ['Street_Name', 'Agency_Name', 'Assignee_name', 'Category', 'Report_Source']
print("Starting one-hot encoding...")
data_cleaned = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)
print("One-hot encoding completed.")

# Splitting the data into features (X) and target (y)
print("Splitting data into training and testing sets...")
X = data_cleaned.drop('Minutes_to_closed', axis=1)
y = data_cleaned['Minutes_to_closed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split completed.")

# Apply Standard Scaler (Scaling the data)
print("Applying Standard Scaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale the whole dataset for cross-validation
print("Scaling completed.")

# Save the scaler and columns for future use
print("Saving scaler and feature columns...")
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns, 'columns.pkl')  # Save the feature columns used during training
print("Scaler and columns saved.")

# Prepare the Random Forest Regressor
print("Preparing Random Forest Regressor...")
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)

# Generate cross-validated predictions for the entire dataset
print("Generating cross-validated predictions...")
cv_predictions = cross_val_predict(rf_model, X_scaled, y, cv=5)  # Using 5-fold cross-validation
data_cleaned['CV_Expected_Minutes_to_Close'] = cv_predictions
print("Cross-validated predictions added to the DataFrame.")

# Train the model on the entire dataset for deployment (optional based on use case)
rf_model.fit(X_scaled, y)
print("Model training completed on entire dataset.")

# Save the trained model for future use
print("Saving trained model...")
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Model saved.")

# Optionally, you might still want to evaluate on a hold-out set
y_pred = rf_model.predict(X_scaled)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(f"Random Forest - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
print("Model evaluation completed.")


# In[25]:


# Get feature importance from the RandomForest model
importances = rf_model.feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plotting the top 20 important features
plt.figure(figsize=(10, 6))
feature_importance_df.head(20).plot(kind='bar', x='Feature', y='Importance', legend=False)
plt.title('Top 20 Feature Importances')
plt.show()


# In[30]:


import matplotlib.pyplot as plt

# Convert minutes to days
actual_days = y / 1440
predicted_days = cv_predictions / 1440

# Sample data
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(np.arange(len(actual_days)), size=100, replace=False)
actual_sample_days = actual_days.iloc[sample_indices]
predicted_sample_days = predicted_days[sample_indices]

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(actual_sample_days)), actual_sample_days, color='blue', label='Actual Values')
plt.scatter(np.arange(len(predicted_sample_days)), predicted_sample_days, color='orange', label='Random Forest Predictions')
plt.title('Random Forest: Actual vs Predicted Values (100 Random Points)')
plt.xlabel('Index')
plt.ylabel('Days to Close')
plt.legend()
plt.show()


# In[27]:


# Saving the DataFrame with the cross-validated predictions to a CSV file
data_cleaned.to_csv('data_with_predictions.csv', index=False)
print("DataFrame saved as 'data_with_predictions.csv'.")


# In[28]:


data_cleaned.head()


# In[31]:


data_cleaned.info()


# In[34]:


data = pd.read_csv('data_cleaned_before_pred.csv')
data.info()


# In[35]:


# Assuming cv_predictions is an array or series containing your model's cross-validated predictions
data['CV_Expected_Minutes_to_Close'] = cv_predictions

# Save this enhanced DataFrame
data.to_csv('enhanced_data.csv', index=False)


# In[37]:


np.random.seed(42)  # For reproducibility
random_indices = np.random.choice(data.index, size=50, replace=False)
sample_data = data.loc[random_indices]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(sample_data.index, sample_data['Minutes_to_closed'], color='blue', label='Actual Minutes to Close', alpha=0.6)
plt.scatter(sample_data.index, sample_data['CV_Expected_Minutes_to_Close'], color='red', label='Predicted Minutes to Close', alpha=0.6)
plt.title('Comparison of Actual and Predicted Minutes to Close (50 Random Points)')
plt.xlabel('Index')
plt.ylabel('Minutes to Close')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid for Random Forest
# param_grid = {
#     'n_estimators': [100, 200, 300],  # Number of trees in the forest
#     'max_depth': [2,5,7,10],  # Maximum number of levels in each decision tree
#     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required at each leaf node
# }

# # Create a GridSearchCV object
# grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
#                            param_grid=param_grid,
#                            cv=3,  # Number of folds in cross-validation
#                            verbose=2,  # Controls the verbosity: the higher, the more messages
#                            n_jobs=-1,  # Number of jobs to run in parallel (-1 means using all processors)
#                            scoring='neg_mean_squared_error')  # Can choose other metrics such as 'r2'

# # Fit the grid search to the data
# print("Starting grid search...")
# grid_search.fit(X_train_scaled, y_train)
# print("Grid search completed.")

# # Best parameters and best score
# print("Best parameters found:")
# for param, value in grid_search.best_params_.items():
#     print(f"{param}: {value}")

# print("Best score (negative MSE) from grid search:", grid_search.best_score_)

# # Use the best estimator to make predictions
# print("Evaluating the best model from grid search...")
# best_rf_model = grid_search.best_estimator_
# y_pred = best_rf_model.predict(X_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print(f"Optimized Random Forest - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

# # Save the best model for future use
# print("Saving the optimized model to 'optimized_random_forest_model.pkl'...")
# joblib.dump(best_rf_model, 'optimized_random_forest_model.pkl')
# print("Optimized model saved successfully.")


# In[ ]:





# In[1]:


import joblib
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# Load the model, scaler, and columns
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')


# Define original input columns before one-hot encoding
original_columns = ['Street_Name', 'Agency_Name', 'Assignee_name', 'Category', 'Report_Source','Day_of_week', 'Month', 'Hour_of_day',  'Request_type']
additional_columns = ['Lat', 'Lng', 'Sla_in_hours']  

# Create widgets for the categorical features with free text input for more flexibility
inputs = {col: widgets.Text(description=col + ':') for col in original_columns}

# Create widgets for numerical inputs
inputs.update({col: widgets.FloatText(description=col + ':', value=0.0) for col in additional_columns})

# Button to make predictions
predict_btn = widgets.Button(description="Predict Minutes to Closed")

# Output widget to display results
output = widgets.Output()

def minutes_to_days_hours(minutes):
    days = minutes // 1440
    hours = (minutes % 1440) // 60
    return days, hours

def on_predict_clicked(b):
    # Create a data frame with all zeros for one-hot columns
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    # Handle free text categorical inputs by searching for matching encoded column
    for col in original_columns:
        input_value = inputs[col].value
        matched_columns = [c for c in columns if c.startswith(col + '_') and c.endswith(input_value)]
        if matched_columns:
            input_df.at[0, matched_columns[0]] = 1

    # Handle numerical inputs
    for col in additional_columns:
        input_df.at[0, col] = inputs[col].value

    # Ensure all missing columns are filled with zeros
    input_df = input_df.fillna(0)

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Convert prediction from minutes to days and hours
    predicted_days, predicted_hours = minutes_to_days_hours(prediction[0])

    # Display the prediction
    output.clear_output()
    with output:
        print(f"Predicted Time to Closed: {predicted_days} days, {predicted_hours} hours")

predict_btn.on_click(on_predict_clicked)

# Display all widgets
for widget in inputs.values():
    display(widget)
display(predict_btn)
display(output)


# # Best Performing Model is Random Forest! Simple yet effective

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




