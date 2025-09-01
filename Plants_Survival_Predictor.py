
# Professional Practice - Indoor Plant Health

%pip install pandas
%pip install matplotlib
%pip install seaborn
%pip install scikit_learn
%pip install numpy
%pip install yellowbrick

# Imports

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
import re
import numpy as np

# Load the data

# Load your dataset
df = pd.read_csv("Indoor_Plant_Health_and_Growth_Factors.csv", keep_default_na=False)

# View the data
df.head ()

# Data quality check and cleanse

# Noticed columns that are related but had inconsistent values so amending these to none, zero. 

# If Pest_Presence is 'None' set Pest_Severity to 'None'
df.loc[df['Pest_Presence'] == 'None', 'Pest_Severity'] = 'None'

# Show a sample where Pest_Presence is 'None' and check Pest_Severity
print("Sample where Pest_Presence is 'None':")
print(df[df['Pest_Presence'] ==  'None'] [['Pest_Presence', 'Pest_Severity']].sample(5))

# If Pest_Severity is 'None' set Pest_Presence to 'None'
df.loc[df['Pest_Severity'] == 'None', 'Pest_Presence'] = 'None'

# Show a sample where Pest_Severity is 'None' and check Pest_Severity
print("Sample where Pest_Severity is 'None':")
print(df[df['Pest_Severity'] ==  'None'] [['Pest_Severity', 'Pest_Presence']].sample(5))

# If Fertilizer_Type is 'None' set Fertilizer_Amount_ml to 0
df.loc[df['Fertilizer_Type'] == 'None', 'Fertilizer_Amount_ml'] = 0

# Show a sample where Fertilizer_Type is 'None' and check Fertilizer_Amount_ml
print("Sample where Fertilizer_Type is 'None':")
print(df[df['Fertilizer_Type'] ==  'None'] [['Fertilizer_Type', 'Fertilizer_Amount_ml']].sample(5))


# If Fertilizer_Amount_ml is 0 set Fertilizer_Type to 'None'
df.loc[df['Fertilizer_Amount_ml'] == 0, 'Fertilizer_Type'] = 'None'

# Show a sample where Fertilizer_Amount_ml is '0' and check Fertilizer_Type
print("Sample where Fertilizer_Type is 'None':")
print(df[df['Fertilizer_Amount_ml'] == 0 ] [['Fertilizer_Amount_ml', 'Fertilizer_Type']].sample(5))

# Check for nulls and duplicates

# Check for nulls
df.isnull().sum()

# Check for full line duplicates
sum(df.duplicated())

# Remove any duplicate rows - if there are 2 rows the same, this deletes one
dfNo_duplicate = df.drop_duplicates()
print(df.shape)
print(dfNo_duplicate.shape) 

# Data Transformations
# A number of transformations to make the data more useful for the regression. 

# Create a new column that maps the 1-5 health score to a 0-100% survival chance using a linear scale

# Create a new column that maps the 1-5 health score to a 0-100% survival chance using a linear scale
df['Survival_Chance_Pct'] = (df['Health_Score'] - 1)/4 * 100

# Display the first few rows to confirm the transformation
df[['Health_Score', 'Survival_Chance_Pct']].head(10)

# Taking categorical and/or related columns and grouping them to make them more meaningful and reduce the number of features. 

# Group Pest columns
# Pest_Presence: convert from pest type string to binary flag just showing whether pests are present or not (1 if pests are present, 0 if not)
df['has_pests'] = (df['Pest_Presence'].str.lower() != 'none').astype(int)

# Pest_Severity: convert string rating to numeric rating
severity_map = {'none': 0, 'low': 1, 'moderate': 2, 'high': 3}
df['Pest_Severity_Score'] = df['Pest_Severity'].str.lower().map(severity_map)

# Pest impact: combine the above two into a single score
df['pest_impact'] = df['has_pests'] * df['Pest_Severity_Score']


# Group Fertilizer columns

# Fertilizer_Type: convert fertilizer type into just whether uses fertilizer or not (1 if fertilizer noted, 0 if not)
df['uses_fertililzer'] = (df['Fertilizer_Type'].str.lower() != 'none').astype(int)

# Uses Fertilizer_Amount_ml a a proxy for impact
df['fertilizer_impact'] = df['Fertilizer_Amount_ml']


# Extract features from Sunlight Exposure

# Extract hours if present otherwise estimate based on key words (e.g., '3h', '6h', all day = 8 hours) 
def extract_sun_hours(val):
    # Extract hours if present (e.g., '3h', '6h') 
    match = re.search(r'(\d+)h', val)
    if match:
        return float(match.group(1))
    # Estimate for 'all day', 'low light', etc.
    if 'all day' in val.lower():
        return 8
    if 'low light' in val.lower():
        return 8
    if 'filtered' in val.lower() or 'indirect' in val.lower():
        return 8
    if 'full sun' in val.lower():
        return 6
    return None # fallback

# Map sunlight type to new categories
def extract_sun_type(val):
    val = val.lower()
    if 'low' in val:
        return 'low'
    if 'filtered' in val or 'indirect' in val:
        return 'indirect'
    if 'direct' in val or 'full' in val:
        return 'direct'
    return 'unknown'

# Assign intensity scores based on sun_type
def sun_type_score(sun_type):
    if sun_type == 'direct':
        return 1.0
    if sun_type == 'indirect':
        return 0.5
    if sun_type == 'low':
        return 0.1
    return None

# Apply above transformations
df['sun_hours'] = df['Sunlight_Exposure'].apply(extract_sun_hours)
df['sun_type'] = df['Sunlight_Exposure'].apply(extract_sun_type)
df['sun_type_score'] = df['sun_type'].apply(sun_type_score)

# Create new column based on above transformations
df['effective_light'] = df['sun_hours'] * df['sun_type_score']

# Show a sample
print(df[['Sunlight_Exposure', 'sun_hours', 'sun_type', 'sun_type_score', 'effective_light']].sample(10))


# Group health notes into a sentiment score +1 for positive note and -1 for negative notes, 0 for neutral/unknown

# Identify 'Positive' notes
positive_notes = ['New bud', 'Strong Stem', 'Dark Green Colour']

# Identify 'Negative' notes
negative_notes = ['Wilting', 'Yellowing Leaves', 'Leaf drop', 'Brittle leaves']

# Allocate postive/negative scores Health_Notes based on above
df['Health_Notes_Sentiment'] = df['Health_Notes'].map(lambda x: 1 if x in positive_notes else (-1 if x in negative_notes else 0))

# Group soil type into broader categories, ready to one-hot encode. 
soil_group_map = {
    'Clay': 'heavy', 
    'Silty': 'heavy', 
    'Loamy': 'balanced', 
    'Peaty': 'rich', 
    'Sandy': 'light', 
    'Chalky': 'alkaline'
    }

# One-hot encoding
df['Soil_Group'] = df['Soil_Type'].map(soil_group_map)
df = pd.get_dummies(df, columns=['Soil_Group'], prefix='Soil')

# Convert to integer
soil_columns = ['Soil_alkaline', 'Soil_balanced', 'Soil_heavy', 'Soil_light', 'Soil_rich']
df[soil_columns] =df[soil_columns].astype(int)


# Drop original columns that have now been transformed
columns_to_drop = ['Fertilizer_Type', 'Fertilizer_Amount_ml', 'Pest_Presence', 'Pest_Severity', 'Sunlight_Exposure', 'Health_Notes', 'Soil_Type', 'Health_Score', 'sun_hours','sun_type', 'sun_type_score']
df.drop(columns=columns_to_drop, inplace=True)

df.head()

# Basic EDA

# Preview the table after changes
df.head()

df.columns


# Correlation

# Select numeric columns and compute correlation
numeric_df = df.select_dtypes(include='number')

# Compute correlaton with the target
correlations = numeric_df.corr() ['Survival_Chance_Pct'].drop('Survival_Chance_Pct').sort_values()

# Normalise correlation values for colour mapping
norm = plt.Normalize(correlations.min(), correlations.max())
colors = [cm.viridis(norm(value)) for value in correlations]

# Plot the correlations
plt.figure(figsize = (8,5))
plt.barh(correlations.index, correlations.values, color=colors)
plt.title("Correlation of Features with Survival_Chance_Pct")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.grid(False)
plt.tight_layout()
plt.show

# Print the most positively and negatively correlated features
most_positive = correlations.idxmax()
most_negative = correlations.idxmin()
print(f"Most positively correlated feature: {most_positive} ({correlations[most_positive]:.3f})")
print(f"Most negatively correlated features: {most_negative} ({correlations[most_negative]:.3f})")


# Set Target column as Survival_Chance_Pct
Target_Column = 'Survival_Chance_Pct'

# Correlation with Target
numeric_df = df.select_dtypes(include='number')
correlations = numeric_df.corr()[Target_Column].drop(Target_Column).sort_values(ascending=False)
print("Feature corrleations with target variable:")
print(correlations)


# Histograms for numeric columns
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols].hist(figsize=(15,10), bins =20)
plt.suptitle("Histograms of Features")
plt.tight_layout()
plt.show

# Ready the data for Machine learning - split into Train and Test sets

# look at the column names again
print(df.columns)

# Drop any remaining categorical columns as can't use in model
print(df.shape)

df_nocat = df.select_dtypes(include=[np.number])

print(df_nocat.shape)

df = df_nocat


# train test split
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.3,
                                     random_state=1234)


print(df.shape)
print(df_train.shape)
print(df_test.shape)


# Work on the train set only - keep Test for the final testing


# look at transformed data
df_train.head()

# look at correlations
df.corr(numeric_only = True) # should only do corr on numerics


# Look at the correlations with Survival_Chance_Pct to see which variables might be useful for prediction
# 
# Also look at the correlations between potential features to check for collinearity (rule of thumb - exclude features if > 0.8)


sns.heatmap(df_train.corr(numeric_only=True));

# Identify any pairs eith correlation greater thn 0.8

# compute correlation matrix
corr_matrix = df_train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

# Find features with correlation greater than 0.8
high_corr_pairs = [
    (col1, col2, upper.loc[col1, col2])
    for col1 in upper.columns
    for col2 in upper.columns
    if upper.loc[col1, col2] >0.8
]

# Print the results
for col1, col2, corr_vlue in high_corr_pairs:
    print(f"{col1} and {col2} have correlations of {corr_vlue: .2f}")


# Drop columns that risk multicollinearity

# identify columns to drop
columns_to_drop = ['has_pests', 'Pest_Severity_Score']

# drop columns
df_train = df_train.drop(columns = columns_to_drop)
df_test = df_test.drop(columns = columns_to_drop)

# Confirm dropped columns
print("Dropped columns:")
print(columns_to_drop)


# Predict Analytics

# look at the column names again
print(df_train.columns)

# Separate target (y) from the input variables X

X_train = df_train.drop('Survival_Chance_Pct', axis=1)
X_test = df_test.drop('Survival_Chance_Pct', axis=1)

y_train = df_train['Survival_Chance_Pct']
y_test = df_test['Survival_Chance_Pct']


# we can inspect the arrays
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

sns.histplot(data=df_train, x='Survival_Chance_Pct').set(title='Survival_Chance_Pct distribution')


sns.scatterplot(x ="Survival_Chance_Pct", y ="Watering_Amount_ml", data = df_train)


# Training the model


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)

print("R-squared:", model.score(X_test, y_test))   # this is R squared we will see again in metrics


# Get the intercept and coefficients

print("Intercept:", model.intercept_)
pd.DataFrame(zip(X_train.columns, model.coef_))


# Look at what the model has built
# 
# Pred Survival_Chance_Pct =
#         31.95 + (0.022225008303011797 * Height_cm) + (-0.12345057664353551 * Leaf_Count) + (0.906843496742978 * New_Growth_Count) +  (-0.008676294411616168 * Watering_Amount_ml) + (-0.5341582262862554 * Watering_Frequency_days) + (0.6820368290910226 * Room_Temperature_C) + (0.17814492619016556 * Humidity_Pct) +  (0.022146818425551305 * Soil_Moisture_Pct) + (-1.9546750638310784 * pest_impact) +  (3.053434832081531 * uses_fertililzer) + (-0.05660231236160166 * fertilizer_impact) + (-0.5370214352587541 * effective_light) + (0.884823865819685 * Health_Notes_Sentiment) + (-0.8794195230745598 * Soil_alkaline) + (-4.474657309371018 * Soil_balanced) + (4.954842222042806 * Soil_heavy) + (0.5440051904738951 * Soil_light) + (-0.14477058007112079 * Soil_rich)


# Manually predict a value


Height_cm = 150
Leaf_Count = 12
New_Growth_Count = 0
Watering_Amount_ml = 5
Watering_Frequency_days = 0
Room_Temperature_C = 30
Humidity_Pct = 5
Soil_Moisture_Pct = 3
pest_impact = 3
uses_fertililzer = 0
fertilizer_impact = 0
effective_light = 1
Health_Notes_Sentiment = 0
Soil_alkaline = 1
Soil_balanced = 0
Soil_heavy = 0
Soil_light = 0
Soil_rich =0



predicted_Value = (
        31.95 + (0.022225008303011797 * Height_cm) + (-0.12345057664353551 * Leaf_Count) + (0.906843496742978 * New_Growth_Count) +  
        (-0.008676294411616168 * Watering_Amount_ml) + (-0.5341582262862554 * Watering_Frequency_days) + (0.6820368290910226 * Room_Temperature_C) + 
        (0.17814492619016556 * Humidity_Pct) +  (0.022146818425551305 * Soil_Moisture_Pct) + (-1.9546750638310784 * pest_impact) +  
        (3.053434832081531 * uses_fertililzer) + (-0.05660231236160166 * fertilizer_impact) + (-0.5370214352587541 * effective_light) + 
        (0.884823865819685 * Health_Notes_Sentiment) + (-0.8794195230745598 * Soil_alkaline) + (-4.474657309371018 * Soil_balanced) + 
        (4.954842222042806 * Soil_heavy) + (0.5440051904738951 * Soil_light) + (-0.14477058007112079 * Soil_rich)
                   )

# print(predicted_Value)
print(f"The predicted value is:  {predicted_Value}")


# Explore Results
# 
# Get predicted ressults for the unused test set


# get predicted results
y_pred = model.predict(X_test)
y_pred[:300] # print first 10 values

# How good is the model?


# Data scatter of predicted values

sns.scatterplot(x = y_test, y =y_pred).set(title='Actual v Predicted Survival Chances')


# this is a helper function that when used will return MAPE (mean_absolute_percentage_error)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
      difference = y_true - y_pred
      actual = y_true
      absolute_pct_error = np.mean(np.abs(difference / actual)) * 100
    except Exception:
      absolute_pct_error = 0
    return np.round(absolute_pct_error, decimals=2)


# Run the Metrics


# now we need to run our models and evaluate their performance
from sklearn.linear_model import LinearRegression
import sklearn.metrics as met  # https://scikit-learn.org/stable/api/sklearn.metrics.html

# Step 1: initialise the model with its key arguments - if any
model = LinearRegression()

# Step 2: train the model on X_train
model.fit(X_train, y_train)

# Step 3: predict y_pred based on X_test
y_pred = model.predict(X_test)

# Step 4: evaluate and compare y_test vs. y_pred
r2_score = met.r2_score(y_test, y_pred)
# rmse - root mean squared error - unit level
rmse = float(format(np.sqrt(met.mean_squared_error(y_test,y_pred)),'.3f'))

# create a mask to exlude zero actuals
mask = y_test !=0
# in % +-
mape = mean_absolute_percentage_error(y_test[mask], y_pred)


print("R-squared:", r2_score)
print("RMSE:", rmse)
print("MAPE:",mape)

# In hyperparameter tuning, we run many models and collect their results into a table and then sort by our results to find the best models!


# create a results table
# useful when more than 1 model to compare
results = pd.DataFrame({
        'Model_Name': ['Linear_Regression_Basic'],
        'R2 Score': [r2_score],
        'rmse': [rmse],
        'MAPE': [mape]
})

results


