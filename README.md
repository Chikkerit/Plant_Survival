# Plant Survival Prediction

For lovers of houseplants. Wouldn't it be great if when you got your new plant home you could work out it's chances of survive?

Perhaps you you could see whether that perfect spot you had picked out is actually the best option, or whether you should fork out for that fertilser or not.  

Well, this project aimed to help you out investigating whether it is possible to predict the chances of a houseplant surviving based on care information provided.


## The Data

**Data source:** 

Kaggle: https://www.kaggle.com/datasets/souvikrana17/indoor-plant-health-and-growth-dataset

**Data structure:** 

1000 rows, each row representing a single plant, 17 columns (features) containing plant care factors

**Features:** 

| Feature Name  | Description   | 
| ------------- | ------------- | 
| Plant_ID | Scientific plant name (e.g., Ficus lyrata, Aloe vera)
| Height_cm | Height of the plant in centimeters
| Leaf_Count | Total number of leaves
| New_Growth_Count | Number of new buds or leaves observed
| Health_Notes | Qualitative notes on plant appearance (e.g., “yellowing leaves”)
| Watering_Amount_ml | Amount of water given in milliliters
| Watering_Frequency_days | Days between watering sessions
| Sunlight_Exposure | Descriptive light exposure (e.g., “3 hrs morning sun”)
| Room_Temperature_C | Room temperature in Celsius
| Humidity_% | Room humidity percentage
| Fertilizer_Type | Type of fertilizer used
| Fertilizer_Amount_ml | Amount of fertilizer applied
| Pest_Presence | Detected pest type (if any)
| Pest_Severity | Level of pest infestation
| Soil_Moisture_% | Soil moisture percentage
| Soil_Type | Soil classification (e.g., loamy, clay, sandy)
| Health_Score | Rating from 1 (dying) to 5 (thriving)

## Project Steps

### Executive Summary

The purpose of this project was to explore whether a dataset containing information on plant care factors could be used to predict the survival chances of indoor houseplants. 
Transformations were performed to make the data suitable for a Linear Regression model which would then be used to estimate the percentage chance of a plants survival given care information. 
Exploratory Data Analysis suggested weak correlations between the input features and survival chances, however the data was still deemed to be a solid foundation for setting up the model.
The model performs as expected and successfully generates survival predictions, but the current data limits its accuracy. With improved data or refined transformations the model has strong potential to deliver accurate and meaningful results.

### Observation/Question

It has long been known that houseplants are good for health, as stated by Oboni et al. (2025) “houseplants have become indispensable for creating environments that provide peace, coziness, and mental comfort”. So it follows that the ability to predict whether a plant will survive in your care would be beneficial. 
This project aims to test the hypothesis that care and environmental factors can be used to predict indoor plant survival chances. The Scientific Method was used to guide the process.
Attempts to create a predictor from this dataset have previously been made, both of which struggled to create a reliable model. With those previous efforts in mind this project attempted feature transformations and shifted the focus from predicting a health score of 1 to 5, to a percentage chance of survival.
  
### Data Collection

The dataset was obtained from Kaggle. It contains 1000 rows, each row representing a single plant. The features hold information on the plants environment and the care it has received, such as volumes of light and water, along with a rating of the plants health. Each of the features are known to be factors that have impact on plant health. The dataset is publicly available and does not contain any personal or sensitive information. 

### Data Preparation

Python was chosen as the best tool for the project for ‘Its ease of use and versatility” (Linear Regression in Python: A Guide to Predictive Modeling | DataCamp, no date). The programming language R should also be considered for regression but was not used for this project due to lack of knowledge. 
Once the data was loaded into Python, checks were made to ensure there were no nulls or duplicates. Inconsistent values between related columns, for instance Pest_Presence and Pest_Severity were fixed by ensuring values in related features made sense. E.g. if Pest_Presence was ‘None’, then Pest_Severity was amended to ‘None’, and vice versa.  

The Health_Score column, originally rated from 1 to 5, was transformed into Survival_Chance_Pct ranging from 0 - 100% using a linear scale. This changed the target variable, from categorical to continuous therefore providing a more precise and easier to understand prediction of survival.

Many of the original columns are categorical so feature engineering was applied to prepare the dataset for regression analysis. Variables were grouped and simplified. Pest_Presence and Pest_Severity were combined into a numeric Pest_Impact score. 

The uses_fertilizer feature was created, where Fertilizer_Type, states a value uses_fertilizer shows 1 and 0 if no type present. Fertilizer_Amount_ml was used as the base for new feature fertilizer_impact, representing how much effect fertilizer will have on the plants survival.

Sunlight exposure was split into estimated hours and intensity scores, which were then combined into an effective_light feature. 

Health notes were mapped to sentiment scores based on whether the notes were positive or negative. 

Soil types were grouped into broader categories and one-hot encoded. 

These transformations enable the original categorical features to be dropped, as well as improve feature relevance for modelling.

### Exploratory Data Analysis (EDA)

Exploratory data analysis was then performed, producing visualisations and correlation analysis to better understand the data and feature relationships.
Histograms of numeric columns were created to show the distribution of values for each feature.

A correlation bar was created to show the relationship of each feature to the Survival_Chance_Pct. It shows some positive and some negative as we would expect, but there don’t appear to be any strong correlations.  

Although the correlations are weak, all features are retained due to their known impact on plant health. 

### Analysis/Modelling

The type of regression analysis used depends on the nature of the outcome variable, linear regression is appropriate for continuous outcomes, while logistic regression is used for categorical ones as stated by Castro and Ferreira, 2022. As the desired output was a continuous 1-100% scale, linear regression would be required. 
The model used in this project was a basic linear regression model, using the ordinary least squares (OLS) method. For this project, scikit-learn’s version of linear regression was chosen as it’s considered a practical option for quickly building and testing a model in an exploratory setting and  “makes linear regression easy to implement” (Sklearn Linear Regression: A Complete Guide with Examples, no date). 
Prior to running the model the data is split into test and train (excluding any remaining categorical features) ready to prepare for the model. A 70/30 split was chosen.

A correlation heatmap was created to view relationships between features. 

The heatmap highlights strong relationships that could signify multicollinearity, so a step was added to identify any feature pairs with a correlation greater than 0.8.

The high correlations between pest_impact, Pest_Severity and has_pests proves multicollinearity prompting the removal of Pest_Severity and has_pests. 
Pest_Impact was retained as this column is a combination of the other two and provides the most value to the model. Removing these fields reduces the risk of bias or instability in the model. 
The target variable Survival_Chance_Pct is then separated from the input variables in both the test and train data frames.

A bar chart is created to show the distribution of  Survival_Chance_Pct to ensure a reasonable distribution. 

The model is then trained on df_train and an R-squared metric provided.

The negative R-squared of -0.05 shows that the model is not effectively capturing the relationship between the variables and the survival chance. 

However, inspecting the model’s build by entering what would be considered good care inputs into a manual prediction, provides a prediction more on the positive side of survival, suggesting that the model does have potential. Testing this further using negative care inputs confirms this thinking.  

Further analysis of metrics confirms the model is not performing well overall. 
The negative R-squared shows it’s not providing any explanatory power. Root Mean Squared Error (RMSE) suggest the predictions are off by about 36%, which, although not the worst outcome, is still underwhelming. And the MAPE (Mean Average Percentage Error) of 0 suggests that either the model is perfect, which is unlikely given the other metrics, or, more likely there are zero’s in the data causing an issue. 

### Results

The hypothesis tested in this project was that care and environmental factors can be used to predict indoor plant survival chances. The corresponding null hypothesis would be that these factors have no predictive relationship to survival. While it is known that the features in the data are either essential or relevant to a plant’s survival, the project does not provide sufficient evidence to reject the null hypothesis. This is likely due to the data itself, or issues with feature transformations, rather than a flaw in the hypothesis. 
It is important to be clear that the model is only as good as the data it is trained on. The Kaggle site the dataset was obtained from does not state how it was collected, so it is unknown whether conditions were controlled or standardized, which affects its reliability. 

Considerations should also be made in regard to bias. The dataset includes a wide range of plant species, but doesn’t account for each species specific needs, this could introduce bias as some plants may survive better under certain conditions than others, skewing the models understanding of what good care looks like. 
An improved dataset, perhaps with more thought about the values used to represent each care factor and a more in depth investigation into the importance of features used, the value they provide and the transformations made to them, may still support the hypothesis. 

### Conclusion

This project set out to explore whether care and environmental factors could be used to predict the survival chances of indoor houseplants. While the model was successfully built and functioned as intended, the results did not provide sufficient evidence to support the hypothesis. The dataset used in this project likely lacks the accuracy and representativeness needed to support the hypothesis and is probably the cause of the model’s poor performance. 
In addition to new data, considerations for future work could include techniques such as grouping plants with  similar care requirements. Alternative regression models, such as random forests, may better capture non-linear relationships. Additionally, incorporating model explainability tools like SHAP could be used to ‘provide a fair and unbiased measure of each feature's contribution to the predicted value of each sample’ (Ponce-Bobadilla et al., 2024) allowing deeper insights into each features influence and its value to the model. 
Overall, while the current data did not support the hypothesis, the project lays a strong foundation for future exploration with improved data and more advanced techniques.


