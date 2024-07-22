# Sleep Data Analysis with Python

### Project:
I set out to analyze the sleep data I recorded using the Sleep Cycle app. Over time, I have gathered 100+ entries on my sleep. Sleep Cycle tracks several statistics while you sleep, including time asleep, movements per hour, regularity, snoring, and more. It also calculates a sleep quality score based on the statistics it records. My objective was to investigate the correlation between sleep quality and several variables to determine how strongly different factors influence my sleep quality score.

To achieve this, I first created data visualizations to illustrate the correlations between various factors and sleep quality. Then, I developed models to predict my sleep quality score based on the input variables. This approach allowed me to identify key factors that affect my sleep quality and provided insights into how I might improve my sleep habits.

### Getting Started
Export the data from the Sleep Cycle app: 
* Access Settings > Navigate to Export Data > Enter Email Address > Download "sleepdata.csv" attachment in same location as project files

Access Coding files through Jupyter Notebook

Import data into dataframe using Pandas library:
* data = pd.read_csv("sleepdata.csv", delimiter = ";", header = 0)

### Data Cleaning:
Clean_Sleep_Data.ipynb file:
* Cleaned dataset by removing invalid entries (naps/ not full night sleep), using unit and datatype conversions to fit analysis, reformatting column names, filling missing values (previous update didn't record location)
* Save new file: data.to_csv('clean_sleepdata.csv', index = False)

### Data Visualizations: 
Sleep_Cycle_Visualizations.ipynb file:
Used Matplotlib and Seaborn libraries to create visualizations
* Boxplot of Sleep Quality vs City (Home or College)
* Scatterplot of Sleep Quality vs Time in Bed and Time Asleep w/ linear reg line
* Scatterplot of Sleep Quality vs Time to Fall Asleep w/ linear reg line
* Scatterplot of Sleep Quality vs Time Snoring w/ linear reg line
* Barplot of Sleep Quality vs Steps (split into 5,000 step increments)
* Scatterplot of Sleep Quality vs Movements per hour w/ linear reg line
* Scatterplot of Sleep Quality vs weather temperature w/ linear reg line
* Scatterplot of Sleep Quality vs regularity w/ linear reg line
* Scatterplot of Sleep Quality vs Percent of Time in Bed actually Asleep w/ linear reg line
* Scatterplot of Snoring vs Percent of Time in Bed actually Asleep w/ linear reg line
* Scatterplot of Snoring vs Steps w/ linear reg line
* Scatterplot of Snoring vs Regularity w/ linear reg line

### Calculating Strength of Correlation:
Sleep_Cycle_Visualizations.ipynb file:
Used Scipy library: from scipy.stats import spearmanr
* Calculated Spearman Rank Correlation to measure strength and direction of association between two ranked variables without assuming linearity
* Calculated Spearman Rank Correlation for each relationship visualized previously

### Create Linear Regression Models and Plot
sleepcycle_model.ipynb file:
Used scikit-learn library: 
* from sklearn.model_selection import train_test_split
* from sklearn.linear_model import LinearRegression
* from sklearn.metrics import mean_squared_error

Linear Regression model for time in bed effect on sleep quality:
* Split 80% of data to train model and 20% to test
* Calculated Mean Squared Error to measure model's accuracy
* ScatterPlot of Linear Reg Line with test data

Linear Regression Model to predict Sleep Quality based on variables with atleast moderate correlation (Spearman rank correlation > .3)
* Filtered features with Spearman correlation above threshold
* Retrain the model using only significant features
* Scatterplot of actual Sleep Quality vs Predicted









