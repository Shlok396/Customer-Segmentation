Project Title: Customer Segmentation Analysis Using K-Means Clustering

OBJECTIVE
The objective of this project is to perform customer segmentation analysis for an e-commerce company using customer demographic and purchasing behavior data. By grouping customers into distinct segments based on income, recency, and spending patterns, the analysis aims to help the business understand different customer types and support targeted marketing, customer retention, and improved business decision-making.

STEPS PERFORMED
1.	Dataset Collection
o	Used the iFood customer dataset (ifood_df.csv) containing customer demographics and purchase behavior.
2.	Data Loading
o	Loaded the dataset into a Jupyter Notebook using Python and the Pandas library.
3.	Data Exploration
o	Examined the structure of the dataset using functions like head() and info().
o	Reviewed statistical summaries to understand customer income, spending, and recency patterns.
4.	Data Cleaning
o	Checked for missing values in the dataset.
o	Removed rows with missing data to ensure accurate analysis.
5.	Feature Selection
o	Selected relevant behavioral features such as Income, Recency, Wine Spending, and Meat Spending for customer segmentation.
6.	Data Scaling
o	Standardized the selected features using StandardScaler to prepare the data for clustering.
7.	Customer Segmentation
o	Applied the K-Means clustering algorithm to group customers into three distinct segments based on their behavior.
8.	Visualization
o	Visualized the customer segments using scatter plots to clearly show differences between customer groups.
9.	Analysis of Segments
o	Calculated average spending and income for each segment to understand their characteristics.
10.	Insights and Recommendations
o	Interpreted each customer segment and provided business recommendations accordingly.

TOOLS USED
•	Google Colab / Jupyter Notebook – For writing and executing Python code
•	Python – Programming language used for analysis
•	Pandas – Data loading, cleaning, and manipulation
•	NumPy – Numerical operations
•	Matplotlib & Seaborn – Data visualization
•	Scikit-learn – K-Means clustering and data scaling

CODE

1)	Importing required libraries:  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
2)	Loading csv File: 
df = pd.read_csv("ifood_df.csv")
df.head()

3)	Understanding the data: 
df.info()

4)	Basic Statistics:
df.describe()

5)	Checking Values:
df.isnull().sum()

6)	Cleaning Data:
df = df.dropna()

7)	Selecting the features:
features = df[['Income', 'Recency', 'MntWines', 'MntMeatProducts']]
8)	Scaling Data:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

9)	Applying K-Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
df['Customer_Segment'] = kmeans.fit_predict(scaled_features)

10)	Visualizing:
plt.figure(figsize=(8,5))
sns.scatterplot(
    x=df['Income'],
    y=df['MntWines'],
    hue=df['Customer_Segment'],
    palette='Set2'
)
plt.title("Customer Segmentation based on Income & Wine Spending")
plt.show()

11)	Understanding Segments:
df.groupby('Customer_Segment')[['Income','MntWines','MntMeatProducts']].mean()

THE OUTCOME IN BRIEF
•	Customers were successfully divided into three distinct segments based on income and purchasing behavior.
•	One segment represented high-income, high-spending customers, another represented low-value customers, and the third showed moderate spending with growth potential.
•	Visualizations clearly demonstrated differences between customer segments.
•	The analysis provided actionable insights to support targeted marketing strategies, personalized offers, and customer retention initiatives.
•	The project demonstrated practical application of data cleaning, exploratory data analysis, clustering techniques, and data visualization.






Output:
 

Cell Output:
	Income	MntWines	MntMeatProducts
Customer_Segment			
0	78171.692090	729.050847	580.757062
1	35268.744996	58.075718	30.729330
2	65000.216524	498.974359	176.092593

