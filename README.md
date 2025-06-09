import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


from google.colab import files
uploaded = files.upload()


df = pd.read_csv('tested.csv')
df.head()


# Histogram for Age
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Countplot for Sex
sns.countplot(x='Sex', data=df)
plt.title('Gender Count')
plt.show()

# Countplot for Pclass
sns.countplot(x='Pclass', data=df)
plt.title('Passenger Class Count')
plt.show()


# Survival rate by gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.show()

# Boxplot of Age by Survival
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age vs Survival')
plt.show()



# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']], hue='Survived')
plt.suptitle('Pairplot of Key Features', y=1.02)
plt.show()


