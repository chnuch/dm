import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"D:\Downloads\edadataset.csv")

# Display basic info
print("Head of the dataset:")
print(df.head(), "\n")

print("Dataset Info:")
df.info()
print("\n")

print("Descriptive Statistics:")
print(df.describe(include="all"), "\n")

# Gender distribution
print("👫 Gender Distribution:")
print(df['Gender'].value_counts(), "\n")

# Department distribution
print("Department Distribution:")
print(df['Department'].value_counts(), "\n")

# Correlation matrix
print("Correlation Matrix:")
print(df[['Age', 'Salary']].corr(), "\n")

# Age vs Salary scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Age", y="Salary", hue="Gender", palette="Set2")
plt.title("Age vs Salary by Gender")
plt.tight_layout()
plt.show()

# Salary distribution by department
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Department", y="Salary", palette="Set3")
plt.title("Salary Distribution by Department")
plt.tight_layout()
plt.show()

# Salary histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['Salary'], bins=10, kde=True, color="skyblue")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Countplot for department and gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Department', hue='Gender', palette="pastel")
plt.title("Count of Employees by Department and Gender")
plt.tight_layout()
plt.show()
