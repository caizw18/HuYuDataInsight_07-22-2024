# Example of training a Random Forest Classifier using Scikit-Learn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Example of web scraping using BeautifulSoup

import requests
from bs4 import BeautifulSoup

# URL to scrape
url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Find all the links on the page
links = soup.find_all('a')

# Print the first 10 links
for index, link in enumerate(links[:10]):
    print(f"Link {index+1}: {link.get('href')}")

# Example of asynchronous programming using asyncio

import asyncio

async def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
        await asyncio.sleep(1)  # Simulate I/O bound operation
    return result

async def main():
    tasks = [factorial(i) for i in range(1, 6)]
    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        print(f"Factorial of {i+1} is {result}")

if __name__ == "__main__":
    asyncio.run(main())

# Example of data analysis using Pandas

import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
    'Age': [25, 30, 35, 40, 45],
    'Salary': [50000, 60000, 75000, 90000, 100000]
}
df = pd.DataFrame(data)

# Calculate statistics
mean_salary = df['Salary'].mean()
max_age = df['Age'].max()

# Plotting
plt.bar(df['Name'], df['Salary'])
plt.xlabel('Name')
plt.ylabel('Salary')
plt.title('Salary Distribution')
plt.show()

print(f"Mean Salary: {mean_salary}")
print(f"Max Age: {max_age}")