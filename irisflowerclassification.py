import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = y
print("First 5 Rows of Dataset")
print(df.head())
print("\nDataset Shape:", df.shape)
plt.figure(figsize=(10, 8))
df.iloc[:, 0:4].hist(figsize=(10, 8))
plt.suptitle("Histogram of Iris Features")
plt.show()
plt.figure(figsize=(8, 6))

for i, species in enumerate(iris.target_names):
    plt.scatter(
        X[y == i, 2],  # Petal Length
        X[y == i, 3],  # Petal Width
        label=species
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Flower Scatter Plot")
plt.legend()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy:")
print(round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n===== Predict Iris Flower Species =====")

sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))
user_data = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]
user_data = scaler.transform(user_data)

prediction = model.predict(user_data)

print("\nPredicted Species:")
print(iris.target_names[prediction][0])
species = iris.target_names[prediction][0]
if species == "setosa":
    print("\nSetosa Flower Detected")
    print("Characteristics:")
    print("- Small petals")
    print("- Small petal width")
    print("- Easily separable from other species")
elif species == "versicolor":
    print("\nVersicolor Flower Detected")
    print("Characteristics:")
    print("- Medium-sized petals")
    print("- Intermediate features")
    print("- Between Setosa and Virginica")
else:
    print("\nVirginica Flower Detected")
    print("Characteristics:")
    print("- Large petals")
    print("- Large petal width")
    print("- Largest Iris species")
