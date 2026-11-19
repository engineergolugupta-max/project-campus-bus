
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------
# STEP 1: Create Sample Dataset
# --------------------------------

data = []

routes = ["Route 1", "Route 2", "Route 3"]

days = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

hours = [7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19]

passenger_counts = [5, 10, 15, 20, 25, 30,
                    35, 40, 45, 50, 55, 60]

for route in routes:
    for day in days:
        for hour in hours:
            for passengers in passenger_counts:

                # Peak hour adjustment
                if hour in [8, 9, 17, 18]:
                    crowd_value = passengers + 8
                else:
                    crowd_value = passengers

                # Crowd classification
                if crowd_value <= 20:
                    crowd = "LOW"
                elif crowd_value <= 40:
                    crowd = "MEDIUM"
                else:
                    crowd = "HIGH"

                data.append([
                    hour, day, route, passengers, crowd
                ])

df = pd.DataFrame(data, columns=[
    "hour", "day", "route",
    "passenger_count", "crowd_level"
])

# --------------------------------
# STEP 2: Separate Input and Output
# --------------------------------

X = df[["hour", "day", "route", "passenger_count"]]
y = df["crowd_level"]

# --------------------------------
# STEP 3: Prepare Machine Learning
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         ["day", "route"])
    ],
    remainder="passthrough"
)

# --------------------------------
# STEP 4: Create Decision Tree Model
# --------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

# --------------------------------
# STEP 5: Train and Test Model
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print(" CAMPUS BUS CROWD CLASSIFIER")
print("================================")

print("Model Training Completed!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --------------------------------
# STEP 6: Take User Input
# --------------------------------

print("\nEnter Bus Details:")

try:
    hour = int(input("Enter time (hour in 24-hour format): "))

    if hour < 0 or hour > 23:
        raise ValueError("Hour must be between 0 and 23.")

    day = input("Enter day (e.g. Monday): ").strip().title()

    if day not in days:
        raise ValueError("Invalid day entered.")

    route = input(
        "Enter route (Route 1, Route 2, Route 3): "
    ).strip().title()

    if route not in routes:
        raise ValueError("Invalid route entered.")

    passengers = int(input("Enter passenger count: "))

    if passengers < 0:
        raise ValueError("Passenger count cannot be negative.")

    # --------------------------------
    # STEP 7: Predict Crowd Level
    # --------------------------------

    new_bus = pd.DataFrame([{
        "hour": hour,
        "day": day,
        "route": route,
        "passenger_count": passengers
    }])

    result = model.predict(new_bus)[0]

    print("\n================================")
    print("       PREDICTION RESULT")
    print("================================")

    print("Bus Time:", hour, ":00")
    print("Day:", day)
    print("Route:", route)
    print("Passengers:", passengers)

    print("\nExpected Crowd Level:", result)

    if result == "LOW":
        print("Bus Status: Less Crowded")

    elif result == "MEDIUM":
        print("Bus Status: Moderately Crowded")

    else:
        print("Bus Status: Highly Crowded")

    print("================================")

except ValueError as error:
    print("\nInput Error:", error)
    
