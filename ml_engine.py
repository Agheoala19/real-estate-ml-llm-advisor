from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model():
    raw_data = fetch_california_housing(as_frame=True)
    df = raw_data.frame

    y = df["MedHouseVal"]
    X = df.drop('MedHouseVal', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    joblib.dump(model, "housing_model.pkl", compress=0, protocol=None)
    print("The model has been saved!")

def predict_price(input_features):
    model = joblib.load("housing_model.pkl")
    prediction = model.predict(input_features)
    return prediction * 100000

if __name__ == "__main__":
    train_model()