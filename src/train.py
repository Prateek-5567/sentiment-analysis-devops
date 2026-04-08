import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Tell MLflow where to save experiment data
mlflow.set_tracking_uri("mlruns")        # saves locally in an 'mlruns' folder
mlflow.set_experiment("sentiment-v1")    # groups all runs under this name

def train():
    # Load dataset
    df = pd.read_csv("../data/reviews.csv")
    X = df["text"]    # input: the review text
    y = df["label"]   # output: 0 or 1

    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start an MLflow run — everything inside gets tracked
    with mlflow.start_run():

        # Build the model pipeline
        # TfidfVectorizer: converts text → numbers (word frequency scores)
        # LogisticRegression: classifies those numbers as 0 or 1
        model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500)),
            ("clf",   LogisticRegression(C=1.0, max_iter=200))
        ])

        model.fit(X_train, y_train)

        # Measure accuracy on the test set
        accuracy = model.score(X_test, y_test)

        # Log to MLflow — visible in the UI
        mlflow.log_param("max_features", 500)
        mlflow.log_param("C", 1.0)
        mlflow.log_metric("accuracy", accuracy)

        # Save the model itself inside MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Training complete. Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train()

# bash  :  mlflow ui   => run this to open mlflow ui to check you application status and get run id.

#RUN ID : aeace55f433b4dd9b13d0940a04e3046