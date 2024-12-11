import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def evaluate_model_performance(history, model, X_test, y_test):
    """
    Evaluate and visualize the performance of a trained model.
    
    Parameters:
        history: History object from model training (model.fit()).
        model: Trained Keras model.
        X_test: Test data.
        y_test: True labels for the test data.
        threshold: Decision threshold for binary classification (default is 0.5).
        
    Returns:
        None. Prints classification report and plots accuracy/loss curves.
    """
    # Plot accuracy and loss
    plt.figure(figsize=(12, 4))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()
    
    # Predict on test data
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))