# MLflow Deep Learning Project with Hyperparameter Optimization 🚀

A comprehensive deep learning project using MLflow for experiment tracking, hyperparameter optimization, and model deployment. This project demonstrates how to build, train, and deploy a neural network model for wine quality prediction while leveraging MLflow's capabilities for experiment management.

## 🌟 Project Highlights

- **Deep Learning Model**: Neural network implemented with Keras for wine quality prediction
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Hyperparameter Optimization**: Using Hyperopt for automated parameter tuning
- **Model Registry**: Streamlined model deployment workflow
- **REST API Ready**: Prepared for deployment as a REST service

## 📊 Project Structure

```
.
├── main.py              # Main training script
├── requirements.txt     # Project dependencies
├── _asserts/           # Project documentation assets
└── mlruns/             # MLflow experiment tracking data
```

## 🛠️ Model Architecture and Training

![Model Experiments](/_asserts/dl%20model%20experiment.png)

The project implements a neural network with:
- Input normalization layer
- Dense layer with 64 units and ReLU activation
- Output layer for regression

### Hyperparameter Optimization

The model uses Hyperopt to optimize:
- Learning rate (log-uniform distribution)
- Momentum (uniform distribution)

![Model Graph](/_asserts/dl%20model%20graph.png)

## 📈 Experiment Tracking

MLflow tracks all experiments with:
- Hyperparameters
- Training metrics (RMSE)
- Model artifacts
- Runtime information

![Experiments](/_asserts/dl%20experiments%202.png)

## 🔄 Model Registry and Deployment

### Registering the Best Model

After training, the best performing model is automatically registered in MLflow's Model Registry:

![Register Model](/_asserts/register%20best%20model.png)

### Model Inference

The model can be easily loaded and used for predictions:

![Inference Code](/_asserts/infrencing%20code.png)

### Loading Model from Registry

The best model is loaded from MLflow registry for inference:
> code 

![alt text](<_asserts/infrencing code photo.png>)

![Model Inference](/_asserts/dl%20infrence%20save.png)

## 🚀 Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training script:
```bash
python main.py
```

3. View experiments in MLflow UI:
```bash
mlflow ui
```

## 📦 Dependencies

- MLflow
- Keras
- NumPy
- Pandas
- Hyperopt
- Scikit-learn

## 🔍 Model Performance

The model is evaluated using Root Mean Squared Error (RMSE) on the validation set. The hyperparameter optimization process automatically selects the best performing model based on this metric.

## 🌐 Deployment

The trained model can be deployed as a REST API using MLflow's built-in serving capabilities:

```bash
mlflow models serve -m "models:/wine-quality/1" -p 5000
```

This exposes the model as a REST endpoint for real-time predictions.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
