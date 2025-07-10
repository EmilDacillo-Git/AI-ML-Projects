<div align="center">
  

  # 📈 IBM Stock Price Prediction using AI/ML 🧠

  **A time-series forecasting project that leverages Long Short-Term Memory (LSTM) networks to predict future IBM stock prices.**

  [![Python][Python-badge]][Python-url]
  [![TensorFlow][TensorFlow-badge]][TensorFlow-url]
  [![Scikit-learn][Scikit-learn-badge]][Scikit-learn-url]
  [![Pandas][Pandas-badge]][Pandas-url]
  [![NumPy][NumPy-badge]][NumPy-url]
  [![Jupyter Notebook][Jupyter-badge]][Jupyter-url]

</div>

---

### Welcome to the IBM Stock Prediction Project! 👋

This repository contains the code and resources for a machine learning project focused on predicting the stock price of IBM. By analyzing historical stock data, we train an LSTM model—a special kind of Recurrent Neural Network (RNN)—to forecast future price movements. This project serves as a practical example of applying deep learning techniques to financial time-series data.

---

### 🚀 Key Features

* **📊 Data Acquisition:** Fetches real-time, historical stock data directly from Yahoo Finance.
* **🧼 Data Preprocessing:** Cleans and scales the data to prepare it for the neural network.
* **🧠 LSTM Model:** Implements a powerful LSTM model using TensorFlow and Keras for robust time-series forecasting.
* **📈 Visualization:** Plots the actual vs. predicted stock prices for a clear visual comparison of the model's performance.
* **📓 Interactive Notebook:** The entire workflow is documented in a single, easy-to-follow Jupyter Notebook.

---

### 🛠️ Tech Stack & Tools

This project is built with a suite of powerful open-source libraries:

| Icon | Tool | Description |
| :--- | :--- | :--- |
| <img src="https://img.icons8.com/color/48/000000/python--v1.png" alt="Python"/> | **Python** | The core programming language for the project. |
| <img src="https://muzny.github.io/csci1200-notes/images/logo/logo.png" alt="Jupyter" width="48"/> | **Jupyter Notebook** | For interactive development, visualization, and documentation. |
| <img src="https://img.icons8.com/color/48/000000/tensorflow.png" alt="TensorFlow"/> | **TensorFlow & Keras** | Used for building and training the LSTM neural network. |
| <img src="https://img.icons8.com/color/48/000000/line-chart.png" alt="Scikit-learn"/> | **Scikit-learn** | Essential for data preprocessing, specifically for scaling features. |
| <img src="https://cdn.worldvectorlogo.com/logos/pandas.svg" alt="Pandas" width="48"/> | **Pandas** | The go-to library for data manipulation and analysis in Python. |
| <img src="https://icon.icepanel.io/Technology/svg/NumPy.svg" alt="NumPy" width="48"/> | **NumPy** | Fundamental for numerical operations and handling multi-dimensional arrays. |
| <img src="https://img.icons8.com/color/48/000000/combo-chart.png" alt="Matplotlib"/> | **Matplotlib** | For creating static, animated, and interactive visualizations. |
| <img src="https://s.yimg.com/cv/apiv2/myc/finance/Finance_icon_0919_250x252.png" alt="Yahoo Finance" width="48"/> | **yfinance** | A reliable and popular library to fetch historical market data from Yahoo Finance. |

---

### 📥 Getting the Data: `yfinance`

To train our model, we need historical stock data. The `yfinance` library offers a simple and robust way to download this data directly into a Pandas DataFrame. It's the modern, community-maintained replacement for the older `pandas-datareader` Yahoo Finance API.

Here’s how we use it in the project:

```python
import yfinance as yf
import pandas as pd

# Set the ticker for IBM
ticker = 'IBM'

# Download historical data from 2010 to the end of 2022
df = yf.download(ticker, start='2010-01-01', end='2022-12-31')

# Display the first few rows of the dataset
print(df.head())
```
This snippet fetches over a decade of daily stock data, including Open, High, Low, Close, Adjusted Close prices, and Volume, providing a rich dataset for our model.

---

### ⚙️ How to Set Up and Run the Project

Want to run this project on your own machine? Follow these simple steps.

#### 1. Clone the Repository

First, clone this repository to your local machine using Git, copy the SSH:

```bash
git clone git@github.com:EmilDacillo-Git/AI-ML-Projects.git
```

#### 2. Create a Virtual Environment (Recommended)

It's best practice to create a virtual environment to keep project dependencies isolated.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies

Create a `requirements.txt` file in the project directory with the following content:

**`requirements.txt`**
```
numpy
pandas
yfinance
matplotlib
scikit-learn
tensorflow
jupyterlab
```

Now, install all the required packages using pip:

```bash
pip install -r requirements.txt
```

#### 4. Launch Jupyter Notebook

You're all set! Launch Jupyter Notebook or JupyterLab to run the project.

```bash
jupyter lab
```

This will open a new tab in your web browser. From there, navigate to and open the `IBM Stock Price Prediction.ipynb` notebook and run the cells sequentially.

---

### 📊 Example Result

The model's predictions are plotted against the actual stock prices to visually assess its accuracy. The graph below shows an example of the kind of output you can expect.



<div align="center">
  <img src="https://github.com/Chando0185/stock_price_prediction/blob/main/static/ema_100_200.png?raw=true" alt="Model Prediction Results" width="700"/>
  <br>
  <i>A comparison of actual vs. predicted IBM closing prices.</i>
</div>

---

### 🤝 Contributing

Contributions are welcome! If you have ideas for improvements, feel free to fork the repository and submit a pull request. You can also open an issue with the "enhancement" tag.

---

<!-- Badges -->
[Python-badge]: https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python
[Python-url]: https://www.python.org/
[TensorFlow-badge]: https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow
[TensorFlow-url]: https://www.tensorflow.org/
[Scikit-learn-badge]: https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn
[Scikit-learn-url]: https://scikit-learn.org/
[Pandas-badge]: https://img.shields.io/badge/pandas-2.x-blue?style=for-the-badge&logo=pandas
[Pandas-url]: https://pandas.pydata.org/
[NumPy-badge]: https://img.shields.io/badge/numpy-1.2x-blue?style=for-the-badge&logo=numpy
[NumPy-url]: https://numpy.org/
[Jupyter-badge]: https://img.shields.io/badge/Jupyter-Lab-orange?style=for-the-badge&logo=jupyter
[Jupyter-url]: https://jupyter.org/
