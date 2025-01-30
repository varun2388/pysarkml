# pysarkml

## Wine Classification Model using PySpark

This project demonstrates how to build a classification model using PySpark on the wine dataset. The model is then deployed using Streamlit for easy interaction and visualization.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/varun2388/pysarkml.git
   cd pysarkml
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the PySpark classification model script:
   ```bash
   python wine_classification.py
   ```

2. Deploy the model using Streamlit:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `wine_classification.py`: Script to build and evaluate the PySpark classification model.
- `app.py`: Streamlit deployment file for the classification model.
- `requirements.txt`: List of required libraries for the project.
- `wine.csv`: Wine dataset file used for training the model.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
