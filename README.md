# Credit Scoring Prediction App

A Streamlit application designed to score credit data using a pre-trained LightGBM model built with PyCaret. This tool allows users to upload a dataset, visualize the model's performance via a confusion matrix, and download the prediction results.

## Key Features

-   **Data Upload:** Supports uploading datasets in `.csv` or `.ftr` (feather) formats.
-   **Model Prediction:** Uses a LightGBM classification model to predict credit scores.
-   **Performance Visualization:** Displays a confusion matrix to evaluate the model's predictions against actual values.
-   **Export Results:** Download the predictions as an Excel file (`.xlsx`).

## Data Requirements

To ensure the application functions correctly, the uploaded dataset must contain the following columns:

-   `renda`: Income value (used for feature engineering `log_renda`).
-   `mau`: Target variable (ground truth) indicating if the credit was bad (1) or good (0). This is required to generate the confusion matrix.

You can find a demo dataset in the `demo/` folder of this repository.

## Technologies Used

This project utilizes the following technologies and libraries:

-   **Streamlit:** For the web application interface.
-   **PyCaret:** For loading the pre-trained classification model and making predictions.
-   **Pandas & NumPy:** For data manipulation and processing.
-   **LightGBM:** The underlying machine learning model.
-   **Plotly, Matplotlib, Seaborn:** For data visualization.
-   **Scikit-learn:** For calculating metrics like the confusion matrix.
-   **XlsxWriter/OpenPyXL:** For exporting data to Excel.

## Author

[Your Name]
