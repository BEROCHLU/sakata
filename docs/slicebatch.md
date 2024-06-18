# README

## Overview

This project reads data from a CSV file, processes it, and generates normalized JSON files. The goal is to slice the data into batches, normalize it, and save the processed data into JSON files for further analysis or machine learning tasks.

## Files

- `hdatexyt.csv`: The CSV file containing the input data.
- `slicebatch.py`: The main script that processes the data and generates JSON files.
- `setting.py`: A settings file that includes the `DESIRED_ERROR` and `PERIOD` parameters.
- `batch/`: Directory where the generated JSON files will be stored.

## CSV File Format
The `hdatexyt.csv` file should have the following columns:

- `date`: This is the label for the date.
- `close_x`: Closing value of DJI ETF, input_x0
- `close_y`: Closing value of USD/JPY, input_x1
- `open_t`: Opening value of N225 ETF, teacher

All of the data are based on EDT.
## Requirements

- Python 3.x
- pandas library

## Usage

1. Place the `hdatexyt.csv` file in the same directory as `slicebatch.py`.
2. Ensure the `setting.py` file is present in the same directory as `slicebatch.py`.
3. Run the npm script:

```bash
npm run cooking
```

## Script Description

### Global Variables

- `old_ite`: A global variable used to store the previous iteration value for percentage change calculation.

### Functions

- `f1(ite)`: Calculates the percentage change from the previous value.

### Main Script

1. Changes the current working directory to the script's directory.
2. Reads the `hdatexyt.csv` file into a pandas DataFrame.
3. Creates a new DataFrame to store the percentage change for each column (`close_x`, `close_y`, `open_t`).
4. Drops the first row as it cannot have a percentage change.
5. Resets the index of the DataFrame.
6. Creates the `batch/` directory if it does not exist, or clears it if it does.
7. Slices the data into batches based on the `PERIOD` setting and normalizes the data.
8. Generates JSON files for each batch, storing the normalized data and the divisor used for normalization.

## Normalization Process

Normalization is a crucial part of data preprocessing, especially for machine learning tasks. In this script, the normalization process is performed to scale the data within a specific range, improving the performance and stability of the models.

### Steps of Normalization

1. **Calculate Percentage Change**:
   - The script calculates the percentage change for each column (`close_x`, `close_y`, `open_t`) from the previous value using the `f1(ite)` function.
   - The first row is excluded since it cannot have a percentage change from a previous value.

2. **Slice Data into Batches**:
   - The data is sliced into batches based on the `PERIOD` setting. Each batch contains a subset of the data, facilitating manageable processing and analysis.

3. **Normalize the Data**:
   - For each batch, the script normalizes the data by dividing each value by the maximum value in the batch, adjusted by the `DESIRED_ERROR` parameter.
   - The normalization formula used is:
```math
\text{normalized\_value} = \frac{\text{original\_value}}{\text{max\_value} \times (1 + \text{DESIRED\_ERROR})}
```
   - This step ensures all values are scaled to a similar range, improving the performance of subsequent analysis or machine learning models.

4. **Store Normalized Data**:
   - The normalized data is stored in JSON format for each batch. Each JSON file contains:
     - `listdc`: A list of dictionaries, each containing the normalized `close_x`, `close_y` values as input, the normalized `open_t` value as output, and the date.
     - `div`: The divisor used to normalize the `open_t` values, allowing for denormalization later if needed.

### Example JSON Structure

```json
{
    "listdc": [
        {
            "input": [
                0.9715755607794289,
                0.9906148995323074
            ],
            "output": [
                0.9708995652086916
            ],
            "date": "2024-06-11"
        }
    ],
    "div": 102.31716586961993
}
```

### Considerations

- The normalization process helps in maintaining the relative importance of different features while ensuring that they are on a comparable scale.
- The `DESIRED_ERROR` parameter can be adjusted to control the scaling factor, providing flexibility based on specific requirements.

## Settings

The `setting.py` file should define the following parameters:

- `DESIRED_ERROR`: The desired error for normalization.
- `PERIOD`: The period for slicing the data into batches.

## Example `setting.py`

```python
DESIRED_ERROR = 0.001
PERIOD = 44
```

## Output

The script will generate JSON files in the `batch/` directory, named `seikikaXX.json`, where `XX` is a zero-padded index.

## Conclusion

This project provides a way to process and normalize time series data, making it suitable for machine learning tasks. The generated JSON files can be used as input for further analysis or modeling. The normalization process ensures that the data is scaled appropriately, facilitating better performance and stability of machine learning models.