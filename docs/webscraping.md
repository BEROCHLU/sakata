### Functions

#### `getDataFrame1()`

This function fetches and processes data from a specific source URL, which is base64 encoded in `str_k`.

1. **Decoding and Fetching Data:**
   - Decodes the base64 encoded URL.
   - Iterates through three pages of the URL, fetching HTML tables and extracting the relevant data.

2. **Processing Data:**
   - Concatenates data from multiple pages.
   - Sorts the data by date and resets the index.
   - Formats the date to `yyyy-mm-dd`.
   - Adjusts dates: moves the date to the previous Friday if it is a Monday, otherwise moves it to the previous day.

3. **Returns:**
   - A concatenated and processed DataFrame containing the data from the specified URL.

#### `getDataFrame2()`

This function retrieves DIA (Dow Jones Industrial Average) stock data and converts it into a DataFrame.

1. **Processing Data:**
   - Decode the URL encoded in str_us to get the page URL.
   - Retrieve data from 3 pages and append each DataFrame to a list.
   - Concatenate the data, sort by date, and reset the index.
   - Convert the date format and convert the date to a string format to create a new column.

#### `getDataFrame3()`

This function fetches currency exchange rate data from a specific source URL, which is base64 encoded in `str_m`.

1. **Decoding and Fetching Data:**
   - Decodes the base64 encoded URL.
   - Sends an HTTP GET request to fetch the data with specified parameters.

2. **Processing Data:**
   - Converts the data into a DataFrame.
   - Converts UNIX timestamps to JST date strings.
   - Removes unnecessary columns (open, high, low).
   - Converts the close column to numerical format and rounds to four decimal places.

3. **Returns:**
   - A processed DataFrame containing currency exchange rate data.

### Lambda Functions

The script defines three lambda functions to perform specific date-related operations:

#### `f1`

```python
f1 = lambda dt: dt + datetime.timedelta(days=-3) if dt.weekday() == 0 else dt + datetime.timedelta(days=-1)
```

- **Purpose:** Adjusts dates for data processing.
- **Functionality:** If the date is a Monday, it subtracts three days to move to the previous Friday. For other days, it subtracts one day to get the previous day.

#### `f2`

```python
f2 = lambda ms: datetime.datetime.fromtimestamp(ms, tz=edt).strftime("%Y-%m-%d")
```

- **Purpose:** Converts UNIX timestamps to EDT date strings.
- **Functionality:** Converts a timestamp in milliseconds to a formatted date string (`yyyy-mm-dd`) in the EDT timezone.

#### `f3`

```python
f3 = lambda ns: datetime.datetime.fromtimestamp(ns / 1000).strftime("%Y-%m-%d")
```

- **Purpose:** Converts UNIX timestamps to JST date strings.
- **Functionality:** Converts a timestamp in nanoseconds to a formatted date string (`yyyy-mm-dd`).

### Main Execution

The main block of the script performs the following actions:

1. **Data Fetching:**
   - Calls `getDataFrame1()`, `getDataFrame2()`, and `getDataFrame3()` to fetch and process data from different sources.

2. **Merging Data:**
   - Merges the DataFrames on the `date` column.
   - Selects specific columns for the final output and renames columns for clarity.

3. **Saving Data:**
   - Ensures the output directory exists and creates it if necessary.
   - Saves the final merged DataFrame to a CSV file with appropriate headers.

### Note
If even one data item is missing, for example a public holiday, the date will be skipped.