# Sakata Index

The Sakata Index is a technical indicator calculated using a neural network designed for the Nikkei 225. Here is a sample of the output log file:

```text
         date  prediction  actual  diff   acc
0  2024-02-14  100.019997  101.05  1.03  1.03
1  2024-02-15  100.050003  101.54  1.50  2.53
...
42 2024-04-18  100.019997   99.70 -0.32  1.19
43 2024-04-19  100.040001   98.80 -1.23 -0.04
===
         date  prediction  actual  diff   acc
0  2024-02-15  100.080002  101.54  1.46  1.46
1  2024-02-16   99.949997   99.40 -0.55  0.91
...
42 2024-04-19  100.070000   98.80 -1.26 -1.56
43 2024-04-22  100.110001  101.57  1.46 -0.10
Mean Absolute Error: 1.01%
Epoch: 314, Final Loss: 0.000135
Norm: 17.02
===
```

- **Date**: This is the label for the date.
- **Prediction**: A value generated by a neural network model that attempts to predict the opening value of the N225 ETF.
- **Actual**: The actual opening value of the N225 ETF, which serves as the "teacher signal" for the neural network's predictions.
- **Difference**: The difference by subtracting the predicted value from the actual value.
- **Accumulator**: A running total of the daily differences.

The Sakata Index is a *Norm: value* for each batch period. The value is the normalized **Accumulator** for each batch period excluding the latest date. The normalization is the min and max values. Because the values are normalized excluding the latest date, they may fall below 0 or exceed 100. The signal becomes strong to buy as the Sakata Index approaches a negative value; conversely, the signal becomes strong to sell as it approaches over 100. Finally, this program's goal is calculating the Sakata index for each period. 

The important aspect of the Sakata index is not bringing **Prediction** close to **Actual**, but accumulating the difference by subtracting **Prediction** from **Actual**. When this accumulated error reaches a certain threshold, it triggers a strong trading signal.

The Sakata Index is not a universal indicator and is weak in identifying trends. When the Nikkei 225 continues to rise, it stays above 80, and when it continues to fall, it stays below 20. In such cases, above 80 does not necessarily mean a sell, and below 20 does not necessarily mean a buy. Additionally, it often exhibits similar characteristics to the RSI.

Why Sakata?
> Named after the Bioneural Device from Sakata Industry featured in Front Mission 2.

# Usage

To use this program, follow these steps:

1. **Prerequisites**:
   - Python 3
   - Node.js 20

2. **Install brain.js and tensorlow**:
   - [brain.js](https://github.com/BrainJS/brain.js):
      - ###### on Linux
         1. `sudo apt-get install -y build-essential libglew-dev libglu1-mesa-dev libxi-dev pkg-config` 
         2. `npm install --no-audit`
            - On a Raspberry Pi 4, it takes approximately 12 minutes.
      - ###### on Windows
         1. Install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads) or later.
         2. Install `Desktop development with C++` workload from Visual Studio Community.
         3. `npm install --no-audit`
      - ##### If you can't install brain.js on Ubuntu 22 linux-x64 due to node-gyp ERR, downgrade the Node.js version to 16.20.2.

   - [TensorFlow](https://www.tensorflow.org):
      - ###### on Linux
         1. `pip install tensorflow`
            - On Raspberry Pi OS, type the following command to install h5py before installing TensorFlow:  
            `sudo apt-get install libhdf5-dev`
      - ###### on Windows
         1. `pip install tensorflow`

3. **Install other packages**:
   - `pip install -r requirements.txt`

3. **Run scripts**:
   1. Get latest raw data.  
   `npm run getdata`
   2. Normalize the raw data and split by batch size.  
   `npm run cooking`
   3. To run scripts and append the results.  
   ```bash
   npm run output1
   npm run output2
   npm run output3
   ```
   4. To visualize the results. All results are stored in the **result** folder.  
   `npm run plot-triple`

**Validation**
   - MinGW-w64: For users who want to validate using GCC with MinGW.  
      [This link](https://code.visualstudio.com/docs/cpp/config-mingw) is provided for setting up MinGW. However, due to an error with "The file has been downloaded incorrectly" [another link](https://winlibs.com/) is recommended.  
      1. Rename `hdatexyt.csv` to `datexyt.csv`
      2. Delete header in csv
      3. `gcc -O2 ./valid/cdevice.c -lm` 
      4. `a.exe` or `./a.out`  

# Process flow

## get data

The `hdatexyt.csv` file should contain the following columns:

- `date`: This is the label for the date.
- `close_x`: Closing value of DJI ETF, input_x0
- `close_y`: Closing value of USD/JPY, input_x1
- `open_t`: Opening value of N225 ETF, teacher signal

All of the data are based on EDT. If the Japanese or U.S. market is on a holiday, the date will be skipped.

Why DJI ETF?
> Due to occasional missing data in the Yahoo Finance API's time series, DJI is represented as an ETF based on the available data. Using ^DJI is also acceptable.

Why the opening value of N225 ETF?
> To avoid training on the initial values of the Nikkei 225 Index.

## Normalization

[slicebatch.md](https://github.com/BEROCHLU/sakata/blob/main/docs/slicebatch.md)

## Training
We will train the same data using three different approaches. It would be good to compare the Norm and the opening price of the Nikkei 225 to find the output with the best performance.

- Train using an original neural network from without using deep learning libraries    
   `npm run output1`
- Train using brain.js  
   `npm run output2`
- Train using TensorFlow  
   `npm run output3`

## Result files

- `output1.log`  
  This is a log of the results obtained by using my implemented neural network to learn market data from approximately six months ago to today, segmented into 44-day intervals and learned day by day. The value of `Norm:` for each period corresponds to the Sakata Index. The training parameters are as follows:
  - Input: 3 layers (including bias)
  - Hidden: 4 layers (including bias)
  - Output: 1 layer
  - Initial weight: 0.5
  - Training iterations: 500000
  - Activation function: Sigmoid
  - Loss function: least-squares method
  - Learning rate: 0.5
  - Biases: -1
  - Training data: The same dataset is used for both training and testing the model.

   Why is there a program that builds a neural network from without using deep learning libraries?
   > I wrote the first program for the Sakata Index in C in 2008. At that time, Python was not as widespread as it is today, neural networks were not given much attention, and there were no libraries available. The output1.js file is a rewrite of the code originally written in C into Node.js.

- `output2.log`  
  This is a log of the results obtained by using brain.js to learn market data from approximately six months ago to today, segmented into 44-day intervals and learned day by day. The value of `Norm:` for each period corresponds to the Sakata Index. The training parameters are the almost same as in output1.log.

- `output3.log`  
  This is a log of the results obtained by using TensorFlow to learn market data from approximately six months ago to today, segmented into 44-day intervals and learned day by day. The value of `Norm:` for each period corresponds to the Sakata Index. The training parameters are as follows:
  - Input: 2 layers
  - Hidden: 16 layers
  - Output: 1 layer
  - Optimization: Adam
  - Initial weight: 0.5
  - Maximum training iterations: 1000
  - Activation function: Sigmoid
  - Loss function: least-squares method
  - Learning rate: 0.001
  - EarlyStopping Parameters:
    - min_delta: 0.0001
    - patience: 300
    - mode: min

   Why is the loss function sigmoid and not ReLU?
   > ReLU is a fast and effective activation function used in many deep learning models. However, for nonlinear and complex problems such as financial market prediction, sigmoid or other activation functions may be more suitable. This is because the sigmoid function constrains the output to a range between 0 and 1, allowing for probabilistic interpretation, which can be useful in financial market prediction.

- `plot-triple.png`  
  An image that combines the logs from `output1.log`, `output2.log`, and `output3.log` into a single line graph.
  When the three log files are trained over the same period, the following command will output a line graph.  
  `npm run plot-triple`

> [!NOTE]
> Despite its vulnerabilities, Node.js 16 is chosen for its high speed on GitHub Actions. In my original neural network (output1.js), I improved the speed of dot product calculations by replacing the **dot** function from mathjs with the standard **reduce** function. Consequently, the learning speed is now faster compared to brain.js.
