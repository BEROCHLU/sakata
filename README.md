# Sakata Index

The Sakata Index is a technical indicator calculated using a neural network designed for the Nikkei 225. Here's an explanation of its results:

- **date**: This is the label for the date.
- **predicted**: A value generated by a neural network model that attempts to predict the opening value of the Nikkei 225 ETF.
- **actual**: The actual opening value of the Nikkei 225 ETF, which serves as the target or "teacher signal" for the neural network's predictions.
- **difference**: Calculated by subtracting the predicted value from the actual value.
- **accumulator**: A running total of the daily differences.

The Sakata Index is a value that is normalized by the most recent 'accumulator' for each period. The signal becomes strong to buy as the Sakata Index approaches 0; conversely, the signal becomes strong to sell as it approaches 100. Finally, this program's goal is calculating the Sakata index.  

The important aspect of the Sakata index is not bringing 'predicted' close to 'actual', but accumulating the difference obtained by subtracting 'predicted' from 'actual'. When this accumulated error reaches a certain threshold, it triggers a strong trading signal.  

# A Custom Implementation of a Simple Neural Network

This section outlines the configuration of a simple neural network used to calculate the Sakata Index:

- **Loss function**: Uses the least-squares method to measure the difference between the predicted and actual values.
- **Activation function**: Employs the sigmoid function, which maps any input into a value between 0 and 1.
- **Learning rate**: Set at 0.5, which determines how much the weights are updated during training.
- **Weight**: Initially set at 0.5.
- **Biases**: Starts with a bias of -1 for the nodes.
- **Epoch**: The number of iterations for which the neural network will be trained, set at 500,000.
- **Layer configuration**: The neural network has three layers with the following number of nodes: [3 (Input), 4 (Hidden), 1 (Output)]. All layers except the Output layer include biases.
- **Training data**: The same dataset is used for both training and testing the model.

# Usage

To use this program, follow these steps:

1. **Install prerequisites**:
   - Python 3
   - Node.js 16 or 20

2. **Install packages**:
   - `pip install -r requirements.txt`
   - `npm install --omit=optional`

3. **Run scripts**:
   - `npm run cooking`  
   Normalize raw data and split by batch size.
   - `npm run main`  
   To run main.js and append the results.
   - `npm run plot-single`  
   To visualize the results.

**Optional install**
   - MinGW-w64: For users who want to use GCC with MinGW.  
      [This link](https://code.visualstudio.com/docs/cpp/config-mingw) is provided for setting up MinGW. However, due to an error with "The file has been downloaded incorrectly" [another link](https://winlibs.com/) is recommended.  
      - `gcc -O2 cdevice.c -lm`  
      - `a.exe` or `./a.out`  

   - [brain.js](https://github.com/BrainJS/brain.js): A GPU accelerated library for Neural Networks written in JavaScript.
      - ###### on Linux
         1. `sudo apt-get install -y build-essential libglew-dev libglu1-mesa-dev libxi-dev pkg-config` 
         2. `npm install --include=optional --no-audit`
            - On a Raspberry Pi 4, it takes approximately 12 minutes.
         3. `npm run sakata`
      - ###### on Windows
         1. Install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads) or later.
         2. Install `Desktop development with C++` workload from Visual Studio Community.
         3. `npm install --include=optional --no-audit`
         4. `npm run sakata`
      - #### When installing brain.js on Ubuntu 22 linux-x64, Node.js 16 is required to avoid node-gyp ERR.

# Note

Despite its vulnerabilities, Node.js 16 is chosen for its high speed. In this repository, I improved the speed of dot product calculations by replacing the 'dot' function from mathjs with the standard 'reduce' function. Consequently, the learning speed is now faster compared to brain.js.
