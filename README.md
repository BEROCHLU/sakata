# Sakata Index
The results are as follows:  
date, predicted, actual, difference, accumulator.

* 'predicted' is a calculated value by a neural network.
* 'actual' represents the teacher signal in the input file. Here, the teacher signal is opening value of Nikkei 225 ETF.
* 'difference' is a value obtained by subtracting 'predicted' from 'actual'.
* 'accumulator' is a daily cumulative 'difference'.

The Sakata index is a value which is normalized by the most recent 'accumulator' for each period.  
The signal is strong buy as the Sakata index gets closer to 0; on the other hand, the signal is strong sell as it gets closer to 100.  
Finally, this program's goal is calculating the Sakata index.  
# Neural Network
This is very simple and out of date neural network program.  
* Loss function: `least-squares method`
* Activation function: `sigmoid`
* Learning rate: 0.5
* Weight: 0.5
* Biases: -1
* Epoch: 500,000
* [Input, Hidden, Output] layer is [3, 4, 1] and all layers, except for the Output layer, include biases.
* Training data is the same as the Test data.
# Usage
1. install
    * Python
    * Node.js
    * git
    * MinGW-w64 (optional)  
        [Using GCC with MinGW](https://code.visualstudio.com/docs/cpp/config-mingw)  
        Due to error `the file has been downloaded incorrectly` not recommend installer. Alternate source https://winlibs.com/
2. git clone this repository
3. package  
    `pip install -r requirements.txt`  
    `npm install`  
4. npm run script  
    `npm run sakata+batch`  
    `npm run matplot`  
# Note 
Despite its vulnerabilities, the matrix multiplication (dot) function of math.js@6.6.5 is used because of its high speed.