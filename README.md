# Sakata Index
The results are as follows:  
date, predicted, actual, difference, accumulator.

* 'predicted' is a calculated value by a neural network.
* 'actual' represents the teacher signal in the input file. Here, the teacher signal is opening value of Nikkei 225 ETF.
* 'difference' is a value obtained by subtracting 'predicted' from 'actual'.
* 'accumulator' is a daily cumulative 'difference'.

The Sakata index is a value which is normalized by the most recent 'accumulator' for each period. The signal is __strong buy__ as the Sakata index approaches 0; on the other hand, the signal is __strong sell__ as it approaches 100. Finally, this program's goal is calculating the Sakata index.  
  
The important aspect of the Sakata index is not bringing 'predicted' close to 'actual', but accumulating the difference obtained by subtracting 'predicted' from 'actual'. The accumulated value flips at a certain point, and then a strong signal occurs.  
# Neural Network
The settings of main.js are as follows:  
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
    * Node.js 16
    * git
    * MinGW-w64 (optional)  
        [Using GCC with MinGW](https://code.visualstudio.com/docs/cpp/config-mingw)  
        Due to error `the file has been downloaded incorrectly` not recommend installer. Alternate source https://winlibs.com/
2. git clone this repository
3. package  
    `pip install -r requirements.txt`  
    `npm install`  
4. npm run script  
    `npm run cooking`  
    `npm run sakata+batch`  
    `npm run main+batch`  
    `npm run plot-multi`  
# Note 
Despite its vulnerabilities, the matrix multiplication (dot) function of math.js@6.6.5 is used because of its high speed.
For the same reason, Node.js 16 is recommended.  