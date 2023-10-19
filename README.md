# Sakata Index
At the right end of each date in the result, you'll find a accumulator.  
The Sakata index is a value normalized by most recent accumulator of the difference between actual values and values predicted by the neural network.  
# Neural Network
This is very simple and out of date neural network program.  
* Loss function is `least-squares method`
* Activation function is `sigmoid`
* Learning rate is fixed 0.5
* Weight is fixed 0.5
* Bias is fixed -1
* [Input, Hidden, Output] layer is [3, 4, 1] and these which except Output layer include biases.
* Training data is the same Test data.
* Epoch is 500 thousand.
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
    `npm install --production`  
4. npm run script  
# Note 
Despite its vulnerabilities, the matrix multiplication (dot) function of math.js@6.6.5 is used because of its high speed.