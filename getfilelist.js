'use strict';
const fs = require('fs');

const arrStrFile = fs.readdirSync('./batchjson');

console.log(arrStrFile);