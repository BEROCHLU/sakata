'use strict';

const fs = require('fs');
const _ = require('lodash');
const brain = require('brain.js');
const {
    performance
} = require('perf_hooks');

const strJsonOut = fs.readFileSync('./json/seikika.json', 'utf8');
const hshOut = JSON.parse(strJsonOut);
const arrHshOut = hshOut.listdc;

const arrTrainX = _.map(arrHshOut, hsh => hsh.input);
const arrTrainT = _.map(arrHshOut, hsh => hsh.output);
const arrDate = _.map(arrHshOut, hsh => hsh.date);

const arrTrainData = _.zipWith(arrTrainX, arrTrainT, arrDate, (x, t, d) => {
    return {
        input: x,
        output: t,
        date: d
    }
});
// provide optional config object (or undefined). Defaults shown.
const config = {
    binaryThresh: 0.5,
    hiddenLayers: [4], // array of ints for the sizes of the hidden layers in the network
    activation: 'sigmoid', // supported activation types: ['sigmoid', 'relu', 'leaky-relu', 'tanh'],
    leakyReluAlpha: 0.01, // supported for activation type 'leaky-relu'
}

const trainOpt = {
    iterations: 600000,
    errorThresh: 0.00001, // the acceptable error percentage from training data --> number between 0 and 1
    log: false, // true to use console.log, when a function is supplied it is used --> Either true or a function
    logPeriod: 100000
}

const strDate = new Date();
console.log(strDate.toLocaleString());
// create a simple feed forward neural network with backpropagation
const net = new brain.brain.NeuralNetwork(config);

const timeStart = performance.now();
// start training
const netrain = net.train(arrTrainData, trainOpt);
const timeEnd = performance.now();
const timeSec = (timeEnd - timeStart) / 1000;

const arrOut = _.map(arrTrainX, arr => {
    return net.run(arr)[0];
});

let arrErate = [];
let valance = 0;
let valanceMin = Number.MAX_SAFE_INTEGER;
let valanceMax = Number.MIN_SAFE_INTEGER;

const DATA_LEN = arrTrainX.length;

for (let i = 0; i < DATA_LEN; i++) {
    arrErate[i] = (arrTrainT[i][0] - arrOut[i]) / arrTrainT[i][0] * 100;

    valance += arrErate[i];

    const undo_out = arrOut[i] * hshOut.div;
    const undo_teacher = arrTrainT[i][0] * hshOut.div;

    const pad_out = undo_out.toFixed(2).padStart(6);
    const pad_teacher = undo_teacher.toFixed(2).padStart(6);
    const pad_erate = arrErate[i].toFixed(2).padStart(5);
    const pad_valance = valance.toFixed(2).padStart(5);

    console.log(`${arrDate[i]} ${pad_out} True: ${pad_teacher} ${pad_erate} ${pad_valance}`);

    valanceMin = (valance < valanceMin) ? valance : valanceMin;
    valanceMax = (valanceMax < valance) ? valance : valanceMax;
}

const averageError = _.chain(arrErate).map(Math.abs).mean().round(2).value();
const valanceMid = (valanceMin + valanceMax) / 2;
const valanceNom = (valance - valanceMin) * 100 / (valanceMax - valanceMin);

console.log(`Average error: ${averageError}%`);
console.log(`Min: ${valanceMin.toFixed(2)} Max: ${valanceMax.toFixed(2)} Mid: ${valanceMid.toFixed(2)}`);
console.log(`epoch: ${netrain.iterations} DATA_LEN: ${DATA_LEN}`);
console.log(`Nom: ${valanceNom.toFixed(2)}`);
console.log(`Time: ${timeSec.toFixed(2)}sec.`);