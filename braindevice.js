'use strict';

const fs = require('fs');
const _ = require('lodash');
const brain = require('brain.js');
const {
    performance
} = require('perf_hooks');

let arrTrainX = [];
let arrTrainT = [];

const DESIRED_ERROR = 0.0002;
const PERIOD = 50;
let days; //学習データ数

const strJson = fs.readFileSync('./json/n225in.json', 'utf8');
const arrHsh = JSON.parse(strJson);

let arrDate = _.map(arrHsh, 'date');
let arrUPRO = _.map(arrHsh, 'upro');
let arrFXY = _.map(arrHsh, 'fxy');
let arrT1570 = _.map(arrHsh, 't1570');

const ARR_LEN = arrDate.length;
arrDate = _.drop(arrDate); //前日比の変化率なので初日を除外

let arrChangeUPRO = [];
let arrChangeFXY = [];
let arrChangeT1570 = [];

for (let i = 0; i < ARR_LEN; i++) {
    if (0 < i) {
        const _f0 = (arrUPRO[i] / arrUPRO[i - 1]) * 100;
        arrChangeUPRO.push(_f0);

        const _f1 = (arrFXY[i] / arrFXY[i - 1]) * 100;
        arrChangeFXY.push(_f1);

        const _f2 = (arrT1570[i] / arrT1570[i - 1]) * 100;
        arrChangeT1570.push(_f2);
    }
}

days = ARR_LEN - 1;

const SKIP = days - PERIOD; //スキップする日数
days = PERIOD;

arrDate = _.drop(arrDate, SKIP);
arrChangeUPRO = _.drop(arrChangeUPRO, SKIP);
arrChangeFXY = _.drop(arrChangeFXY, SKIP);
arrChangeT1570 = _.drop(arrChangeT1570, SKIP);

const UPRO_div = _.max(arrChangeUPRO) * (1 + DESIRED_ERROR); //スキップ後に最大値取得
const FXY_div = _.max(arrChangeFXY) * (1 + DESIRED_ERROR); //スキップ後に最大値取得
const T1570_div = _.max(arrChangeT1570) * (1 + DESIRED_ERROR); //スキップ後の最大値に期待値誤差を加えて除数とする
//学習データ正規化
for (let i = 0; i < days; i++) {
    const _x0 = arrChangeUPRO[i] / UPRO_div;
    const _x1 = arrChangeFXY[i] / FXY_div;

    arrTrainX.push([_x0, _x1]); //training data without bias
    arrTrainT.push([arrChangeT1570[i] / T1570_div]); //teacher data
}

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
    iterations: 3600000,
    errorThresh: DESIRED_ERROR, // the acceptable error percentage from training data --> number between 0 and 1
    log: true, // true to use console.log, when a function is supplied it is used --> Either true or a function
    logPeriod: 500000
}

const strDate = new Date();
console.log(strDate.toLocaleString());
// create a simple feed forward neural network with backpropagation
const net = new brain.NeuralNetwork(config);

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

for (let i = 0; i < days; i++) {
    arrErate[i] = (arrTrainT[i][0] - arrOut[i]) / arrTrainT[i][0] * 100;

    valance += arrErate[i];

    const undo_out = arrOut[i] * T1570_div;
    const undo_teacher = arrTrainT[i][0] * T1570_div;

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

console.log(`Average error: ${averageError}%`);
console.log(`Min: ${valanceMin.toFixed(2)} Max: ${valanceMax.toFixed(2)} Mid: ${valanceMid.toFixed(2)}`);
console.log(`epoch: ${netrain.iterations} days: ${days}`);
console.log(`Time: ${timeSec.toFixed(2)}sec.`);