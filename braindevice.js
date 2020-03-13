'use strict';

const brain = require('brain.js');
const _ = require('lodash');
const XLSX = require('xlsx');
const moment = require('moment');
const {
    performance
} = require('perf_hooks');
const path = require('path');
const fs = require('fs');

let arrTrainX = [];
let arrTrainT = [];

let arrDate = [];
let arrUPRO = [];
let arrFXY = [];
let arrT1570 = [];

const DESIRED_ERROR = 0.000193;
const PERIOD = 53;
let days; //学習データ数

const CSV_PATH = path.join('T:\\ProgramFilesT\\pleiades\\workspace\\node225', 'nt1570.csv');
const workbook = XLSX.readFile(CSV_PATH);
const worksheet = workbook.Sheets['Sheet1'];
const arrHashExcel = XLSX.utils.sheet_to_json(worksheet);

_.forEach(arrHashExcel, HashExcel => {
    const nExcelValue = HashExcel.date - 2; //Excelでは1900年が閏年判定されるので2月29日まである。-2
    const mDate = moment(['1900', '0', '1']).add(nExcelValue, 'days').format('YYYY-MM-DD');

    arrDate.push(mDate);
    arrUPRO.push(HashExcel.upro);
    arrFXY.push(HashExcel.fxy);
    arrT1570.push(HashExcel.t1570);
});

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
    logPeriod: 100000
}

console.log(moment().format('YYYY-MM-DD HH:mm:ss'));
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
