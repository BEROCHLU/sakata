'use strict';

const fs = require('fs');
const _ = require('lodash');
const _brain = require('brain.js');
const { performance } = require('perf_hooks');

// provide optional CONFIG object (or undefined). Defaults shown.
const CONFIG = {
    binaryThresh: 0.5,
    hiddenLayers: [4], // array of ints for the sizes of the hidden layers in the network
    activation: 'sigmoid' // supported activation types: ['sigmoid', 'relu', 'leaky-relu', 'tanh']
}
const TRAIN_OPT = {
    iterations: 400000,
    errorThresh: 0.00001, // the acceptable error percentage from training data --> number between 0 and 1
    log: false, // true to use console.log, when a function is supplied it is used --> Either true or a function
    logPeriod: 100000,
    learningRate: 0.5 // scales with delta to effect training rate --> number between 0 and 1
}
const BATCH_PATH = './batch';

// Main function to encapsulate the logic
(async () => {
    //計測開始
    const timeStart = performance.now();
    console.log(new Date().toLocaleString());

    const arrStrFile = await fs.promises.readdir(BATCH_PATH);

    for (const strFile of arrStrFile) {
        // バッチファイルの読み込みとJSONへの変換
        const strJson = await fs.promises.readFile(`${BATCH_PATH}/${strFile}`, 'utf8');
        const objJson = JSON.parse(strJson);
        const collection = objJson.listdc;

        const arrTrainX = _.map(collection, obj => obj.input);
        const arrTrainT = _.map(collection, obj => obj.output);
        const arrDate = _.map(collection, obj => obj.date);

        const arrTrainData = _.zipWith(arrTrainX, arrTrainT, arrDate, (x, t, d) => {
            return {
                input: x,
                output: t,
                date: d
            }
        });

        // create a simple feed forward neural network with backpropagation
        const BNN = new _brain.NeuralNetwork(CONFIG);
        // start training
        const netrain = BNN.train(arrTrainData, TRAIN_OPT);

        const arrOut = _.map(arrTrainX, arr => {
            return BNN.run(arr)[0];
        });

        let arrErate = [];
        let acc = 0;
        let acc_min = Number.MAX_SAFE_INTEGER;
        let acc_max = Number.MIN_SAFE_INTEGER;

        const DATA_LEN = arrTrainX.length;

        console.log('      date predic actual  diff   acc'); //header

        for (let i = 0; i < DATA_LEN; i++) {
            arrErate[i] = (arrTrainT[i][0] - arrOut[i]) / arrTrainT[i][0] * 100;

            acc = _.reduce(arrErate, (presum, current) => {
                acc_min = Math.min(acc_min, presum); //前回の蓄積結果で最小値を更新
                acc_max = Math.max(acc_max, presum); //前回の蓄積結果で最大値を更新
                return presum + current; // 配列最後のreturnは最大最小の更新対象にならない
            });

            const undo_out = arrOut[i] * objJson.div;
            const undo_teacher = arrTrainT[i][0] * objJson.div;

            const pad_out = undo_out.toFixed(2).padStart(6);
            const pad_teacher = undo_teacher.toFixed(2).padStart(6);
            const pad_erate = arrErate[i].toFixed(2).padStart(5);
            const pad_acc = acc.toFixed(2).padStart(5);

            console.log(`${arrDate[i]} ${pad_out} ${pad_teacher} ${pad_erate} ${pad_acc}`);
        }

        const averageError = _.chain(arrErate).map(Math.abs).mean().round(2).value();
        //const acc_mid = (acc_min + acc_max) / 2;
        const acc_norm = (acc - acc_min) * 100 / (acc_max - acc_min);

        console.log(`Mean Absolute Error: ${averageError}%`);
        //console.log(`Min: ${acc_min.toFixed(2)} Max: ${acc_max.toFixed(2)} Mid: ${acc_mid.toFixed(2)}`);
        console.log(`Epoch: ${netrain.iterations} BatchSize: ${DATA_LEN}`);
        console.log(`Norm: ${acc_norm.toFixed(2)}`);
        console.log(`===`);
    };// for of
    //計測終了
    const timeEnd = performance.now();
    const timeSec = (timeEnd - timeStart) / 1000;

    console.log(`Time: ${Math.floor(timeSec / 60)} min ${Math.floor(timeSec % 60)} sec.\n`);
})();