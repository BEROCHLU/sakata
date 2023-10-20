'use strict';

const fs = require('fs');
const _ = require('lodash');
const math = require('mathjs');
const {
    performance
} = require('perf_hooks');

let IN_NODE; //入力ノード数（バイアス含む）
let HID_NODE; //隠れノード数
let OUT_NODE = 1; //出力ノード数

let DATA_LEN; //学習データ数
const ETA = 0.5; //学習係数
const THRESH = 500000;
const BATCH_PATH = './batch';

const sigmoid = x => 1 / (1 + Math.exp(-x)); //シグモイド関数
const dsigmoid = x => x * (1 - x); //シグモイド関数微分

//乱数生成
const frandWeight = () => 0.5; //  0 <= x < 1.0, Math.random()
const frandBias = () => -1;

function calculateNode(n, hid, out, x, v, w) {
    for (let i = 0; i < HID_NODE; i++) {
        const fDot = math.dot(x[n], v[i]);
        hid[i] = sigmoid(fDot);
    }

    hid[HID_NODE - 1] = frandBias(); //配列最後にバイアス

    for (let i = 0; i < OUT_NODE; i++) {
        const fDot = math.dot(w[i], hid);
        out[i] = sigmoid(fDot);
    }

    return [hid, out];
}

function mseArray(arrArr) {
    const MSE_AVE = _.meanBy(arrArr, arr => {
        return _.mean(arr);
    });
    return MSE_AVE;
}

function printResult(arrHsh, DIV_T, arrMSE, epoch, t, hid, out, x, v, w) {

    let arrErate = [];
    let accumulator;
    let accumulatorMin = Number.MAX_SAFE_INTEGER;
    let accumulatorMax = Number.MIN_SAFE_INTEGER;

    for (let i = 0; i < DATA_LEN; i++) {

        const ret = calculateNode(i, hid, out, x, v, w);
        [hid, out] = [ret[0], ret[1]];

        arrErate[i] = (t[i][0] - out[0]) / out[0] * 100; //t[i][0] | out[0]

        accumulator = _.reduce(arrErate, (presum, current) => {
            accumulatorMin = Math.min(accumulatorMin, presum); //前回の蓄積結果で最小値を更新
            accumulatorMax = Math.max(accumulatorMax, presum); //前回の蓄積結果で最大値を更新
            return presum + current; // 配列最後のreturnは最大最小の更新対象にならない
        });

        const undo_out = out[0] * DIV_T;
        const undo_teacher = t[i][0] * DIV_T;

        const pad_out = undo_out.toFixed(2).padStart(6);
        const pad_teacher = undo_teacher.toFixed(2).padStart(6);
        const pad_erate = arrErate[i].toFixed(2).padStart(5);
        const pad_accumulator = accumulator.toFixed(2).padStart(5);

        console.log(`${arrHsh[i].date} ${pad_out} True: ${pad_teacher} ${pad_erate}% ${pad_accumulator}`);
    }

    const averageError = _.chain(arrErate).map(Math.abs).mean().round(2).value();

    const accumulatorMid = (accumulatorMin + accumulatorMax) / 2;
    const accumulatorNom = (accumulator - accumulatorMin) * 100 / (accumulatorMax - accumulatorMin);
    const MSE_AVE = mseArray(arrMSE);

    console.log(`Average error: ${averageError}%`);
    console.log(`Min: ${accumulatorMin.toFixed(2)} Max: ${accumulatorMax.toFixed(2)} Mid: ${accumulatorMid.toFixed(2)}`);
    console.log(`Epoch: ${epoch} DATA_LEN: ${DATA_LEN} MSE_AVE: ${MSE_AVE.toFixed(6)}`);
    console.log(`Norm: ${accumulatorNom.toFixed(2)}\n`);
}

//main
{
    //計測開始
    const timeStart = performance.now();
    const strDate = new Date();
    console.log(strDate.toLocaleString());

    const arrStrFile = fs.readdirSync(BATCH_PATH);

    _.forEach(arrStrFile, (strFile, idx) => {
        if (!(0 <= idx && idx <= Number.MAX_SAFE_INTEGER)) return;
        //ローカル変数初期化
        let delta_out = [];
        let delta_hid = [];
        let epoch; //学習回数
        let hid = []; //隠れノード
        let out = []; //出力ノード
        let x = undefined;
        let t = undefined;
        let v = []; //v[HID_NODE][IN_NODE]
        let w = []; //w[OUT_NODE][HID_NODE]
        let arrMSE = [];

        const strJson = fs.readFileSync(`${BATCH_PATH}/${strFile}`, 'utf8');
        const hshData = JSON.parse(strJson);
        const arrHsh = hshData["listdc"];
        const DIV_T = hshData["div"];

        x = _.map(arrHsh, hsh => {
            let arrBuf = hsh.input;
            arrBuf.push(frandBias()); //add input layer bias
            return arrBuf;
        });
        t = _.map(arrHsh, hsh => hsh.output);

        IN_NODE = x[0].length // get input length include bias
        HID_NODE = IN_NODE + 1;
        DATA_LEN = x.length;

        //中間層の結合荷重を初期化
        for (let i = 0; i < HID_NODE; i++) {
            v.push([]);
        }
        for (let i = 0; i < HID_NODE; i++) {
            for (let j = 0; j < IN_NODE; j++) {
                v[i].push(frandWeight());
            }
        }
        //出力層の結合荷重の初期化
        for (let i = 0; i < OUT_NODE; i++) {
            w.push([]);
        }
        for (let i = 0; i < OUT_NODE; i++) {
            for (let j = 0; j < HID_NODE; j++) {
                w[i].push(frandWeight());
            }
        }

        for (epoch = 0; epoch < THRESH; epoch++) {

            for (let n = 0; n < DATA_LEN; n++) {
                let arrDiff = [];
                const ret = calculateNode(n, hid, out, x, v, w);
                [hid, out] = [ret[0], ret[1]];

                for (let k = 0; k < OUT_NODE; k++) {
                    arrDiff[k] = Math.pow((t[n][k] - out[k]), 2); //平均二乗誤差
                    // Δw
                    delta_out[k] = (t[n][k] - out[k]) * out[k] * (1 - out[k]); //δ=(t-o)*f'(net); net=Σwo; δo/δnet=f'(net);
                }

                for (let k = 0; k < OUT_NODE; k++) { // Δw
                    for (let j = 0; j < HID_NODE; j++) {
                        w[k][j] += ETA * delta_out[k] * hid[j]; //Δw=ηδH
                    }
                }

                for (let i = 0; i < HID_NODE; i++) { // Δv
                    delta_hid[i] = 0;

                    for (let k = 0; k < OUT_NODE; k++) {
                        delta_hid[i] += delta_out[k] * w[k][i]; //Σδw
                    }

                    delta_hid[i] = dsigmoid(hid[i]) * delta_hid[i]; //H(1-H)*Σδw
                }

                for (let i = 0; i < HID_NODE; i++) { // Δv
                    for (let j = 0; j < IN_NODE; j++) {
                        v[i][j] += ETA * delta_hid[i] * x[n][j]; //Δu=ηH(1-H)XΣδw
                    }
                }
                arrMSE[n] = arrDiff;
            } //for DATA_LEN
        } //for epoch
        printResult(arrHsh, DIV_T, arrMSE, epoch, t, hid, out, x, v, w);
    }); // _.forEach
    //計測終了
    const timeEnd = performance.now();
    const timeSec = (timeEnd - timeStart) / 1000;
    console.log(`Time: ${Math.floor(timeSec / 60)} min ${Math.floor(timeSec % 60)} sec.\n`);
}