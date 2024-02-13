'use strict';

const fs = require('fs');
const _ = require('lodash');
const { performance } = require('perf_hooks');

let IN_NODE; //入力ノード数（バイアス含む）
let HID_NODE; //隠れノード数
let OUT_NODE = 1; //出力ノード数

const ETA = 0.5; //学習係数
const THRESH = 500000;

let epoch; //学習回数
let DATA_LEN; //学習データ数

let hid = []; //隠れノード
let out = []; //出力ノード

let delta_out = [];
let delta_hid = [];

let x = [];
let t = [];
let v = []; //v[HID_NODE][IN_NODE]
let w = []; //w[OUT_NODE][HID_NODE]

let timeStart;
let timeEnd;

const sigmoid = x => 1 / (1 + Math.exp(-x)); //シグモイド関数
const dsigmoid = x => x * (1 - x); //シグモイド関数微分
//乱数生成
const frandWeight = () => 0.5; //  0 <= x < 1.0, Math.random()
const frandBias = () => -1;
//内積計算
const dotProduct = (vec1, vec2) => {
    return vec1.reduce((acc, current, index) => acc + current * vec2[index], 0);
}


function calculateNode(n) {
    for (let i = 0; i < HID_NODE; i++) {
        hid[i] = sigmoid(dotProduct(x[n], v[i]));
    }

    hid[HID_NODE - 1] = frandBias(); //配列最後にバイアス

    for (let i = 0; i < OUT_NODE; i++) {
        out[i] = sigmoid(dotProduct(w[i], hid));
    }
}


function printResult(arrHsh, DIV_T, errorLSM) {

    let arrErate = [];
    let accumulator;
    let accumulatorMin = Number.MAX_SAFE_INTEGER;
    let accumulatorMax = Number.MIN_SAFE_INTEGER;

    for (let i = 0; i < DATA_LEN; i++) {

        calculateNode(i); //最終的なNode計算

        arrErate[i] = (t[i][0] - out[0]) / out[0] * 100;

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
    const timeSec = (timeEnd - timeStart) / 1000;

    const accumulatorMid = (accumulatorMin + accumulatorMax) / 2;
    const accumulatorNom = (accumulator - accumulatorMin) * 100 / (accumulatorMax - accumulatorMin);

    console.log(`Average error: ${averageError}%`);
    console.log(`Min: ${accumulatorMin.toFixed(2)} Max: ${accumulatorMax.toFixed(2)} Mid: ${accumulatorMid.toFixed(2)}`);
    console.log(`Epoch: ${epoch} DATA_LEN: ${DATA_LEN} FinalLSM: ${errorLSM.toFixed(5)}`);
    console.log(`Norm: ${accumulatorNom.toFixed(2)}`);
    console.log(`Time: ${timeSec.toFixed(2)}sec.\n`);
}

//main
{
    const strJson = fs.readFileSync('./json/seikika.json', 'utf8');
    const hshData = JSON.parse(strJson);
    const arrHsh = hshData["listdc"];
    const DIV_T = hshData["div"];
    let errorLSM;

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

    //計測開始
    timeStart = performance.now();
    const strDate = new Date();
    console.log(strDate.toLocaleString());

    for (epoch = 0; epoch < THRESH; epoch++) {
        errorLSM = 0;

        for (let n = 0; n < DATA_LEN; n++) {
            calculateNode(n);

            for (let k = 0; k < OUT_NODE; k++) {
                errorLSM += 0.5 * Math.pow((t[n][k] - out[k]), 2); //最小二乗法
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
        } // for DATA_LEN
        if (epoch % 100 === 0) { //logging
            const s = epoch + '';
            //console.log(`${errorLSM.toFixed(5)}`);
        }
    } //for epoch
    //計測終了
    timeEnd = performance.now();
    printResult(arrHsh, DIV_T, errorLSM);
}