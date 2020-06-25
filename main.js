'use strict';

const fs = require('fs');
const _ = require('lodash');
const math = require('mathjs');
const {
    performance
} = require('perf_hooks');

let IN_NODE; //入力ノード数（バイアス含む）
let HID_NODE; //隠れノード数
const OUT_NODE = 1; //出力ノード数

const ETA = 0.5; //学習係数
const THRESH = 1000000;

const sigmoid = x => 1 / (1 + Math.exp(-x)); //シグモイド関数
const dsigmoid = x => x * (1 - x); //シグモイド関数微分

let hid = []; //隠れノード
let out = []; //出力ノード

let delta_out = [];
let delta_hid = [];

let x = [];
let t = [];

let epoch = 0; //学習回数
let DATA_LEN; //学習データ数
let fError = Number.MAX_SAFE_INTEGER;

let v = []; //v[HID_NODE][IN_NODE]
let w = []; //w[OUT_NODE][HID_NODE]

let timeStart;
let timeEnd;

//乱数生成
//const frandFix = () => math.random(0.5, 1.0); // 0.5 <= x < 1.0
const frandFix = () => Math.random(); //  0 <= x < 1.0

/**
 * 
 * @param {number} n 
 */
const calcHidOut = (n) => {
    for (let i = 0; i < HID_NODE; i++) {
        hid[i] = sigmoid(math.dot(x[n], v[i]));
    }

    hid[HID_NODE - 1] = frandFix(); //配列最後にバイアス

    for (let i = 0; i < OUT_NODE; i++) {
        out[i] = sigmoid(math.dot(w[i], hid));
    }
}

/**
 * 
 * @param {Array<Object>} arrHsh 
 * @param {number} DIV_T 
 */
const printResult = (arrHsh, DIV_T) => {

    let arrErate = [];
    let valance = 0;
    let valanceMin = Number.MAX_SAFE_INTEGER;
    let valanceMax = Number.MIN_SAFE_INTEGER;

    for (let i = 0; i < DATA_LEN; i++) {

        calcHidOut(i);

        arrErate[i] = (t[i][0] - out[0]) / t[i][0] * 100;

        valance += arrErate[i];

        const undo_out = out[0] * DIV_T;
        const undo_teacher = t[i][0] * DIV_T;

        const pad_out = undo_out.toFixed(2).padStart(6);
        const pad_teacher = undo_teacher.toFixed(2).padStart(6);
        const pad_erate = arrErate[i].toFixed(2).padStart(5);
        const pad_valance = valance.toFixed(2).padStart(5);

        console.log(`${arrHsh[i].date} ${pad_out} True: ${pad_teacher} ${pad_erate}% ${pad_valance}`);

        valanceMin = (valance < valanceMin) ? valance : valanceMin;
        valanceMax = (valanceMax < valance) ? valance : valanceMax;

    }

    const averageError = _.chain(arrErate).map(Math.abs).mean().round(2).value();
    const timeSec = (timeEnd - timeStart) / 1000;

    const valanceMid = (valanceMin + valanceMax) / 2;
    const valanceNom = (valance - valanceMin) * 100 / (valanceMax - valanceMin);

    console.log(`Average error: ${averageError}%`);
    console.log(`Min: ${valanceMin.toFixed(2)} Max: ${valanceMax.toFixed(2)} Mid: ${valanceMid.toFixed(2)}`);
    console.log(`epoch: ${epoch} DATA_LEN: ${DATA_LEN}`);
    console.log(`Nom: ${valanceNom.toFixed(2)}`);
    console.log(`Time: ${timeSec.toFixed(2)}sec. err: ${fError.toFixed(5)}`);
}

/**
 * Main
 */
{
    const strPath = './json/n225out.json';
    const strPath2 = './json/setting.json';
    const strJson = fs.readFileSync(strPath, 'utf8');
    const strJson2 = fs.readFileSync(strPath2, 'utf8');
    const arrHsh = JSON.parse(strJson);
    const hshSetting = JSON.parse(strJson2);

    x = arrHsh.map(hsh => {
        let arrBuf = hsh.input;
        arrBuf.push(frandFix()); //add input bias
        return arrBuf;
    });
    t = arrHsh.map(hsh => hsh.output);

    IN_NODE = x[0].length // get input length include bias
    HID_NODE = IN_NODE + 1;
    DATA_LEN = x.length;

    //中間層の結合荷重を初期化
    for (let i = 0; i < HID_NODE; i++) {
        v.push([]);
    }
    for (let i = 0; i < HID_NODE; i++) {
        for (let j = 0; j < IN_NODE; j++) {
            v[i].push(frandFix());
        }
    }
    //出力層の結合荷重の初期化
    for (let i = 0; i < OUT_NODE; i++) {
        w.push([]);
    }
    for (let i = 0; i < OUT_NODE; i++) {
        for (let j = 0; j < HID_NODE; j++) {
            w[i].push(frandFix());
        }
    }

    //計測開始
    timeStart = performance.now();
    const strDate = new Date();
    console.log(strDate.toLocaleString());

    while (hshSetting.DESIRED_ERROR < fError) {
        epoch++;
        fError = 0;

        for (let n = 0; n < DATA_LEN; n++) {
            calcHidOut(n);

            for (let k = 0; k < OUT_NODE; k++) {
                fError += 0.5 * Math.pow((t[n][k] - out[k]), 2); //誤差を日数分加算する
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
        }

        if (epoch % 500000 === 0) {
            epoch = epoch + '';
            console.log(`${epoch.padStart(5)}: ${_.round(fError, 6)}`);
        }
        if (THRESH <= epoch) {
            console.log(`force quit`);
            break;
        }
    } //while

    //計測終了
    timeEnd = performance.now();
    printResult(arrHsh, hshSetting.DIV_T);

}