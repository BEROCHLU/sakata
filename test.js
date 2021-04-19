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

let DATA_LEN; //学習データ数
const ETA = 0.5; //学習係数
const THRESH = 500000;

const sigmoid = x => 1 / (1 + Math.exp(-x)); //シグモイド関数
const dsigmoid = x => x * (1 - x); //シグモイド関数微分

let hid; //隠れノード
let out; //出力ノード

let x;
let t;

let v; //v[HID_NODE][IN_NODE]
let w; //w[OUT_NODE][HID_NODE]

const BATCH_PATH = './batch';

//乱数生成
const frandWeight = () => 0.5; //  0 <= x < 1.0, Math.random()
const frandBias = () => -1;

const updateHidOut = (n) => {
    for (let i = 0; i < HID_NODE; i++) {
        hid[i] = sigmoid(math.dot(x[n], v[i]));
    }

    hid[HID_NODE - 1] = frandBias(); //配列最後にバイアス

    for (let i = 0; i < OUT_NODE; i++) {
        out[i] = sigmoid(math.dot(w[i], hid));
    }
}


const printResult = (arrHsh, DIV_T, fError, epoch) => {

    let arrErate = [];
    let accumulate;
    let accumulateMin = Number.MAX_SAFE_INTEGER;
    let accumulateMax = Number.MIN_SAFE_INTEGER;

    for (let i = 0; i < DATA_LEN; i++) {

        updateHidOut(i);

        arrErate[i] = (t[i][0] - out[0]) / t[i][0] * 100;

        accumulate = _.reduce(arrErate, (result, current) => {
            accumulateMin = (result < accumulateMin) ? result : accumulateMin; //蓄積中の最小値
            accumulateMax = (accumulateMax < result) ? result : accumulateMax; //蓄積中の最大値
            return result + current;
        });

        const undo_out = out[0] * DIV_T;
        const undo_teacher = t[i][0] * DIV_T;

        const pad_out = undo_out.toFixed(2).padStart(6);
        const pad_teacher = undo_teacher.toFixed(2).padStart(6);
        const pad_erate = arrErate[i].toFixed(2).padStart(5);
        const pad_accumulate = accumulate.toFixed(2).padStart(5);

        console.log(`${arrHsh[i].date} ${pad_out} True: ${pad_teacher} ${pad_erate}% ${pad_accumulate}`);
    }

    const averageError = _.chain(arrErate).map(Math.abs).mean().round(2).value();

    const accumulateMid = (accumulateMin + accumulateMax) / 2;
    const accumulateNom = (accumulate - accumulateMin) * 100 / (accumulateMax - accumulateMin);

    console.log(`Average error: ${averageError}%`);
    console.log(`Min: ${accumulateMin.toFixed(2)} Max: ${accumulateMax.toFixed(2)} Mid: ${accumulateMid.toFixed(2)}`);
    console.log(`Epoch: ${epoch} DATA_LEN: ${DATA_LEN}`);
    console.log(`Nom: ${accumulateNom.toFixed(2)}`);
    console.log(`FinalErr: ${fError.toFixed(5)}\n`);
}

//main
{
    //計測開始
    const timeStart = performance.now();
    const strDate = new Date();
    console.log(strDate.toLocaleString());

    const arrStrFile = fs.readdirSync(BATCH_PATH);

    _.forEach(arrStrFile, (strFile, ii) => {
        if (!(0 <= ii && ii <= Number.MAX_SAFE_INTEGER)) return;
        //グローバル変数初期化
        hid = [];
        out = [];
        [x, t] = [undefined, undefined];
        v = [];
        w = [];
        //ローカル変数初期化
        let delta_out = [];
        let delta_hid = [];
        let epoch = 0; //学習回数
        let fError = Number.MAX_SAFE_INTEGER;

        const strJson = fs.readFileSync(`${BATCH_PATH}/${strFile}`, 'utf8');
        const hshData = JSON.parse(strJson);
        const arrHsh = hshData["listdc"];
        const DIV_T = hshData["div"];

        x = arrHsh.map(hsh => {
            let arrBuf = hsh.input;
            arrBuf.push(frandBias()); //add input layer bias
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

        while (epoch < THRESH) {
            epoch++;
            fError = 0;

            for (let n = 0; n < DATA_LEN; n++) {
                updateHidOut(n);

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
        } //while
        printResult(arrHsh, DIV_T, fError, epoch);
    }); // _.forEach
    //計測終了
    const timeEnd = performance.now();
    const nSec = (timeEnd - timeStart) / 1000;
    console.log(`Time: ${Math.floor(nSec / 60)} min ${Math.floor(nSec % 60)} sec.\n`);
}