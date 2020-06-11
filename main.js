'use strict';

const fs = require('fs');
const _ = require('lodash');
const math = require('mathjs');
const moment = require('moment');
const {
    performance
} = require('perf_hooks');
//const crypto = require('crypto');

const IN_NODE = 3; //入力ノード数（バイアス含む）
const HID_NODE = 4; //隠れノード数
const OUT_NODE = 1; //出力ノード数

const ETA = 0.5; //学習係数
const DESIRED_ERROR = 0.005; //期待値誤差
const ACTIVE = 0; //0: sigmoid 1: ReLU
const PERIOD = 55;
const THRESH = 1000000;

const sigmoid = x => 1 / (1 + Math.exp(-x)); //シグモイド関数
const dsigmoid = x => x * (1 - x); //シグモイド関数微分
const dfmax = x => (0 < x) ? 1 : 0; //ReLU関数微分

let hid = []; //隠れノード
let out = []; //出力ノード

let delta_out = [];
let delta_hid = [];

let x = [];
let t = [];

let epoch = 0; //学習回数
let days; //学習データ数
let fError = Number.MAX_SAFE_INTEGER;

let v = []; //v[HID_NODE][IN_NODE]
let w = []; //w[OUT_NODE][HID_NODE]

let timeStart;
let timeEnd;

//乱数生成
//const frandFix = () => math.random(0.5, 1.0); // 0.5 <= x < 1.0
const frandFix = () => Math.random(); //  0 <= x < 1.0

/**
 * 隠れ層、出力層の計算
 */
const findHiddenOutput = (n) => {
    const MU = 1; //accelerate

    for (let i = 0; i < HID_NODE; i++) {
        if (ACTIVE === 0) {
            hid[i] = sigmoid(math.dot(x[n], v[i])) * MU;
        } else {
            hid[i] = Math.max(0, math.dot(x[n], v[i]));
        }
    }

    hid[HID_NODE - 1] = frandFix(); //1 | frandFix() 配列最後にバイアス

    for (let i = 0; i < OUT_NODE; i++) {
        out[i] = sigmoid(math.dot(w[i], hid));
    }
}
/**
 * 結果表示
 */
const printResult = (arrDate, TEACH_DIV) => {

    let arrErate = [];
    let valance = 0;
    let valanceMin = Number.MAX_SAFE_INTEGER;
    let valanceMax = Number.MIN_SAFE_INTEGER;

    for (let i = 0; i < days; i++) {

        findHiddenOutput(i);

        arrErate[i] = (t[i][0] - out[0]) / t[i][0] * 100;

        valance += arrErate[i];

        const undo_out = out[0] * TEACH_DIV;
        const undo_teacher = t[i][0] * TEACH_DIV;

        const pad_out = undo_out.toFixed(2).padStart(6);
        const pad_teacher = undo_teacher.toFixed(2).padStart(6);
        const pad_erate = arrErate[i].toFixed(2).padStart(5);
        const pad_valance = valance.toFixed(2).padStart(5);

        console.log(`${arrDate[i]} ${pad_out} True: ${pad_teacher} ${pad_erate}% ${pad_valance}`);

        valanceMin = (valance < valanceMin) ? valance : valanceMin;
        valanceMax = (valanceMax < valance) ? valance : valanceMax;

    }

    const averageError = _.chain(arrErate).map(Math.abs).mean().round(2).value();
    const timeSec = (timeEnd - timeStart) / 1000;

    const valanceMid = (valanceMin + valanceMax) / 2;
    const valanceNom = (valance - valanceMin) * 100 / (valanceMax - valanceMin);

    console.log(`Average error: ${averageError}%`);
    console.log(`Min: ${valanceMin.toFixed(2)} Max: ${valanceMax.toFixed(2)} Mid: ${valanceMid.toFixed(2)}`);
    console.log(`epoch: ${epoch} days: ${days}`);
    console.log(`Time: ${timeSec.toFixed(2)}sec. err: ${fError.toFixed(5)}`);
    console.log(`Nom: ${valanceNom.toFixed(2)}`);
}
/**
 * Main
 */
{
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
    } // _.zip

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

    for (let i = 0; i < days; i++) {
        const _x0 = arrChangeUPRO[i] / UPRO_div;
        const _x1 = arrChangeFXY[i] / FXY_div;
        const _x2 = -1; //配列最後にバイアス

        const _arrX = [_x0, _x1, _x2];
        const _arrT = [arrChangeT1570[i] / T1570_div];

        x.push(_arrX);
        t.push(_arrT);
    } // _.zip

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
    console.log(moment().format('YYYY-MM-DD HH:mm:ss'));

    while (DESIRED_ERROR < fError) {
        epoch++;
        fError = 0;

        for (let n = 0; n < days; n++) {
            findHiddenOutput(n);

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

                if (ACTIVE === 0) {
                    delta_hid[i] = dsigmoid(hid[i]) * delta_hid[i]; //H(1-H)*Σδw
                } else {
                    delta_hid[i] = dfmax(hid[i]) * delta_hid[i]; //H(1-H)*Σδw
                }
            }

            for (let i = 0; i < HID_NODE; i++) { // Δv
                for (let j = 0; j < IN_NODE; j++) {
                    v[i][j] += ETA * delta_hid[i] * x[n][j]; //Δu=ηH(1-H)XΣδw
                }
            }
            epoch = epoch + 0; //debug
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
    printResult(arrDate, T1570_div);

}