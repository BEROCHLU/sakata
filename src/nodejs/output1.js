'use strict';

const fs = require('fs');
const _ = require('lodash');
const { performance } = require('perf_hooks');

let IN_NODE; //入力ノード数（バイアス含む）
let HID_NODE; //隠れノード数
let OUT_NODE = 1; //出力ノード数は1とする

let DATA_LEN; //学習データ数
const ETA = 0.5; //学習係数
const THRESH = 500000; //学習回数
const BATCH_PATH = './batch'; // バッチ処理のデータが保存されているパス

const sigmoid = x => 1 / (1 + Math.exp(-x)); //シグモイド関数
const dsigmoid = x => x * (1 - x); //シグモイド関数微分

const INI_WEIGHT = 0.5; //  0 <= x < 1.0, Math.random()
const BIAS = -1;
//超高速内積計算
const dotProduct = (vec1, vec2) => {
    return vec1.reduce((acc, current, index) => acc + current * vec2[index], 0);
}

// ノードの計算を行う関数
function calculateNode(n, hid, out, x, v, w) {
    for (let i = 0; i < HID_NODE; i++) {
        hid[i] = sigmoid(dotProduct(x[n], v[i]));// 隠れ層の活性化
    }

    hid[HID_NODE - 1] = BIAS; //配列最後にバイアス

    for (let i = 0; i < OUT_NODE; i++) {
        out[i] = sigmoid(dotProduct(w[i], hid)); // 出力層の活性化
    }

    return [hid, out];
}

// 結果を表示する関数
function printResult(arrHsh, DIV_T, errorLSM, epoch, t, hid, out, x, v, w) {

    let arrErate = [];
    let accumulator;
    let accumulatorMin = Number.MAX_SAFE_INTEGER;
    let accumulatorMax = Number.MIN_SAFE_INTEGER;

    console.log('      date predic actual  diff   acc'); //header

    for (let i = 0; i < DATA_LEN; i++) {

        const ret = calculateNode(i, hid, out, x, v, w);
        [hid, out] = [ret[0], ret[1]];

        arrErate[i] = (t[i][0] - out[0]) / out[0] * 100; //t[i][0] | out[0]

        accumulator = _.reduce(arrErate, (presum, current) => {
            accumulatorMin = Math.min(accumulatorMin, presum); //前回の蓄積結果で最小値を更新
            accumulatorMax = Math.max(accumulatorMax, presum); //前回の蓄積結果で最大値を更新
            return presum + current; // 配列最後のreturnは最大最小の更新対象にならない
        });

        const undo_out = out[0] * DIV_T;// 出力値の変換
        const undo_teacher = t[i][0] * DIV_T;// 教師データの変換

        // 表示用に値を整形
        const pad_out = undo_out.toFixed(2).padStart(6);
        const pad_teacher = undo_teacher.toFixed(2).padStart(6);
        const pad_erate = arrErate[i].toFixed(2).padStart(5);
        const pad_accumulator = accumulator.toFixed(2).padStart(5);

        // 結果の表示
        console.log(`${arrHsh[i].date} ${pad_out} ${pad_teacher} ${pad_erate} ${pad_accumulator}`);
    }
    // 平均絶対誤差の計算と表示
    const averageError = _.chain(arrErate).map(Math.abs).mean().round(2).value();
    // 累積誤差の中間値、正規化値の計算
    //const accumulatorMid = (accumulatorMin + accumulatorMax) / 2;
    const accumulatorNom = (accumulator - accumulatorMin) * 100 / (accumulatorMax - accumulatorMin);
    // 追加の統計情報の表示
    console.log(`Mean Absolute Error: ${averageError}%`);
    // console.log(`Min: ${accumulatorMin.toFixed(2)} Max: ${accumulatorMax.toFixed(2)} Mid: ${accumulatorMid.toFixed(2)}`);
    console.log(`Epoch: ${epoch} BatchSize: ${DATA_LEN} FinalLSM: ${errorLSM.toFixed(5)}`);
    console.log(`Norm: ${accumulatorNom.toFixed(2)}`);
    console.log(`===`);
}

// main関数的な挙動をするブロック
(async () => {
    //計測開始
    const timeStart = performance.now();
    const strDate = new Date();
    console.log(strDate.toLocaleString());// 現在時刻の表示
    // バッチ処理ファイルの読み込み
    const arrStrFile = await fs.promises.readdir(BATCH_PATH);

    for (const strFile of arrStrFile) {
        //各種変数の初期化
        let delta_out = [];
        let delta_hid = [];
        let epoch; //学習回数
        let hid = []; //隠れノード
        let out = []; //出力ノード
        let x = undefined;
        let t = undefined;
        let v = []; //v[HID_NODE][IN_NODE], 中間層の重み
        let w = []; //w[OUT_NODE][HID_NODE], 出力層の重み
        let errorLSM; //最小二乗法の誤差

        // バッチファイルの読み込みとJSONへの変換
        const strJson = await fs.promises.readFile(`${BATCH_PATH}/${strFile}`, 'utf8');
        const hshData = JSON.parse(strJson);
        const arrHsh = hshData["listdc"];
        const DIV_T = hshData["div"];

        // 入力データと教師データの準備
        x = _.map(arrHsh, hsh => {
            let arrBuf = hsh.input;
            arrBuf.push(BIAS); // 入力層にバイアスを追加
            return arrBuf;
        });
        t = _.map(arrHsh, hsh => hsh.output);

        // ノード数の設定
        IN_NODE = x[0].length // 入力層のノード数（バイアス含む）
        HID_NODE = IN_NODE + 1;// 隠れ層のノード数
        DATA_LEN = x.length;// データ数

        //中間層の結合荷重を初期化
        for (let i = 0; i < HID_NODE; i++) {
            v.push([]);
        }
        for (let i = 0; i < HID_NODE; i++) {
            for (let j = 0; j < IN_NODE; j++) {
                v[i].push(INI_WEIGHT);
            }
        }
        //出力層の結合荷重の初期化
        for (let i = 0; i < OUT_NODE; i++) {
            w.push([]);
        }
        for (let i = 0; i < OUT_NODE; i++) {
            for (let j = 0; j < HID_NODE; j++) {
                w[i].push(INI_WEIGHT);
            }
        }
        // 学習プロセス
        for (epoch = 0; epoch < THRESH; epoch++) {
            errorLSM = 0;

            for (let n = 0; n < DATA_LEN; n++) {
                const ret = calculateNode(n, hid, out, x, v, w);
                [hid, out] = [ret[0], ret[1]];

                for (let k = 0; k < OUT_NODE; k++) {
                    errorLSM += 0.5 * Math.pow((t[n][k] - out[k]), 2); //最小二乗法による誤差の計算
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
            } //データセット全体の学習終了
        } //学習サイクル終了
        printResult(arrHsh, DIV_T, errorLSM, epoch, t, hid, out, x, v, w);
    }; //バッチファイルごとの処理終了
    //計測終了
    const timeEnd = performance.now();
    const timeSec = (timeEnd - timeStart) / 1000;
    console.log(`Time: ${Math.floor(timeSec / 60)} min ${Math.floor(timeSec % 60)} sec.\n`);
})();