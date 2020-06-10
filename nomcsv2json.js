'use strict';

const _ = require('lodash');
const XLSX = require('xlsx');
const moment = require('moment');
const path = require('path');
const fs = require('fs');

let arrTrainX = [];
let arrTrainT = [];

let arrDate = [];
let arrUPRO = [];
let arrFXY = [];
let arrT1570 = [];

const DESIRED_ERROR = 0.005;
const PERIOD = 55;
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

fs.writeFileSync('./json/py225.json', JSON.stringify(arrTrainData), 'utf8');
