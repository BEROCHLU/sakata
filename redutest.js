const _ = require('lodash');

let arrErate = [37.13, 100.11, 24.77];
let accumulateMin = Number.MAX_SAFE_INTEGER;
let accumulateMax = Number.MIN_SAFE_INTEGER;

accumulate = _.reduce(arrErate, (presum, current) => {
    let result = presum + current;

    accumulateMin = (result < accumulateMin) ? result : accumulateMin; //蓄積中の最小値
    accumulateMax = (accumulateMax < result) ? result : accumulateMax; //蓄積中の最大値
    
    console.log(result);
    return result;
});

const accumulateMid = (accumulateMin + accumulateMax) / 2;
const accumulateNom = (accumulate - accumulateMin) * 100 / (accumulateMax - accumulateMin);

console.log(arrErate);
console.log(accumulate);
console.log(`Min: ${accumulateMin.toFixed(2)} Max: ${accumulateMax.toFixed(2)} Mid: ${accumulateMid.toFixed(2)}`);
console.log(`Nom: ${accumulateNom.toFixed(2)}`);