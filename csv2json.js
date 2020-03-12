'use strict';

const csv2json = require('csv2json');
const fs = require('fs');
const path = require('path');

const CSV_PATH = path.join('T:\\ProgramFilesT\\pleiades\\workspace\\node225', 'nt1570.csv');

fs.createReadStream(CSV_PATH)
    .pipe(csv2json({
        // Defaults to comma.
        //separator: ';'
    }))
    .pipe(fs.createWriteStream('./json/nt1570.json'));