var express = require('express');
var app = express();
var fs = require('fs');
const multer = require("multer");
const path = require("path");
const util = require('util');
const exec = util.promisify(require('child_process').exec);
const {PythonShell} = require('python-shell')
const fetch = require('node-fetch');


const handleError = (err, res) => {
  res
    .status(500)
    .contentType("text/plain")
    .end("Oops! Something went wrong!");
};

const upload = multer({
  dest: "/tempsample"
  // you might also want to set some limits: https://github.com/expressjs/multer#limits
});

async function run_sample() {
  const { stdout, stderr } = await exec('python take_sample.py');
  console.log('stdout:', stdout);
  console.log('stderr:', stderr);
}

async function run_matching() {
  const { stdout, stderr } = await exec('python code.py');
  console.log('stdout:', stdout);
  console.log('stderr:', stderr);
}

app.post(
  "/upload",
  upload.single("video" /* name attribute of <file> element in your form */),
  (req, res) => {
    const tempPath = req.file.path;
    const targetPath = path.join(__dirname, "./sample/video.mp4");
    const database = JSON.parse(fs.readFileSync('db.json', 'utf8'));

    if (path.extname(req.file.originalname).toLowerCase() === ".mp4") {
      fs.rename(tempPath, targetPath, err => {
        if (err) return handleError(err, res);

        // run_sample().then(run_matching);
        PythonShell.run('take_sample.py', null, function (err) {
          if (err) throw err;
          console.log('finished');
          fetch('http://localhost:5000').then((resp) => {
            if (resp.ok) {
              resp.text().then(result => {
              const name = result.split('/')[1];
              const row = database.find(field => field.name === name)
              console.log(name, row)
              res
                .status(200)
                .send(row)
              })
            }
          })
          // PythonShell.run('code.py', null, function (err,results) {
          //   if (err) throw err;
          //   console.log('results: %j', results);
          //   const result = results && results.length > 0 && results[0];
          //   const name = result.split('/')[1];
          //   const row = database.find(field => field.name === name)
          //   console.log(name, row)
          //   res
          //     .status(200)
          //     .send(row)
          // });
        });
      })
    }
  }
);


app.get('/', function (req, res) {
  res.send('Hello World!');
});

app.listen(3000, function () {
  console.log('Example app listening on port 3000!');
});
