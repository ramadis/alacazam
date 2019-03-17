## Alacazam

A tool to find videos from similar video clips

## Set up

1) Run `yarn install`
2) Run `pip install -r requirements.txt`

## Running

To generate model db from raw videos: `python generate_training.py`
To run the processing server: `FLASK_APP=code.py flask run`
To run the public API: `sudo node server.js`