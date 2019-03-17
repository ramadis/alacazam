## Alacazam

A tool to find videos from similar video clips

## Set up

1) Run `yarn install`
2) Run `pip install -r requirements.txt`

## Running

To generate model db from raw videos: `python generate_training.py`
To run the processing server: `FLASK_APP=code.py flask run`
To run the public API: `sudo node server.js`

Then, the endpoint `http://localhost:3000/upload` expects a MP3 video in a FormData body, with the field name `video`. The endpoint returns an object with a link to the complete video:

```
{ 
  "name": "sprite",
  "link": "https://drive.google.com/uc?export=download&id=16frYcF91IxDuHSseZwemWTvWwO3Ynr47",
  "description": "Sprite: Love wins"
}
```