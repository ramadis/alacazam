## Alacazam

A tool to find videos from similar video clips

## Set up

1) Run `yarn install`
2) Run `pip install -r requirements.txt`
3) Create folder `tf_models`, and download the next models into it: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy and ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy

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

## Running on mobile

1) Download Expo app.
2) Open https://expo.io/@ramadis/abracadabra
3) Send the sample videos to your cellphone and load them through the app