# Web app for real time face mask detection

## Environment setup
1. Make sure your python version is `3.7`  
2. Install dependencies with `pip`: 
```bash
pip install -r requirements.txt
```

## Configure HTTPS
You need to get a license to enable https protocol. Check [Let's encrypt](https://letsencrypt.org/) on how to obtain a free license.  
If you just want to start and test the app locally, change the last line of `app.py` from  
```python
socketio.run(app, host='0.0.0.0', ssl_context=('fullchain.pem', 'privkey.pem'), port=5000)
```

to
```python
socketio.run(app, host='0.0.0.0', port=5000)
```

## Download MobileNetV3 weights
Download from [Google Drive](https://drive.google.com/file/d/1GAkFJewmuDRasZ7XGh4SxlP4Z8gNxeka/view?usp=sharing)

## Start web app
```python
python app.py
```

## Acknowledgement
Inspired from https://github.com/afifsauqil/FaceMaskDetectionWebApp
