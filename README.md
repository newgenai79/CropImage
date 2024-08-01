# CropImage

### Credits for image crop code to [EdTalk](https://github.com/tanshuai0219/EDTalk)

Step 1: Clone the repository

```
git clone https://github.com/newgenai79/CropImage
```

Step 2: Navigate inside the cloned repository

```
cd CropImage
```

Step 3: Create virtual environment

```
python -m venv venv
```

Step 4: Activate virtual environment
```
venv\Scripts\activate
```

Step 5: Install requirements

```
pip install -r requirements.txt
```

Step 6: Download weights
Download and place in root directory

```
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
```

Download and place in data_preprocess folder

```
https://raw.githubusercontent.com/tanshuai0219/EDTalk/main/data_preprocess/M003_template.npy
```

Step 7: Launch Gradio based WebUI

```
python app.py
```