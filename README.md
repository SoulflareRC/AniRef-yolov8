# AniRef-yolov8
### What does AniRef do?
This project mainly presents a toolchain for artists to quickly extract reference images from anime videos. We first use an object detection model to crop out the characters, and then use [Deepdanbooru](https://github.com/KichangKim/DeepDanbooru) to tag a character on a subset of Danbooru Tags and then goes to identify the character in cropped out images based on the tags inferenced from a few reference images. 
### Model
We trained an object detection model for detecting anime characters based on the state-of-the-art [YOLOv8](https://github.com/ultralytics/ultralytics/tree/main). We provide 4 models that are based on 4 sizes of YOLOv8 and all of them are trained on hand-annotated dataset focused on anime screenshots. <br>
### Dataset
The dataset is first collected with [Yet-Another-Anime-Segmenter](https://github.com/zymk9/Yet-Another-Anime-Segmenter) on keyframes of anime compilation videos from the internet, and then manually corrected on Roboflow. The most recent version includes 10k images(40k after augmentation) and the datasets are available on [google drive](https://drive.google.com/drive/folders/1q1F1pJhRNboJkdi8XVVRiL7-_aeBFvTh?usp=share_link).
### Installation
1. Clone this repository 
``` 
git clone git@github.com:SoulflareRC/AniRef-yolov8.git
cd AniRef-yolov8
```
2. Create a virtual environment(optional)
``` 
python -m venv venv 
venv\Scripts\activate
```
3. Install the requirements
```
pip install -r requirements.txt
```
4. Run the UI and enjoy!
```
python gradio_interface.py
```
### Usage
