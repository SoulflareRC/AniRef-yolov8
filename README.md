# AniRef-yolov8
![akihito](https://user-images.githubusercontent.com/107384280/225947977-94856df1-dfb7-4eb8-a4a2-ce8edf72edaf.png)
### What does AniRef do?
This project mainly presents a toolchain for artists to quickly extract reference images from anime videos. We first use an object detection model to crop out the characters, and then use [Deepdanbooru](https://github.com/KichangKim/DeepDanbooru) to tag a character on a subset of Danbooru Tags and then goes to identify the character in cropped out images based on the tags inferenced from a few reference images. 
##### Currently Supported Features
- Character Detection
- Character Identification
- Image Restoration using [Real-ESRGAN ncnn Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
- Image Grids
- Line Art Extraction
### Model
We trained an object detection model for detecting anime characters based on the state-of-the-art [YOLOv8](https://github.com/ultralytics/ultralytics/tree/main). We provide 4 models that are based on 4 sizes of YOLOv8 and all of them are trained on hand-annotated dataset focused on anime screenshots. Detailed metrics of each model can be found in [validation_metrics](https://github.com/SoulflareRC/AniRef-yolov8/tree/main/validation_metrics) <br>
| Model                     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>test<br>50-95 | Speed<br><sup>RTX 3060 12G<br>(ms)  | 
| ------------------------- | --------------------- | -------------------- | ----------------------| ----------------------------------- | 
| [AniRef40000-n-epoch40](https://github.com/SoulflareRC/AniRef-yolov8/releases/download/model/AniRef40000-n-epoch40.pt) | 640                   | 49.9                 | 45.9                  | 3.0                                 | 
| [AniRef40000-s-epoch40](https://github.com/SoulflareRC/AniRef-yolov8/releases/download/model/AniRef40000-s-epoch40.pt) | 640                   | 52.2                 | 50.6                  | 6.4                                 | 
| [AniRef40000-m-epoch75](https://github.com/SoulflareRC/AniRef-yolov8/releases/download/model/AniRef40000-m-epoch75.pt) | 640                   | 50.2                 | 48.2                  | 15.0                                |
| [AniRef40000-l-epoch50](https://github.com/SoulflareRC/AniRef-yolov8/releases/download/model/AniRef40000-l-epoch50.pt) | 640                   | 52.9                 | 50.5                  | 23.1                                | 
### Known Issues
  - Performance of the models worsens when there exists too many overlapping characters. 
  - AniRef40000-m-epoch75 was trained on an older version of the dataset for 40 epochs and needs a retrain. 
### Dataset
The lastest raw dataset is now uploaded on [Kaggle](https://www.kaggle.com/datasets/ruochongchen69/anidet-7000). The dataset is first collected with [Yet-Another-Anime-Segmenter](https://github.com/zymk9/Yet-Another-Anime-Segmenter) on keyframes of anime compilation videos from the internet, and then manually corrected on Roboflow. The most recent version includes 10k images(40k after augmentation) and the datasets are available on [google drive](https://drive.google.com/drive/folders/1q1F1pJhRNboJkdi8XVVRiL7-_aeBFvTh?usp=share_link).
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
See the [project wiki](https://github.com/SoulflareRC/AniRef-yolov8/wiki) for detailed instructions on how to use each function of the app. We also provide a demo video for how to use this program. 


[![image](https://user-images.githubusercontent.com/107384280/226233017-0921e78e-d33c-41ee-89c4-676779fa14d7.png)](https://youtu.be/mNdLELk8-zA)

