from ultralytics.yolo.v8.detect import DetectionTrainer
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics import YOLO
from pathlib import Path
#train model
if __name__ == "__main__":

    model = YOLO(r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\yolov8\runs\detect\train6\weights\epoch240.pt")
    # data_path = r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\datasets\AniDet2-1200\data.yaml"
    # wdir = r"D:\pycharmWorkspace\squirrelDet\train_res"
    # trainer:BaseTrainer = DetectionTrainer(cfg=r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\train\default.yaml")
    # trainer.save_dir =Path ("save_dir_train")
    # trainer.train()
    # print(trainer.best)
    # model.train(data=data_path,epochs=6,workers=1,save_period = 2,save=True )
    metrics = model.val()
    # results = model("chonky/test/images/1165cjve4jfa1-2-_jpg.rf.02e3c28e063fd51ca43fafb38d2e7d03.jpg")


