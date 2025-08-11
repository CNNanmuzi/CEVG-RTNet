import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') 
    model = YOLO(model=r'E:\Object_detection\smoke_detection\ultralytics-main\ultralytics\cfg\models\11\YOLO11-CRNet.yaml')
    model.train(data=r'E:\Object_detection\smoke_detection\ultralytics-main\firesmoke.yaml',
                imgsz=640,
                epochs=150,
                batch=16,
                workers=8,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train-CEVG-RTNet',
                name='exp',
                single_cls=False,
                cache=False,
                )
