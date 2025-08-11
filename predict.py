import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\Object_detection\smoke_detection\ultralytics-main\runs\train-CEVG-RTNet\exp\weights\best.pt') # select your model.pt path
    model.predict(source=r'E:\Object_detection\smoke_detection\ultralytics-main\VIGP_FS_datasets\test\images',
                  imgsz=640,
                  project='runs-CEVG-RTNet/detect',
                  name='result',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )