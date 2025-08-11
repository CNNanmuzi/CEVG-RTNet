import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\Object_detection\smoke_detection\ultralytics-main\runs\train-CEVG-RTNet\exp\weights\best.pt')
    model.val(data=r'E:\Object_detection\smoke_detection\ultralytics-main\firesmoke.yaml',
              split='val',
              imgsz=640,
              batch=1,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val-CEVG-RTNet',
              name='exp',
              )