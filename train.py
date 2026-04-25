import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO training for AL_Yolo')
    parser.add_argument('--model', default='yolo11n.pt', help='pretrained model (yolo11n/s/m/l/x)')
    parser.add_argument('--data', default=str(ROOT / 'configs/AL_data.yaml'), help='dataset config')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--device', default='0', help='CUDA device (0/1/2 or cpu)')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate factor')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--resume', action='store_true', help='resume from last checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        resume=args.resume,
    )


if __name__ == '__main__':
    main()
