import torch 
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description ="Road damage detection Testing and producing outputs.")

    parser.add_argument("--root_dir", default="", help="Specify the path to the root folder contains test images.")
    parser.add_argument("--model_path", default="", help="Choose a prefered pretrained model(.pt)")
    parser.add_argument("--output_dir", default="saved_predictions", 
                        help="Specify the path to save model's prediction results.") 

    args = parser.parse_args()
    return args

def detect(model, img):
    res = model(img)
    print(res.xyxy[0])



if __name__=='__main__':
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.hub.load('ultralytics/yolov5', 'custom', args.model_path)
    model.to(device)

    imgs_list = os.listdir(args.root_dir)
    for img_name in imgs_list:
        pred = detect(model, img_name)
        break
