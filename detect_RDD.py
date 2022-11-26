import torch 
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description ="Road damage detection Testing and producing outputs.")

    parser.add_argument("--root_dir", default="", help="Specify the path to the root folder contains test images.")
    parser.add_argument("--model_path", default="", help="Choose a prefered pretrained model(.pt)")
    parser.add_argument("--out_name", default="result", 
                        help="Specify the result txt filename.") 

    args = parser.parse_args()
    return args



if __name__=='__main__':
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.hub.load('ultralytics/yolov5', 'custom', args.model_path)
    model.to(device)

    open(os.path.join(args.output_dir, "_prediction.txt"), "w").close()

    imgs_list = os.listdir(args.root_dir)
    for img_name in imgs_list:
        img_path = os.path.join(args.root_dir, img_name)
        pred = model(img_path)
        result = pred.pandas().xyxy[0]
        with open(os.path.join(args.output_dir, "_prediction.txt"), "a") as f:
            f.write(str(img_name)+",")
            for i, row in result.iterrows():
                if i == 5:
                    break
                
                f.write(str(row['class'])+" "+str(round(row['xmin']))+" "+str(round(row['ymin']))
                        +" "+str(round(row['xmax']))+" "+str(round(row['ymax']))+" ")
            f.write('\n')

        

