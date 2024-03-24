from ultralytics import YOLO
from PIL import Image
import os
import warnings
import numpy as np
import shutil  # Import shutil for file copying
warnings.filterwarnings('ignore')

def predict_images(input_folder, output_folder):
    # Load a model
#     model = YOLO('endx/train_split1/weights/best.pt')  # load a custom model
#     model = YOLO('kuochong/train_split0/weights/best.pt')
    model = YOLO('3500/8_22/weights/best.pt')
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Predict with the model for each image
    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        output_txt_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.txt')  # Create a .txt file path
        output_tp_path = os.path.join(output_folder, image_file)

        # Predict on the image
        results = model(input_image_path, save=False,iou=0.5,conf=0.3,augment=True)  # Predict without saving

        # Extract prediction data and save to .txt file
        for i in results:
            result_array = np.concatenate((i.boxes.cls.cpu().numpy()[:, np.newaxis], i.boxes.xywhn.cpu().numpy()), axis=1)
            # 保存为txt文件，第一列为整数，后面的列为浮点数（保留6位小数），使用空格分隔
            fmt = ['%d'] + ['%.6f'] * (result_array.shape[1] - 1)
            delimiter = ' '  # 使用空格分隔
            np.savetxt(output_txt_path, result_array, delimiter=delimiter, fmt=fmt)

#         for r in results:
#             im_array = r.plot()  # plot a BGR numpy array of predictions
#             im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#             im.show()  # show image
#             im.save(output_tp_path)  # save image
        # Copy the original image to the output folder 
#         shutil.copy(input_image_path, output_folder)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python your_script.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    predict_images(input_folder, output_folder)
