# Face-Recognition-Alert-System-by-OpenCV

## Introduce
A real-time face recognition and alert system implemented with OpenCV. It detects faces via camera, judges whether the face is known/unknown based on a confidence threshold (80), and automatically sends alert notifications to WeChat Work when an unknown person is detected (confidence > 80).

## Data Preparation
A single training image (1.jpg) is used to train the face recognition model（just ）, and the trained model file is stored in the `trainer` folder.
> **Tips**: For better recognition accuracy, it is recommended to use **multiple training images** (different angles, lighting conditions) of the target person instead of a single image.

File structure:
./
├── face/ 
│ └──  1.jpg

├── trainer/
│ └── trainer.yml 

├── haarcascade_frontalface_default.xml 

├── main.py 

└── train.py 

## Training
Run the training script to generate the face recognition model (`trainer.yml`) in the `trainer` folder:
```bash python train.py```

## Running the System
Execute the main script to start real-time face recognition and alert monitoring:
```bash python main.py```
