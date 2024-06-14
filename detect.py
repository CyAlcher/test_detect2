from ultralytics import YOLO
from plot import predict

if __name__ == "__main__":
    model = YOLO('yolov8n.yaml')  # build a new model from scratch
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    data_config_path = '/home/huangping/scratch/3DAnomalyDetect/exp/3d_printer.yaml'
    results = model.train(data=data_config_path, epochs=400)  # train the model
    val_results = model.val(data=data_config_path, mode='val')
    train_results = model.val(data=data_config_path, split='train')
    test_results = model.val(data=data_config_path, split='test')

    predict(model, image_path='/home/huangping/scratch/3DAnomalyDetect/data/frames/spaghetti1/*.jpg', 
            save_path='/home/huangping/scratch/3DAnomalyDetect/exp/results/spaghetti1')
    predict(model, image_path='/home/huangping/scratch/3DAnomalyDetect/data/frames/spaghetti2/*.jpg', 
            save_path='/home/huangping/scratch/3DAnomalyDetect/exp/results/spaghetti2')
