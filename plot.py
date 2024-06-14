import glob
from PIL import Image
from ultralytics import YOLO

def predict(model, image_path, save_path):
    # Run inference on 'bus.jpg'
    image_files = glob.glob(image_path)
    results = model(image_files)  # results list
    
    # Visualize the results
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image    
        # Save results to disk
        name = image_files[i].split('/')[-1]
        r.save(filename=f"{save_path}/{name}")
