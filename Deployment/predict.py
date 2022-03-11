import sys
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

labels = {
    1: 'Atelectasis',
    2: 'Cardiomegaly',
    3: 'Consolidation',
    4: 'Edema',
    5: 'Effusion',
    6: 'Emphysema',
    7: 'Fibrosis',
    8: 'Hernia',
    9: 'Infiltration',
    10: 'Mass',
    11: 'Nodule',
    12: 'Pleural_Thickening',
    13: 'Pneumonia',
    14: 'Pneumothorax'
    }

if __name__ == '__main__':

    # Load image for prediction
    path_to_img = sys.argv[1]
    img = load_img(path_to_img, target_size=(256, 256))
    print('Successfully loaded image - ' + path_to_img, file=sys.stdout)
    
    # Preprocess input
    print('Preprocessing image for prediction..', file=sys.stdout)
    img_arr = img_to_array(img)
    img_batch = np.expand_dims(img_arr, axis=0)
    img_processed = preprocess_input(img_batch)

    # Load model
    print('Loading model..', file=sys.stdout)
    model = load_model('./ucsd-mle-dl-prototype')
    #model.compile()

    # Debugging
    print('Predicting..', file=sys.stdout)
    predictions = model.predict(img_processed)

    print()
    print('PREDICTIONS:', file=sys.stdout)
    for i, pred in enumerate(predictions[0]):
        print(str(labels[i+1]) + ": " + f"{pred:.4f}", file=sys.stdout)