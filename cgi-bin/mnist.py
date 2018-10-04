#!/usr/bin/python3

"""
CGI script that accepts image urls and feeds them into a ML classifier. Results
are returned in JSON format. 
"""

import io
import subprocess
import json
import sys
import os
import re
import base64
import numpy as np
from PIL import Image
#from model import model

# Default output
res = {"result": 0,
       "data": [], 
       "error": ''}

try:
    # Get post data
    if os.environ["REQUEST_METHOD"] == "POST":
        data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))

        # Convert data url to numpy array
        img_str = re.search(r'base64,(.*)', data).group(1)
        image_bytes = io.BytesIO(base64.b64decode(img_str))
        im = Image.open(image_bytes)
        # Resize image to 28x28
        im = im.resize((28,28))
        arr = np.array(im)[:,:,0:1]

        # Normalize and invert pixel values
        #arr = (255 - arr) / 255.
        arr = (255 - arr)
        np.savetxt("image.txt", arr.reshape(784), fmt='%d')

        # Load TF trained model
        #model.load('cgi-bin/models/model.tfl')

        # Predict class
        #predictions = model.predict([arr])[0]

        # Run Arm NN model
        try:
            completed = subprocess.run(['./armnn-draw/mnist_tf_convol', '1', 'image.txt'], stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as err:
            print('ERROR:', err, file=sys.stderr)
        
        # set predictions to stderr
        predictions = completed.stderr.decode('utf-8').split()

        # Return label data
        res['result'] = 1
        results = [float(num) for num in predictions] 
        print("results: ", results, file=sys.stderr)
        print("max ", max(results), file=sys.stderr)
        maxpos = results.index(max(results))

        # Normalise result data
        probs = [x/max(results) for x in results]
        print("probabilities: : ", probs, file=sys.stderr)
        res['data'] = probs
        print("done: ", res, file=sys.stderr)

except Exception as e:
    # Return error data
    res['error'] = str(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))


