from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from .apps import KneeAppConfig
from django.contrib import messages
import cv2, numpy as np, json
from django.views.generic.base import TemplateView

def class_result(result):
    class_names = ""
    if result == 0:
        class_names = 'Minimal'
    elif result == 1:
        class_names = 'Healthy'
    elif result == 2:
        class_names = 'Moderate'
    elif result == 3:
        class_names = 'Doubtful'
    elif result == 4:
        class_names = 'Severe'
    return class_names

def ImageUploadTempView(request):
    if request.method == 'POST':
        # messages.success(request, 'Detected Successfully!')
        test_pixel_data = []
        image_size = 100
        imageFile = request.FILES.get('myfile', False)            
        if imageFile:
            #convert string data to numpy array
            filestr = imageFile.read()
            npimg = np.fromstring(filestr, np.uint8)
            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)               
            new_img_array = cv2.resize(img, (image_size, image_size))
            # print("****New Image Array****", new_img_array, "******")
            # print("Shape***********", new_img_array.shape)
            test_pixel_data.append(new_img_array)
            test_pixel_data = np.array(test_pixel_data)
            test_pixel_data = test_pixel_data.reshape(-1, image_size, image_size, 1)
            pred = KneeAppConfig.load_model.predict(test_pixel_data)
            result = int(np.argmax(pred))
            return HttpResponse(json.dumps({"status":"Successful", 'result':class_result(result), 'grade': result}))
        else:
            return messages.error(request, 'Upload an Image!')
    return render(request, 'index.html')

