from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from PIL import Image
import os
import torch
from torchvision import transforms, models
import cv2
from django.conf import settings

def index(request):
    return render(request, 'index.html')

def analyze(request):
    if request.method == 'POST':
        uploaded_image = request.FILES['image_input']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        static_image_path = 'static/' + uploaded_image.name
        fs.save(static_image_path, uploaded_image)

        # Load the trained model
        model = models.resnet152(pretrained=False) # Ensure the model is not pretrained
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 6) 
        model.load_state_dict(torch.load('E:/Haegl/Haegl ML Projects/CHILLI-DISEASE-DETECTION/best.pth'))
        model.eval()

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        class_mapping = {
            0: 'Healthy',
            1: 'Leaf Curl',
            2: 'Leaf Spot',
            3: 'Powdery Mildew',
            4: 'WhiteFly',
            5: 'Yellowish'
        }

        with torch.no_grad():
            image = cv2.imread(os.path.join(settings.MEDIA_ROOT, uploaded_image.name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = transforms.ToPILImage()(image)
            input_tensor = test_transform(image).unsqueeze(0)

            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = class_mapping[predicted.item()]

            params = {'label': predicted_label, 'img_pth': uploaded_image.name}
            os.remove(os.path.join(settings.MEDIA_ROOT, filename))
            return render(request, 'result.html', params)

def delete(request):
    return render(request, 'index.html')
