# CatsvsDogs_ImageClassifier

A deep learning image classifier built with PyTorch and transfer learning, deployed as a Streamlit app on Hugging Face Spaces.


Model Weights: https://huggingface.co/Trisrota/cats-vs-dogs-resnet18

Live demo:https://huggingface.co/spaces/Trisrota/Catsvsdogs



In a nutshell:
1. Task: Binary image classification (between cat and dog)
2. Model: ResNet-18 (ImageNet pretrained)
3. Framework: PyTorch
4. Deployment: Streamlit on Hugging Face Spaces
5. Validation Accuracy: ~98%

The final classification layer was fine-tuned on the Kaggle Dogs vs Cats dataset. Rest of the layers remained untouched. 

How It Works:
Upload an image
The model predicts Cat or Dog
A confidence score is displayed
Model weights are downloaded dynamically from the Hugging Face Hub at runtime

To run locally:
pip install -r requirements.txt
streamlit run app.py

Tech Stack

PyTorch · Torchvision · Streamlit · Hugging Face Hub
