import torch
import clip
from PIL import Image
import pandas as pd

# 모델 및 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지 파일 이름과 텍스트 설명
image_files = ["cat.jpg", "dog.jpg", "lion.jpg", "tiger.jpg"]
text_descriptions = [
    "a photo of a cat crossing the street",
    "a photo of puppies lying on the grass",
    "a photo of two lion cubs playing",
    "a photo of a tiger with her cubs, stretching"
]

# 텍스트 특징 벡터
with torch.no_grad():
    text_tokens = clip.tokenize(text_descriptions).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 이미지별 유사도 측정
similarity_matrix = []
for img_file in image_files:
    image = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        similarity = (image_feature @ text_features.T).squeeze().cpu().numpy()
        similarity_matrix.append(similarity)

# 결과 출력
df = pd.DataFrame(similarity_matrix, index=image_files, columns=text_descriptions)
print("\n=== CLIP Image-Text Similarity Matrix ===\n")
print(df)