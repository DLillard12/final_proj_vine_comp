# cool! now running on a subset of the data.

# it will take about 3 hours for this to run on all the thumbnails on a CPU
import os
import time
import torch
import clip
from PIL import Image

vine_thumbnails_dir = "C:\\file_struct\\documents\\utk_omscs\\cosc545_digital_archeology\\final_project\\raw_files\\thumbs"


files = os.listdir(vine_thumbnails_dir)

print(f"Found {len(files)} files in {vine_thumbnails_dir}")

# files = files[:10]  # limit for testing

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

prob_dict = {}

start_time = time.time()

for file in files:
    file_path = os.path.join(vine_thumbnails_dir, file)
    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["a person", "a dog", "a cat", "other"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    prob_dict[file] = list(zip(probs[0].tolist(), ["a person", "a dog", "a cat", "other"]))

# print(prob_dict)

with open("clip_probs.txt", "w") as f:
    for file, probs in prob_dict.items():
        f.write(f"{file}: {probs}\n")


end_time = time.time()
print(f"Processed {len(files)} files in {end_time - start_time:.2f} seconds")