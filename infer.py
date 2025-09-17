import torch
from FooocusSDXLInpaintAllInOnePipeline import FooocusSDXLInpaintPipeline
from diffusers.utils import load_image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DEVICE = "cuda:0"
torch.cuda.set_device(DEVICE)

# TF-IDF
vectorizer = TfidfVectorizer()

# Example attribute dictionary
attribute_dict = {
  "object": "teapot",
  "category": "Yixing clay (zisha)",
  "shape": "round, squat body; domed lid with knob",
  "color": "dark reddish-brown",
  "finish": "matte, unglazed",
  "spout": "short, straight",
  "handle": "single loop handle",
  "saucer": "hammered/bronze-textured plate",
  "setting": "rustic wooden table, linen cloth",
  "background": "muted, soft-focus gray/wooden panel",
  "lighting": "soft diffused natural light, warm tone",
  "style": "minimalist still-life photography",
  "props": ["white porcelain cup", "autumn leaf", "flower blur"],
  "camera_angle": "slightly high or eye-level",
  "depth_of_field": "shallow (blurred background)",
  "mood": "calm, contemplative",
  "palette": ["earthy brown", "warm bronze", "neutral gray"]
}

attribute_list = []
for key, value in attribute_dict.items():
    if isinstance(value, list):
        for feature in value:
            attribute_list.append(f"{feature}")
    else:
        attribute_list.append(f"{key}: {value}")

# Example prompt
prompt = "A sks small, round Yixing clay teapot"

# TF-IDF
documents = [prompt] + attribute_list
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate cosine similarity between the prompt and attribute list
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
similarity_dict = {}
for idx, similarity in enumerate(similarities[0]):
    similarity_dict[attribute_list[idx]] = similarity

# Print similarity for each feature
for feature, similarity in similarity_dict.items():
    print(f"Similarity between prompt and '{feature}': {similarity:.4f}")

# Create a similarity matrix for heatmap visualization
similarity_matrix = np.array(similarities)  # Similarity between prompt and attributes

# Visualization: Create a heatmap for the similarity matrix
plt.figure(figsize=(10, 8))

# Use seaborn to create the heatmap
sns.heatmap(similarity_matrix, annot=True, cmap='Blues', xticklabels=[prompt] + attribute_list, yticklabels=[prompt] + attribute_list)

# # Set title and labels
# plt.title('Cosine Similarity Heatmap between Prompt and Features')
# plt.xlabel('Features and Prompt')
# plt.ylabel('Features and Prompt')
#
# # Show the heatmap
# plt.show()

# Find most relevant feature based on cosine similarity
most_relevant_feature = max(similarity_dict, key=similarity_dict.get)
relevant_prompt = f"A sks teapot  {most_relevant_feature} "

# NOT CHANGE prompt
print(f"\nMost relevant prompt: {relevant_prompt}")

# Image generation setup
lora_config = [
    {
        "model_path": f"lora/teapot/checkpoint-1000",
        "scale": 1,
        "for_raw_unet": False,
        "for_fooocus_unet": True,
    },
]

pipe = FooocusSDXLInpaintPipeline.from_pretrained(
    "frankjoshua/juggernautXL_v8Rundiffusion",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to(DEVICE)
pipe.preload_fooocus_unet(
    fooocus_model_path="./models/fooocus_inpaint/inpaint_v26.fooocus.patch",
    lora_configs=lora_config,
    add_double_sa=False,
)

img_url = f"1_1.jpg"
mask_url = f"1_2.jpg"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

# Image generation
image = pipe(
    isf_global_time=20,
    isf_global_ia=1,
    decompose_prefix_prompt="a photo of a sks ",
    sks_decompose_words=[""],
    fooocus_model_head_path="./models/fooocus_inpaint/fooocus_inpaint_head.pth",
    fooocus_model_head_upscale_path="./models/upscale_model/fooocus_upscaler_s409985e5.bin",
    pag_scale=1,
    guidance_scale=4,
    ref_image_type="no",
    double_sa_alpha=1,
    save_self_attn=False,
    save_cross_attn=False,
    fooocus_time=0.8,
    inpaint_respective_field=0.5,
    sharpness=1,
    adm_scaler_positive=1.5,
    adm_scaler_negative=0.8,
    adm_scaler_end=0.3,
    seed=42,
    image=init_image,
    mask_image=mask_image,
    prompt=relevant_prompt,  # Final relevant prompt
    negative_prompt="green",
    num_inference_steps=30,
    strength=1,
)

# Get the generated image
generated_image = image.images[0]  # The resulting image

# If it's not a PIL image, convert it to one
if not isinstance(generated_image, Image.Image):
    generated_image = Image.fromarray(generated_image)

# Show the generated image
generated_image.show()

