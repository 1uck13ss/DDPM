In this post, I’ll be sharing my exploration of Denoising Diffusion Probabilistic Models (DDPM), inspired by the original paper. I’ve built a personal implementation using PyTorch, and this project has helped solidify my understanding of how generative models learn to reverse noise into meaningful images. 

# DDPM
At its core, a DDPM trains a model to generate data (like images) by learning how to gradually remove noise from pure randomness. This happens in three key phases:

Forward Process — Adding Noise: Starting with a clean image, we slowly inject random noise step-by-step until it's completely scrambled.

Training — Learning to Remove Noise: The model is trained to predict the noise at each step using noisy images and their original counterparts. This teaches it how to reconstruct clean images from noise.

Reverse Process — Denoising: Once trained, the model can take pure noise and iteratively denoise it to generate a realistic image.

## Model
Following the original paper, a UNet model is used for structured image generation and reconstruction. 
The U-Net model can be broken down into three parts
Encoder: This part compresses the input image, capturing deep hierarchical features.
Decoder: It then reconstructs the image by expanding those compressed features back to the original shape.
Skip Connections: These direct links between encoder and decoder layers help preserve fine-grained spatial details that would otherwise be lost.
Following further works from the likes of the 2023 deep attention U net, I was inspired to use self attention layers that allowed the model to focus on the relationships between distant parts of the images
UNet’s balance of local and global representation makes it ideal for diffusion-based generation, where every small detail and broad structure matter.

I trained my model on CIFAR-10 images, specifically the fighter planes. The images below are the results of DDPM for every 100 epochs. From this experiment, we can see that the model is able to produce more coherent images.

<img width="622" height="215" alt="image" src="https://github.com/user-attachments/assets/48103623-34de-44cb-9e56-7882fb01b4f8"/>
<img width="627" height="197" alt="image" src="https://github.com/user-attachments/assets/3c1fad65-8b7a-4c67-9597-6405231d6d05" />
<img width="638" height="205" alt="image" src="https://github.com/user-attachments/assets/4005ee92-6b2a-499e-96fc-360e51bb5b66" />
<img width="596" height="175" alt="image" src="https://github.com/user-attachments/assets/2abdb693-25e7-496c-988b-0a6d9c4ce1ad" />
<img width="606" height="192" alt="image" src="https://github.com/user-attachments/assets/faf0ab08-0e32-4f38-bc52-cdefc1f95c01" />
<img width="600" height="172" alt="image" src="https://github.com/user-attachments/assets/3afee3bd-3478-49a7-9896-33682b477df8" />
<img width="612" height="185" alt="image" src="https://github.com/user-attachments/assets/d7da2101-faf1-4d0d-b0ea-5dc168ad58f7" />
<img width="607" height="172" alt="image" src="https://github.com/user-attachments/assets/efbd1ead-cde8-4676-9cae-618b3c4be125" />











