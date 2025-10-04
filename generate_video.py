import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# Load the model (using a specific model)
pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16, variant='fp16')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# Define the prompt for video generation
prompt = 'Spiderman is surfing. Darth Vader is also surfing and following Spiderman'

# Generate video frames
video_frames = pipe(prompt, num_inference_steps=25, num_frames=200).frames

# Convert frames to video
video_path = '/content/drive/MyDrive/generated_video.mp4'  # Path to save in Google Drive
video = export_to_video(video_frames)

# Save the video to the given path
video.save(video_path)

print(f'Video has been saved to: {video_path}')
