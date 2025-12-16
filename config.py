import os

# --- Configuration & Credentials ---
KEYS = {
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "CLOUDINARY_CLOUD_NAME": os.getenv("CLOUDINARY_CLOUD_NAME"),
    "CLOUDINARY_API_KEY": os.getenv("CLOUDINARY_API_KEY"),
    "CLOUDINARY_API_SECRET": os.getenv("CLOUDINARY_API_SECRET"),
    "CLOUDINARY_UPLOAD_PRESET": os.getenv("CLOUDINARY_UPLOAD_PRESET"),
    "INSTAGRAM_ACCOUNT_ID": os.getenv("INSTAGRAM_ACCOUNT_ID"),
    "INSTAGRAM_ACCESS_TOKEN": os.getenv("INSTAGRAM_ACCESS_TOKEN"),
    "REPLICATE_API_TOKEN": os.getenv("REPLICATE_API_TOKEN"),
    "HUGGINGFACE_API_TOKEN": os.getenv("HUGGINGFACE_API_TOKEN"),
    "CLIENT_ID_YOUTUBE": os.getenv("CLIENT_ID_YOUTUBE"),
    "CLIENT_SECRET_YOUTUBE": os.getenv("CLIENT_SECRET_YOUTUBE"),
    "REFRESH_TOKEN_YOUTUBE": os.getenv("REFRESH_TOKEN_YOUTUBE"),
}

# --- Assets ---
VOICE_ID = "en-US-ChristopherNeural"  # Options: en-US-AriaNeural, en-US-GuyNeural

MUSIC_LIBRARY = {
    'energetic': ['https://www.bensound.com/bensound-music/bensound-energy.mp3'],
    'calm': ['https://www.bensound.com/bensound-music/bensound-relaxing.mp3'],
    'upbeat': ['https://www.bensound.com/bensound-music/bensound-sunny.mp3'],
    'intense': ['https://www.bensound.com/bensound-music/bensound-epic.mp3'],
    'chill': ['https://www.bensound.com/bensound-music/bensound-jazzyfrenchy.mp3']
}
