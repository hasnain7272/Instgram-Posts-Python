import os

def get_clean_env(key, default=None):
    """Helper to read env vars and strip hidden newlines/spaces"""
    val = os.getenv(key, default)
    if val:
        return val.strip() # <--- THE FIX: Removes \n and spaces
    return val

# --- Configuration & Credentials ---
KEYS = {
    "GOOGLE_API_KEY": get_clean_env("GOOGLE_API_KEY"),
    "OPENAI_API_KEY": get_clean_env("OPENAI_API_KEY"),
    
    # Social Keys
    "CLOUDINARY_CLOUD_NAME": get_clean_env("CLOUDINARY_CLOUD_NAME"),
    "CLOUDINARY_API_KEY": get_clean_env("CLOUDINARY_API_KEY"),
    "CLOUDINARY_API_SECRET": get_clean_env("CLOUDINARY_API_SECRET"),
    "CLOUDINARY_UPLOAD_PRESET": get_clean_env("CLOUDINARY_UPLOAD_PRESET"),
    "INSTAGRAM_ACCOUNT_ID": get_clean_env("INSTAGRAM_ACCOUNT_ID"),
    "INSTAGRAM_ACCESS_TOKEN": get_clean_env("INSTAGRAM_ACCESS_TOKEN"),
    
    # AI Keys
    "REPLICATE_API_TOKEN": get_clean_env("REPLICATE_API_TOKEN"),
    "HUGGINGFACE_API_TOKEN": get_clean_env("HUGGINGFACE_API_TOKEN"),
    
    # YouTube Keys
    "CLIENT_ID_YOUTUBE": get_clean_env("CLIENT_ID_YOUTUBE"),
    "CLIENT_SECRET_YOUTUBE": get_clean_env("CLIENT_SECRET_YOUTUBE"),
    "REFRESH_TOKEN_YOUTUBE": get_clean_env("REFRESH_TOKEN_YOUTUBE"),
}

# --- Assets ---
VOICE_ID = "en-US-ChristopherNeural" 

MUSIC_LIBRARY = {
    'energetic': ['https://www.bensound.com/bensound-music/bensound-energy.mp3'],
    'calm': ['https://www.bensound.com/bensound-music/bensound-relaxing.mp3'],
    'upbeat': ['https://www.bensound.com/bensound-music/bensound-sunny.mp3'],
    'intense': ['https://www.bensound.com/bensound-music/bensound-epic.mp3'],
    'chill': ['https://www.bensound.com/bensound-music/bensound-jazzyfrenchy.mp3']
}
