import os
import json
import time
import base64
import requests
import hashlib
import random
import io
from datetime import datetime, timedelta
from typing import List
from dataclasses import dataclass, asdict
from urllib.parse import quote

# Libraries
import google.generativeai as genai
from huggingface_hub import InferenceClient

# --- CONFIGURATION ---
def get_clean_env(key, default=None):
    val = os.getenv(key, default)
    if val:
        # Aggressively strip whitespace, newlines, and quotes
        return val.strip().replace('"', '').replace("'", "")
    return None

KEYS = {
    "GOOGLE_API_KEY": get_clean_env("GOOGLE_API_KEY"),
    "HUGGINGFACE_TOKEN": get_clean_env("HUGGINGFACE_TOKEN"),
    "CLOUDINARY_CLOUD_NAME": get_clean_env("CLOUDINARY_CLOUD_NAME"),
    "CLOUDINARY_API_KEY": get_clean_env("CLOUDINARY_API_KEY"),
    "CLOUDINARY_API_SECRET": get_clean_env("CLOUDINARY_API_SECRET"),
    "CLOUDINARY_UPLOAD_PRESET": get_clean_env("CLOUDINARY_UPLOAD_PRESET"),
    "INSTAGRAM_ACCOUNT_ID": get_clean_env("INSTAGRAM_ACCOUNT_ID"),
    "INSTAGRAM_ACCESS_TOKEN": get_clean_env("INSTAGRAM_ACCESS_TOKEN"),
}

# --- DATA CLASSES ---
@dataclass
class InspirationPost:
    id: str
    username: str
    caption: str
    imageDescription: str

@dataclass
class GeneratedPost:
    base64Image: str
    caption: str
    hashtags: List[str]

@dataclass
class PostMetadata:
    id: str
    timestamp: str
    inspiration_source: str
    image_description_hash: str
    caption_keywords: List[str]
    hashtags_used: List[str]
    engagement_niche: str

# --- TEXT GENERATION ENGINE ---
class TextEngine:
    def __init__(self):
        # 1. Setup Gemini
        if KEYS["GOOGLE_API_KEY"]:
            genai.configure(api_key=KEYS["GOOGLE_API_KEY"])
            # Fallback to gemini-pro which is often more stable in free tier
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # 2. Setup HuggingFace (Fallback)
        if KEYS["HUGGINGFACE_TOKEN"]:
            self.hf_client = InferenceClient(token=KEYS["HUGGINGFACE_TOKEN"])

    def generate(self, prompt: str) -> str:
        """Try Gemini -> Fail -> Try Hugging Face"""
        
        # Attempt 1: Gemini
        if KEYS["GOOGLE_API_KEY"]:
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"   ⚠️ Gemini Failed: {str(e)[:100]}. Switching to Fallback...")

        # Attempt 2: Hugging Face (Qwen)
        if KEYS["HUGGINGFACE_TOKEN"]:
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.hf_client.chat_completion(
                    messages, 
                    model="Qwen/Qwen2.5-72B-Instruct", 
                    max_tokens=4096
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"   ⚠️ HF Text Failed: {e}")

        raise Exception("❌ All Text Generators Failed")

# --- IMAGE GENERATOR ---
class ImageGenerator:
    def __init__(self):
        if KEYS["HUGGINGFACE_TOKEN"]:
            self.hf_client = InferenceClient(token=KEYS["HUGGINGFACE_TOKEN"])

    def generate_image(self, prompt: str) -> str:
        # 1. Hugging Face Flux
        try:
            print(f"   🎨 Generating with Flux (HF)...")
            image = self.hf_client.text_to_image(
                prompt + ", highly detailed, 4k, instagram aesthetic",
                model="black-forest-labs/FLUX.1-schnell"
            )
            # Safe conversion to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"   ⚠️ Flux Failed: {e}")

        # 2. Pollinations (Fallback)
        try:
            print(f"   🎨 Generating with Pollinations...")
            encoded = quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1080&nologo=true&model=flux"
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            print(f"   ⚠️ Pollinations Failed: {e}")
            
        raise Exception("All image generation methods failed")

# --- CORE LOGIC ---
class InstagramPostGenerator:
    def __init__(self):
        self.text_engine = TextEngine()
        self.image_generator = ImageGenerator()
        self.search_templates = [
            "viral Instagram posts {niche} aesthetic 2025",
            "trending {niche} photography instagram",
            "best {niche} content ideas instagram"
        ]

    def fetch_inspiration(self, niche: str) -> List[InspirationPost]:
        season = self._get_season()
        query = random.choice(self.search_templates).format(niche=niche, season=season)
        
        prompt = f"""
        Act as a social media researcher. Based on the query "{query}", invent 3 viral Instagram post concepts for the niche "{niche}".
        
        Return STRICT JSON array:
        [
            {{
                "id": "1",
                "username": "@viral_{niche}",
                "caption": "Example caption...",
                "imageDescription": "Detailed visual description of a photo (not video) related to {niche}..."
            }}
        ]
        """
        
        json_str = self.text_engine.generate(prompt)
        parsed = self._clean_json(json_str)
        
        return [InspirationPost(**p) for p in parsed]

    def generate_content(self, inspiration: InspirationPost, niche: str) -> GeneratedPost:
        # 1. Generate Image
        b64_img = self.image_generator.generate_image(inspiration.imageDescription)
        
        # 2. Generate Caption
        prompt = f"""
        Write an engaging Instagram caption for the {niche} niche.
        Context: {inspiration.imageDescription}
        
        Return JSON:
        {{
            "caption": "Catchy caption with emojis",
            "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]
        }}
        """
        
        json_str = self.text_engine.generate(prompt)
        data = self._clean_json(json_str)
        
        return GeneratedPost(
            base64Image=b64_img,
            caption=data.get('caption', f"Love this {niche} vibes! ✨"),
            hashtags=data.get('hashtags', [f"#{niche}"])
        )

    def _clean_json(self, text):
        if '```' in text:
            text = text.split('```json')[1].split('```')[0] if '```json' in text else text.split('```')[1]
        try:
            return json.loads(text)
        except:
            return []

    def _get_season(self):
        m = datetime.now().month
        if m in [12, 1, 2]: return "winter"
        if m in [3, 4, 5]: return "spring"
        if m in [6, 7, 8]: return "summer"
        return "autumn"
        
    def _extract_keywords(self, caption):
        return [w for w in caption.lower().split() if len(w) > 4][:10]

# --- HISTORY & UPLOAD MANAGERS ---
class PostHistoryManager:
    def __init__(self):
        self.cloud_name = KEYS['CLOUDINARY_CLOUD_NAME']
        self.api_key = KEYS['CLOUDINARY_API_KEY']
        self.api_secret = KEYS['CLOUDINARY_API_SECRET']
        self.preset = KEYS['CLOUDINARY_UPLOAD_PRESET']
        self.file_name = "post_history.json"

    def _sign(self, params):
        # Exclude file/resource_type from signature
        s = '&'.join(f"{k}={v}" for k, v in sorted(params.items()) if k not in ['file', 'resource_type'])
        return hashlib.sha1((s + self.api_secret).encode('utf-8')).hexdigest()

    def download(self) -> List[PostMetadata]:
        try:
            ts = int(time.time())
            sig = self._sign({'api_key': self.api_key, 'timestamp': ts})
            url = f"[https://res.cloudinary.com/](https://res.cloudinary.com/){self.cloud_name}/raw/upload/v1/{self.file_name}?api_key={self.api_key}&timestamp={ts}&signature={sig}"
            resp = requests.get(url)
            if resp.status_code == 200:
                return [PostMetadata(**p) for p in resp.json().get('posts', [])]
        except: pass
        return []

    def upload(self, history: List[PostMetadata]):
        data = json.dumps({'posts': [asdict(p) for p in history], 'updated': str(datetime.now())})
        ts = int(time.time())
        params = {
            'api_key': self.api_key, 'public_id': self.file_name,
            'timestamp': ts, 'upload_preset': self.preset
        }
        params['signature'] = self._sign(params)
        
        files = {'file': ('history.json', data, 'application/json')}
        requests.post(f"[https://api.cloudinary.com/v1_1/](https://api.cloudinary.com/v1_1/){self.cloud_name}/raw/upload", files=files, data=params)

    def get_next_niche(self, history):
        niches = ['fitness', 'motivation', 'food', 'travel', 'tech', 'wellness']
        recent = [p.engagement_niche for p in history[-5:]]
        for n in niches:
            if n not in recent: return n
        return random.choice(niches)

class CloudinaryUploader:
    @staticmethod
    def upload(b64_img):
        ts = int(time.time())
        
        # Ensure clean keys
        cloud_name = KEYS['CLOUDINARY_CLOUD_NAME']
        api_key = KEYS['CLOUDINARY_API_KEY']
        api_secret = KEYS['CLOUDINARY_API_SECRET']
        preset = KEYS['CLOUDINARY_UPLOAD_PRESET']
        
        params = {
            'api_key': api_key,
            'timestamp': ts,
            'upload_preset': preset
        }
        s = '&'.join(f"{k}={v}" for k, v in sorted(params.items())) + api_secret
        params['signature'] = hashlib.sha1(s.encode()).hexdigest()
        
        files = {'file': f"data:image/jpeg;base64,{b64_img}"}
        
        # Explicitly strip whitespace from URL construction
        url = f"[https://api.cloudinary.com/v1_1/](https://api.cloudinary.com/v1_1/){cloud_name.strip()}/image/upload"
        
        resp = requests.post(url, files=files, data=params)
        if resp.status_code != 200: raise Exception(f"Cloudinary: {resp.text}")
        return resp.json()['secure_url']

class InstagramPublisher:
    @staticmethod
    def publish(img_url, caption):
        acc_id = KEYS['INSTAGRAM_ACCOUNT_ID']
        token = KEYS['INSTAGRAM_ACCESS_TOKEN']
        base = f"[https://graph.facebook.com/v20.0/](https://graph.facebook.com/v20.0/){acc_id}"
        
        # 1. Container
        resp = requests.post(f"{base}/media", data={'image_url': img_url, 'caption': caption, 'access_token': token})
        if resp.status_code != 200: raise Exception(f"IG Container: {resp.text}")
        cont_id = resp.json()['id']
        
        # 2. Wait
        for _ in range(10):
            time.sleep(3)
            s = requests.get(f"[https://graph.facebook.com/v20.0/](https://graph.facebook.com/v20.0/){cont_id}", params={'fields': 'status_code', 'access_token': token}).json()
            if s.get('status_code') == 'FINISHED': break
            
        # 3. Publish
        resp = requests.post(f"{base}/media_publish", data={'creation_id': cont_id, 'access_token': token})
        if resp.status_code != 200: raise Exception(f"IG Publish: {resp.text}")
        return resp.json()['id']

# --- MAIN ---
def main():
    print("🚀 Static Image Bot Starting...")
    
    if not KEYS['GOOGLE_API_KEY'] and not KEYS['HUGGINGFACE_TOKEN']:
        print("❌ Missing API Keys")
        return

    try:
        history_mgr = PostHistoryManager()
        bot = InstagramPostGenerator()
        
        # 1. History & Niche
        history = history_mgr.download()
        niche = history_mgr.get_next_niche(history)
        print(f"🎯 Niche: {niche}")
        
        # 2. Generate
        print("🔍 Fetching Ideas...")
        ideas = bot.fetch_inspiration(niche)
        if not ideas: raise Exception("No ideas generated")
        
        print("🎨 Creating Content...")
        post = bot.generate_content(ideas[0], niche)
        print(f"📝 Caption: {post.caption[:50]}...")
        
        # 3. Upload & Publish
        if KEYS['INSTAGRAM_ACCESS_TOKEN']:
            print("☁️ Uploading Image...")
            img_url = CloudinaryUploader.upload(post.base64Image)
            
            print("📱 Publishing to Instagram...")
            pid = InstagramPublisher.publish(img_url, f"{post.caption}\n\n{' '.join(post.hashtags)}")
            print(f"✅ Published: {pid}")
            
            # 4. Save History
            meta = PostMetadata(
                id=pid, timestamp=str(datetime.now()),
                inspiration_source=niche,
                image_description_hash=hashlib.md5(ideas[0].imageDescription.encode()).hexdigest(),
                caption_keywords=bot._extract_keywords(post.caption),
                hashtags_used=post.hashtags,
                engagement_niche=niche
            )
            history.append(meta)
            history_mgr.upload(history)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
