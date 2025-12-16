import os
import time
import requests
import traceback

# Import our modules
from config import KEYS
from generators import TrulyAIReelGenerator

# Google Libraries
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# --- UPLOADERS ---

class CloudinaryUploader:
    @staticmethod
    def upload(video_base64, cloud_name, preset, api_key, api_secret):
        import hashlib
        ts = int(time.time())
        # Ensure preset is stripped just in case
        preset = preset.strip() if preset else ""
        
        s = f"timestamp={ts}&upload_preset={preset}{api_secret}"
        sig = hashlib.sha1(s.encode()).hexdigest()
        
        files = {'file': f"data:video/mp4;base64,{video_base64}"}
        data = {'api_key': api_key, 'signature': sig, 'timestamp': ts, 'upload_preset': preset}
        
        url = f"https://api.cloudinary.com/v1_1/{cloud_name}/video/upload"
        print(f"☁️ Uploading to {url.replace(cloud_name, '***')}...")
        
        resp = requests.post(url, files=files, data=data, timeout=120)
        if resp.status_code != 200: raise Exception(f"Cloudinary Failed: {resp.text}")
        return resp.json()

    @staticmethod
    def delete(cloud_name, api_key, api_secret, public_id):
        auth = (api_key, api_secret)
        requests.delete(f"https://api.cloudinary.com/v1_1/{cloud_name}/resources/video/upload", 
                       auth=auth, data={'public_ids[]': public_id, 'invalidate': True})

class InstagramPublisher:
    def publish(self, acc_id, token, video_url, caption):
        # 1. Container
        url = f"https://graph.facebook.com/v20.0/{acc_id}/media"
        print("📸 Creating IG Container...")
        resp = requests.post(url, data={'media_type': 'REELS', 'video_url': video_url, 'caption': caption, 'access_token': token})
        if resp.status_code != 200: raise Exception(f"IG Container: {resp.text}")
        cont_id = resp.json()['id']
        
        # 2. Wait
        print("⏳ Processing IG Video...")
        for _ in range(12): # Wait up to 60s
            time.sleep(5)
            s = requests.get(f"https://graph.facebook.com/v20.0/{cont_id}", params={'fields': 'status_code', 'access_token': token}).json()
            if s.get('status_code') == 'FINISHED': break
            if s.get('status_code') == 'ERROR': raise Exception("IG Processing Error")
            
        # 3. Publish
        print("🚀 Publishing to Feed...")
        resp = requests.post(f"https://graph.facebook.com/v20.0/{acc_id}/media_publish", data={'creation_id': cont_id, 'access_token': token})
        if resp.status_code != 200: raise Exception(f"IG Publish: {resp.text}")
        return resp.json()['id']

class YouTubePublisher:
    def __init__(self, client_id, client_secret, refresh_token):
        print("📺 Authenticating YouTube...")
        # Hardcoded clean URL to prevent formatting errors
        TOKEN_URI = "https://oauth2.googleapis.com/token"
        
        self.creds = Credentials(
            None, 
            refresh_token=refresh_token, 
            token_uri=TOKEN_URI,
            client_id=client_id, 
            client_secret=client_secret, 
            scopes=['https://www.googleapis.com/auth/youtube.upload']
        )
        # Force refresh to check validity
        self.creds.refresh(Request())
        self.youtube = build("youtube", "v3", credentials=self.creds)

    def upload(self, path, title, desc, tags, cat_id="22"):
        print(f"📤 Uploading: {title[:30]}...")
        body = {
            "snippet": {"title": title[:100], "description": desc[:5000], "tags": tags.split(','), "categoryId": cat_id},
            "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False}
        }
        media = MediaFileUpload(path, chunksize=256*1024, resumable=True, mimetype='video/mp4')
        req = self.youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        
        resp = None
        while resp is None:
            status, resp = req.next_chunk()
            if status: print(f"   Upload: {int(status.progress() * 100)}%")
        return resp.get('id')

# --- MAIN ---

def main():
    print("🚀 STARTING 0-BUDGET VIRAL ENGINE")
    
    # 1. Generate Content
    gen = TrulyAIReelGenerator(KEYS)
    niche = os.getenv('REEL_NICHE', 'Mind blowing facts')
    num_images = int(os.getenv('REEL_IMAGES', '5'))    
    
    try:
        result = gen.generate_reel(niche, num_images)
    except Exception as e:
        print(f"❌ Generation Failed: {e}")
        return

    # 2. Instagram Upload (via Cloudinary)
    video_url = None
    public_id = None
    
    if KEYS["INSTAGRAM_ACCESS_TOKEN"] and KEYS["CLOUDINARY_CLOUD_NAME"]:
        print("\n📸 Starting Instagram...")
        try:
            up = CloudinaryUploader.upload(result['video_base64'], KEYS['CLOUDINARY_CLOUD_NAME'], 
                                         KEYS['CLOUDINARY_UPLOAD_PRESET'], KEYS['CLOUDINARY_API_KEY'], KEYS['CLOUDINARY_API_SECRET'])
            video_url = up['secure_url']
            public_id = up['public_id']
            
            ig = InstagramPublisher()
            caption = f"{result['caption']}\n\n{' '.join(result['hashtags'])}"
            pid = ig.publish(KEYS['INSTAGRAM_ACCOUNT_ID'], KEYS['INSTAGRAM_ACCESS_TOKEN'], video_url, caption)
            print(f"✅ IG Published: {pid}")
            
            # Cleanup
            time.sleep(10)
            CloudinaryUploader.delete(KEYS['CLOUDINARY_CLOUD_NAME'], KEYS['CLOUDINARY_API_KEY'], KEYS['CLOUDINARY_API_SECRET'], public_id)
            
        except Exception as e:
            print(f"❌ Instagram Failed: {e}")
            import traceback
            traceback.print_exc()

    # 3. YouTube Upload
    if KEYS["REFRESH_TOKEN_YOUTUBE"]:
        print("\n📺 Starting YouTube...")
        try:
            yt = YouTubePublisher(KEYS['CLIENT_ID_YOUTUBE'], KEYS['CLIENT_SECRET_YOUTUBE'], KEYS['REFRESH_TOKEN_YOUTUBE'])
            vid = yt.upload(f"{result['temp_dir']}/reel.mp4", result['title'], result['description'], result['tags'], result['category_id'])
            print(f"✅ YouTube Published: {vid}")
        except Exception as e:
            print(f"❌ YouTube Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
