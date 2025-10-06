import os
import requests
import base64
import time
from urllib.parse import quote
from datetime import datetime
import google.generativeai as genai
import subprocess
import tempfile
import json

class ReelGenerator:
    def __init__(self, google_api_key: str):
        genai.configure(api_key=google_api_key)
        
    def generate_reel(self, niche: str, num_images: int = 20, duration: int = 15):
        """Generate Instagram Reel from multiple AI images with music"""
        print(f"🎬 Generating {num_images}-image reel for {niche} ({duration}s)...")
        
        # 1. Generate image prompts
        prompts = self._generate_image_prompts(niche, num_images)
        print("prompts.........", prompts)
        
        # 2. Download background music
        music_path = self._download_music(niche)
        
        # 3. Generate all images
        image_files = []
        temp_dir = tempfile.mkdtemp()
        
        for i, prompt in enumerate(prompts):
            print(f"🎨 Generating image {i+1}/{num_images}...")
            image_data = self._generate_image(prompt)
            
            img_path = f"{temp_dir}/img_{i:03d}.jpg"
            with open(img_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
            image_files.append(img_path)
            
            time.sleep(0.5)
        
        # 4. Create video with music
        video_path = f"{temp_dir}/reel.mp4"
        duration_per_image = duration / num_images
        self._create_video_with_music(image_files, music_path, video_path, duration_per_image, duration)
        
        # 5. Read video as base64
        with open(video_path, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode()
        
        # 6. Generate caption
        caption_data = self._generate_caption(niche)
        
        print("✅ Reel generated successfully!")
        return {
            'video_base64': video_base64,
            'caption': caption_data['caption'],
            'hashtags': caption_data['hashtags']
        }
    
    def _generate_image_prompts(self, niche: str, count: int) -> list:
        """Generate diverse prompts for the niche"""
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""Generate {count} diverse, visually stunning image prompts for {niche} Instagram content in 2025 current.
        Each prompt should be different but cohesive and synchronize for a slideshow reel.
        Make each of them ultra-detailed for AI image generation.
        
        Return as JSON array of strings:
        ["prompt 1 here...", "prompt 2 here...", ...]
        """
        
        response = model.generate_content(prompt)
        json_str = response.text.strip()
        
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '[' in json_str:
            json_str = json_str[json_str.find('['):json_str.rfind(']')+1]
        
        prompts = json.loads(json_str)
        return prompts[:count]
    
    def _generate_image(self, prompt: str) -> str:
        """Generate single image with retry logic"""
        enhanced_prompt = f"{prompt}, ultra detailed, professional photography, vibrant colors, 4K quality"
        encoded = quote(enhanced_prompt)
        
        # Try multiple times with different approaches
        attempts = [
            
            f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&model=flux",

f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&seed=105&nologo=true&model=nanobanana",
            f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true",
            f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920"
        ]
        
        for i, url in enumerate(attempts):
            try:
                response = requests.get(url, timeout=160)
                if response.status_code == 200:
                    return base64.b64encode(response.content).decode()
                print(f"⚠️ Attempt {i+1} failed: {response.status_code}")
            except Exception as e:
                print(f"⚠️ Attempt {i+1} error: {e}")
            
            if i < len(attempts) - 1:
                time.sleep(2)  # Wait before retry
        
        raise Exception(f"All image generation attempts failed")
    
    def _download_music(self, niche: str) -> str:
        """Download royalty-free music based on niche"""
        
        # Free Background Music (direct working links)
        music_urls = {
            'travel': 'https://www.bensound.com/bensound-music/bensound-sunny.mp3',
            'food': 'https://www.bensound.com/bensound-music/bensound-jazzyfrenchy.mp3',
            'fitness': 'https://www.bensound.com/bensound-music/bensound-energy.mp3',
            'motivation': 'https://www.bensound.com/bensound-music/bensound-epic.mp3',
            'lifestyle': 'https://www.bensound.com/bensound-music/bensound-slowmotion.mp3',
            'aesthetic': 'https://www.bensound.com/bensound-music/bensound-creativeminds.mp3',
            'fashion': 'https://www.bensound.com/bensound-music/bensound-cute.mp3',
            'tech': 'https://www.bensound.com/bensound-music/bensound-highoctane.mp3',
            'nature': 'https://www.bensound.com/bensound-music/bensound-relaxing.mp3',
        }
        
        music_url = music_urls.get(niche.lower(), music_urls['lifestyle'])
        
        print(f"🎵 Downloading {niche} music...")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(music_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                temp_dir = tempfile.gettempdir()
                music_path = f"{temp_dir}/background_music.mp3"
                with open(music_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Music downloaded successfully")
                return music_path
            else:
                print(f"⚠️ Music download failed (status {response.status_code}), generating silent audio...")
                return self._generate_silent_audio()
                
        except Exception as e:
            print(f"⚠️ Music download error: {e}, generating silent audio...")
            return self._generate_silent_audio()
    
    def _generate_silent_audio(self) -> str:
        """Generate silent audio as fallback if music download fails"""
        temp_dir = tempfile.gettempdir()
        silent_path = f"{temp_dir}/silent.mp3"
        
        # Generate 60 seconds of silence using FFmpeg
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', 'anullsrc=r=44100:cl=stereo',
            '-t', '60',
            '-q:a', '9',
            '-acodec', 'libmp3lame',
            silent_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print("✅ Silent audio generated as fallback")
        return silent_path
    
    def _create_video_with_music(self, image_files: list, music_path: str, output_path: str, 
                                  duration_per_image: float, total_duration: int):
        """Create video from images with Ken Burns effect and background music"""
        
        temp_dir = os.path.dirname(image_files[0])
        
        # Create concat file for FFmpeg
        concat_file = f"{temp_dir}/concat.txt"
        with open(concat_file, 'w') as f:
            for img in image_files:
                f.write(f"file '{img}'\n")
                f.write(f"duration {duration_per_image}\n")
            f.write(f"file '{image_files[-1]}'\n")
        
        # FFmpeg command with zoom/pan effects and music
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-i', music_path,
            '-vf', f"zoompan=z='min(zoom+0.0015,1.5)':d=25*{duration_per_image}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920,format=yuv420p",
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-r', '25',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-t', str(total_duration),
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _generate_caption(self, niche: str) -> dict:
        """Generate engaging caption and hashtags"""
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""Create viral Instagram Reel caption for {niche} content.
        Keep it short, punchy, and engaging.
        
        Return as JSON:
        {{
            "caption": "short caption here",
            "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5", "#reels", "#viral"]
        }}
        """
        
        response = model.generate_content(prompt)
        json_str = response.text.strip()
        
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '{' in json_str:
            json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]
        
        return json.loads(json_str)


class CloudinaryVideoUploader:
    @staticmethod
    def upload_video(video_base64: str, cloud_name: str, upload_preset: str, 
                     api_key: str, api_secret: str) -> str:
        """Upload video to Cloudinary"""
        import hashlib
        
        timestamp = int(time.time())
        params = {
            'api_key': api_key,
            'timestamp': timestamp,
            'upload_preset': upload_preset,
        }
        
        filtered = {k: v for k, v in params.items() if k not in {'file', 'cloud_name', 'resource_type', 'api_key'}}
        sorted_params = sorted(filtered.items())
        string_to_sign = '&'.join(f"{k}={v}" for k, v in sorted_params) + api_secret
        signature = hashlib.sha1(string_to_sign.encode()).hexdigest()
        
        url = f"https://api.cloudinary.com/v1_1/{cloud_name}/video/upload"
        
        files = {'file': f"data:video/mp4;base64,{video_base64}"}
        data = {
            'api_key': api_key,
            'signature': signature,
            'timestamp': timestamp,
            'upload_preset': upload_preset,
        }
        
        print("☁️ Uploading video to Cloudinary...")
        response = requests.post(url, files=files, data=data, timeout=120)
        response_data = response.json()
        
        if response.status_code != 200:
            raise Exception(f"Upload failed: {response_data.get('error', {}).get('message', 'Unknown error')}")
        
        return response_data['secure_url']


class InstagramReelPublisher:
    def __init__(self):
        self.api_version = 'v20.0'
        self.base_url = f'https://graph.facebook.com/{self.api_version}'
    
    def publish_reel(self, account_id: str, access_token: str, 
                     video_url: str, caption: str) -> str:
        """Publish Reel to Instagram"""
        
        create_url = f"{self.base_url}/{account_id}/media"
        create_params = {
            'media_type': 'REELS',
            'video_url': video_url,
            'caption': caption,
            'share_to_feed': True,
            'access_token': access_token
        }
        
        print("📱 Creating Reel container...")
        response = requests.post(create_url, data=create_params, timeout=30)
        data = response.json()
        
        if response.status_code != 200:
            raise Exception(data.get('error', {}).get('message', 'Container creation failed'))
        
        container_id = data['id']
        print(f"✅ Container created: {container_id}")
        
        print("⏳ Processing video...")
        max_retries = 30
        for i in range(max_retries):
            status_url = f"https://graph.facebook.com/{container_id}"
            status_params = {'fields': 'status_code', 'access_token': access_token}
            
            status_response = requests.get(status_url, params=status_params, timeout=10)
            status_data = status_response.json()
            status_code = status_data.get('status_code')
            
            print(f"Status: {status_code}")
            
            if status_code == 'FINISHED':
                break
            elif status_code in ['ERROR', 'EXPIRED']:
                raise Exception(f"Video processing failed: {status_code}")
            
            time.sleep(5)
        
        print("🚀 Publishing Reel...")
        publish_url = f"{self.base_url}/{account_id}/media_publish"
        publish_params = {'creation_id': container_id, 'access_token': access_token}
        
        publish_response = requests.post(publish_url, data=publish_params, timeout=30)
        publish_data = publish_response.json()
        
        if publish_response.status_code != 200:
            raise Exception(publish_data.get('error', {}).get('message', 'Publishing failed'))
        
        print(f"✅ Reel published! ID: {publish_data['id']}")
        return publish_data['id']


def main():
    print("🎬 Instagram Reel Generator with Music")
    print("=" * 50)
    
    google_api_key = os.getenv('GOOGLE_API_KEY')
    cloudinary_cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    cloudinary_upload_preset = os.getenv('CLOUDINARY_UPLOAD_PRESET')
    cloudinary_api_key = os.getenv('CLOUDINARY_API_KEY')
    cloudinary_api_secret = os.getenv('CLOUDINARY_API_SECRET')
    instagram_account_id = os.getenv('INSTAGRAM_ACCOUNT_ID')
    instagram_access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')
    
    required = {
        'GOOGLE_API_KEY': google_api_key,
        'CLOUDINARY_CLOUD_NAME': cloudinary_cloud_name,
        'CLOUDINARY_UPLOAD_PRESET': cloudinary_upload_preset,
        'CLOUDINARY_API_KEY': cloudinary_api_key,
        'CLOUDINARY_API_SECRET': cloudinary_api_secret,
        'INSTAGRAM_ACCOUNT_ID': instagram_account_id,
        'INSTAGRAM_ACCESS_TOKEN': instagram_access_token
    }
    
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        return
    
    try:
        # Configuration
        niche = os.getenv('REEL_NICHE', 'travel')
        num_images = int(os.getenv('REEL_IMAGES', '20'))
        duration = int(os.getenv('REEL_DURATION', '15'))  # Total video duration in seconds
        
        print(f"🎯 Niche: {niche}")
        print(f"📸 Images: {num_images}")
        print(f"⏱️ Duration: {duration} seconds")
        print("=" * 50)
        
        # Generate reel
        generator = ReelGenerator(google_api_key)
        reel_data = generator.generate_reel(niche, num_images, duration)
        
        # Upload to Cloudinary
        uploader = CloudinaryVideoUploader()
        video_url = uploader.upload_video(
            reel_data['video_base64'],
            cloudinary_cloud_name,
            cloudinary_upload_preset,
            cloudinary_api_key,
            cloudinary_api_secret
        )
        
        print(f"✅ Video URL: {video_url}")
        
        # Publish to Instagram
        publisher = InstagramReelPublisher()
        full_caption = f"{reel_data['caption']}\n\n{' '.join(reel_data['hashtags'])}"
        
        post_id = publisher.publish_reel(
            instagram_account_id,
            instagram_access_token,
            video_url,
            full_caption
        )
        
        print("=" * 50)
        print("🎉 REEL PUBLISHED SUCCESSFULLY!")
        print(f"📝 Caption: {reel_data['caption']}")
        print(f"🏷️ Hashtags: {' '.join(reel_data['hashtags'])}")
        print(f"🆔 Post ID: {post_id}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
