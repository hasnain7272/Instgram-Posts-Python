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
        
    def generate_reel(self, niche: str, num_images: int = 20):
        """Generate Instagram Reel from multiple AI images"""
        print(f"🎬 Generating {num_images}-image reel for {niche}...")
        
        # 1. Generate image prompts
        prompts = self._generate_image_prompts(niche, num_images)
        print("Images prompts are as follows.....", prompts)
        
        # 2. Generate all images
        image_files = []
        temp_dir = tempfile.mkdtemp()
        
        for i, prompt in enumerate(prompts):
            print(f"🎨 Generating image {i+1}/{num_images}...")
            image_data = self._generate_image(prompt)
            
            # Save temporarily
            img_path = f"{temp_dir}/img_{i:03d}.jpg"
            with open(img_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
            image_files.append(img_path)
            
            time.sleep(0.5)  # Avoid rate limits
        
        # 3. Create video with transitions
        video_path = f"{temp_dir}/reel.mp4"
        self._create_video(image_files, video_path, duration_per_image=0.8)
        
        # 4. Read video as base64
        with open(video_path, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode()
        
        # 5. Generate caption
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
        
        prompt = f"""Generate {count} diverse, visually stunning image prompts for {niche} Instagram content.
        Each prompt should be different but cohesive for a slideshow reel.
        Make each of them ultra-detailed and long for AI image generation.
        
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
        """Generate single image using Pollinations.ai"""
        enhanced_prompt = f"{prompt}, ultra detailed, professional photography, vibrant colors, 4K quality"
        encoded = quote(enhanced_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&model=flux"
        
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode()
        raise Exception(f"Image generation failed: {response.status_code}")
    
    def _create_video(self, image_files: list, output_path: str, duration_per_image: float = 0.8):
        """Create video from images with Ken Burns effect"""
        
        # Create concat file for FFmpeg
        concat_file = f"{os.path.dirname(image_files[0])}/concat.txt"
        with open(concat_file, 'w') as f:
            for img in image_files:
                f.write(f"file '{img}'\n")
                f.write(f"duration {duration_per_image}\n")
            # Last image needs to be repeated
            f.write(f"file '{image_files[-1]}'\n")
        
        # FFmpeg command with zoom/pan effects
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-vf', 'zoompan=z=\'min(zoom+0.0015,1.5)\':d=25*0.8:x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\':s=1080x1920,format=yuv420p',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-r', '25',
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
        
        # Generate signature
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
        
        # Create video container
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
        
        # Wait for video processing
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
        
        # Publish
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
    print("🎬 Instagram Reel Generator")
    print("=" * 50)
    
    # Get environment variables
    google_api_key = os.getenv('GOOGLE_API_KEY')
    cloudinary_cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    cloudinary_upload_preset = os.getenv('CLOUDINARY_UPLOAD_PRESET')
    cloudinary_api_key = os.getenv('CLOUDINARY_API_KEY')
    cloudinary_api_secret = os.getenv('CLOUDINARY_API_SECRET')
    instagram_account_id = os.getenv('INSTAGRAM_ACCOUNT_ID')
    instagram_access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')
    
    # Check required vars
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
        niche = os.getenv('REEL_NICHE', 'travel')  # Default to travel
        num_images = int(os.getenv('REEL_IMAGES', '20'))  # Default 20 images
        
        print(f"🎯 Niche: {niche}")
        print(f"📸 Images: {num_images}")
        print("=" * 50)
        
        # Generate reel
        generator = ReelGenerator(google_api_key)
        reel_data = generator.generate_reel(niche, num_images)
        
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
        
        Publish to Instagram
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
