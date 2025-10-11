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
import random

class EnhancedReelGenerator:
    def __init__(self, google_api_key: str):
        genai.configure(api_key=google_api_key)
        
        # Fallback defaults
        self.default_style = {
            'font': 'Arial-Bold',
            'font_size': 60,
            'color': '#FFFFFF',
            'position': 'center',
            'transitions': ['fade', 'slideleft', 'slideright'],
            'mood': 'energetic'
        }
        
        # Expanded music library organized by mood
        self.music_library = {
            'energetic': [
                'https://www.bensound.com/bensound-music/bensound-energy.mp3',
                'https://www.bensound.com/bensound-music/bensound-highoctane.mp3',
                'https://www.bensound.com/bensound-music/bensound-epic.mp3',
            ],
            'calm': [
                'https://www.bensound.com/bensound-music/bensound-relaxing.mp3',
                'https://www.bensound.com/bensound-music/bensound-slowmotion.mp3',
            ],
            'upbeat': [
                'https://www.bensound.com/bensound-music/bensound-sunny.mp3',
                'https://www.bensound.com/bensound-music/bensound-creativeminds.mp3',
            ],
            'intense': [
                'https://www.bensound.com/bensound-music/bensound-epic.mp3',
                'https://www.bensound.com/bensound-music/bensound-actionable.mp3',
            ],
            'chill': [
                'https://www.bensound.com/bensound-music/bensound-jazzyfrenchy.mp3',
                'https://www.bensound.com/bensound-music/bensound-cute.mp3',
            ]
        }
        
    def generate_reel(self, niche: str, num_images: int = 20, duration: int = 15):
        """Generate enhanced Instagram Reel with AI-driven styling"""
        print(f"🎬 Generating enhanced {num_images}-image reel for {niche} ({duration}s)...")
        
        # 1. Single AI call for everything
        content_data = self._generate_complete_content(niche, num_images)
        print("✅ AI Content generated")
        print(f"🎨 Style: {content_data['style']}")
        print(f"🎵 Mood: {content_data['style']['mood']}")
        
        # 2. Sort images by hook score (highest first)
        sorted_prompts = self._sort_by_hook_score(content_data['prompts'])
        print(f"🎯 Hook-first sequencing applied")
        
        # 3. Download smart music
        music_path = self._download_smart_music(content_data['style']['mood'])
        
        # 4. Generate all images
        image_files = []
        temp_dir = tempfile.mkdtemp()
        
        for i, prompt_data in enumerate(sorted_prompts):
            print(f"🎨 Generating image {i+1}/{num_images}...")
            image_data = self._generate_image(prompt_data['prompt'])
            
            img_path = f"{temp_dir}/img_{i:03d}.jpg"
            with open(img_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
            image_files.append(img_path)
            
            time.sleep(0.5)
        
        # 5. Create enhanced video with text overlays and dynamic transitions
        video_path = f"{temp_dir}/reel.mp4"
        duration_per_image = duration / num_images
        self._create_enhanced_video(
            image_files, 
            music_path, 
            video_path, 
            duration_per_image, 
            duration,
            content_data['text_overlays'],
            content_data['style']
        )
        
        # 6. Read video as base64
        with open(video_path, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode()
        
        print("✅ Enhanced Reel generated successfully!")
        return {
            'video_base64': video_base64,
            'caption': content_data['caption'],
            'hashtags': content_data['hashtags']
        }
    
    def _generate_complete_content(self, niche: str, count: int) -> dict:
        """Single AI call for ALL content decisions"""
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""You are a professional Instagram content creator. Generate COMPLETE content package for a {niche} Instagram Reel with {count} images.

IMPORTANT: Return ONLY valid JSON, no markdown, no explanations.

Generate:
1. {count} diverse, ultra-detailed image prompts (realistic, cohesive for slideshow)
2. Each prompt gets a hook_score (1-10) rating its visual impact/attention-grabbing power
3. {count} short, punchy text overlays (quotes/phrases) to burn onto each image
4. Viral caption + 20 hashtags
5. Complete style guide for this niche

Return this EXACT JSON structure:
{{
    "prompts": [
        {{"prompt": "detailed prompt here...", "hook_score": 8}},
        {{"prompt": "another prompt...", "hook_score": 6}}
    ],
    "text_overlays": ["Short text 1", "Text 2", "Quote 3"...],
    "caption": "viral caption here",
    "hashtags": ["#tag1", "#tag2", ..., "#reels", "#viral"],
    "style": {{
        "font": "Arial-Bold or DejaVuSans-Bold or Impact (FFmpeg compatible)",
        "font_size": 50-80,
        "color": "#HEXCOLOR (e.g., #FFD700)",
        "position": "center or top or bottom",
        "transitions": ["fade", "slideleft", "slideright", "circleopen", "dissolve"],
        "mood": "energetic or calm or upbeat or intense or chill"
    }}
}}
"""
        
        response = model.generate_content(prompt)
        json_str = response.text.strip()
        
        # Extract JSON
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0].strip()
        elif '{' in json_str:
            json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]
        
        try:
            data = json.loads(json_str)
            
            # Apply fallbacks for missing/invalid fields
            if 'style' not in data or not isinstance(data['style'], dict):
                data['style'] = self.default_style.copy()
            else:
                # Fill in missing style fields
                for key, value in self.default_style.items():
                    if key not in data['style']:
                        data['style'][key] = value
            
            # Ensure text_overlays match count
            if 'text_overlays' not in data or len(data['text_overlays']) < count:
                data['text_overlays'] = [f"Text {i+1}" for i in range(count)]
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed, using defaults: {e}")
            # Return safe fallback structure
            return {
                'prompts': [{'prompt': f'{niche} scene {i+1}', 'hook_score': 5} for i in range(count)],
                'text_overlays': [f'Moment {i+1}' for i in range(count)],
                'caption': f'Amazing {niche} content',
                'hashtags': ['#reels', '#viral', '#instagram'],
                'style': self.default_style.copy()
            }
    
    def _sort_by_hook_score(self, prompts: list) -> list:
        """Sort prompts by hook_score (highest first)"""
        return sorted(prompts, key=lambda x: x.get('hook_score', 5), reverse=True)
    
    def _download_smart_music(self, mood: str) -> str:
        """Download music based on AI-detected mood"""
        mood = mood.lower()
        music_urls = self.music_library.get(mood, self.music_library['energetic'])
        music_url = random.choice(music_urls)
        
        print(f"🎵 Downloading {mood} music...")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(music_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                temp_dir = tempfile.gettempdir()
                music_path = f"{temp_dir}/background_music.mp3"
                with open(music_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ {mood.capitalize()} music downloaded")
                return music_path
            else:
                return self._generate_silent_audio()
                
        except Exception as e:
            print(f"⚠️ Music error: {e}, using silent audio")
            return self._generate_silent_audio()
    
    def _generate_image(self, prompt: str) -> str:
        """Generate single image with retry logic"""
        enhanced_prompt = f"{prompt}, ultra detailed, professional photography, vibrant colors, 4K quality"
        encoded = quote(enhanced_prompt)
        
        attempts = [
            f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&model=flux",
            f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&model=turbo",
            f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true",
        ]
        
        for i, url in enumerate(attempts):
            try:
                response = requests.get(url, timeout=260)
                if response.status_code == 200:
                    return base64.b64encode(response.content).decode()
            except Exception as e:
                print(f"⚠️ Attempt {i+1} error: {e}")
            
            if i < len(attempts) - 1:
                time.sleep(20)
        
        raise Exception("All image generation attempts failed")
    
    def _generate_silent_audio(self) -> str:
        """Generate silent audio fallback"""
        temp_dir = tempfile.gettempdir()
        silent_path = f"{temp_dir}/silent.mp3"
        
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
        return silent_path
    
    def _create_enhanced_video(self, image_files: list, music_path: str, output_path: str,
                               duration_per_image: float, total_duration: int,
                               text_overlays: list, style: dict):
        """Create video with text overlays and dynamic transitions"""
        temp_dir = os.path.dirname(image_files[0])
        
        # Step 1: Add text overlays to images
        overlay_images = []
        for i, (img_path, text) in enumerate(zip(image_files, text_overlays)):
            overlay_path = f"{temp_dir}/overlay_{i:03d}.jpg"
            self._add_text_overlay(img_path, overlay_path, text, style)
            overlay_images.append(overlay_path)
        
        # Step 2: Create video with dynamic transitions
        self._create_video_with_transitions(
            overlay_images,
            music_path,
            output_path,
            duration_per_image,
            total_duration,
            style['transitions']
        )
    
    def _add_text_overlay(self, input_path: str, output_path: str, text: str, style: dict):
        """Burn text onto image using FFmpeg"""
        # Escape text for FFmpeg
        text_escaped = text.replace("'", "'\\\\\\''").replace(":", "\\:")
        
        # Position mapping
        position_map = {
            'top': 'x=(w-text_w)/2:y=100',
            'center': 'x=(w-text_w)/2:y=(h-text_h)/2',
            'bottom': 'x=(w-text_w)/2:y=h-text_h-100'
        }
        
        position = position_map.get(style.get('position', 'center'), position_map['center'])
        
        # Font size and color
        font_size = style.get('font_size', 60)
        color = style.get('color', '#FFFFFF').replace('#', '0x')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', f"drawtext=text='{text_escaped}':fontsize={font_size}:fontcolor={color}:{position}:box=1:boxcolor=black@0.5:boxborderw=10",
            '-q:v', '2',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _create_video_with_transitions(self, image_files: list, music_path: str,
                                       output_path: str, duration_per_image: float,
                                       total_duration: int, transitions: list):
        """Create video with dynamic transitions between images"""
        temp_dir = os.path.dirname(image_files[0])
        
        # For simplicity, use concat with zoompan (transitions are complex with xfade)
        # Full xfade implementation would require complex filter graphs
        concat_file = f"{temp_dir}/concat.txt"
        with open(concat_file, 'w') as f:
            for img in image_files:
                f.write(f"file '{img}'\n")
                f.write(f"duration {duration_per_image}\n")
            f.write(f"file '{image_files[-1]}'\n")
        
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
    print("🎬 ENHANCED Instagram Reel Generator (AI-Driven)")
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
        niche = os.getenv('REEL_NICHE', 'Womens Motivation, Mens Toughness')
        num_images = int(os.getenv('REEL_IMAGES', '5'))
        duration = int(os.getenv('REEL_DURATION', '10'))
        
        print(f"🎯 Niche: {niche}")
        print(f"📸 Images: {num_images}")
        print(f"⏱️ Duration: {duration} seconds")
        print("=" * 50)
        
        # Generate enhanced reel
        generator = EnhancedReelGenerator(google_api_key)
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
        print("🎉 ENHANCED REEL PUBLISHED SUCCESSFULLY!")
        print(f"📝 Caption: {reel_data['caption']}")
        print(f"🏷️ Hashtags: {' '.join(reel_data['hashtags'])}")
        print(f"🆔 Post ID: {post_id}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
