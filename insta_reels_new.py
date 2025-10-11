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
    def __init__(self, google_api_key: str = None, openai_api_key: str = None):
        self.google_api_key = google_api_key
        self.openai_api_key = openai_api_key

        if google_api_key:
            genai.configure(api_key=google_api_key)

        # Fallback defaults
        self.default_style = {
            'font': 'Sans-Serif',
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
            ],
            'chill': [
                'https://www.bensound.com/bensound-music/bensound-jazzyfrenchy.mp3',
                'https://www.bensound.com/bensound-music/bensound-cute.mp3',
            ]
        }

    def generate_reel(self, niche: str, num_images: int = 20, duration: int = 15):
        """Generate enhanced Instagram Reel with AI-driven styling"""
        print(f"🎬 Generating enhanced {num_images}-image reel for {niche} ({duration}s)...")

        # Use persistent temp directory for checkpointing
        base_temp = tempfile.gettempdir()
        session_id = f"reel_{niche.replace(' ', '_')[:30]}_{num_images}_{duration}"
        temp_dir = f"{base_temp}/{session_id}"
        os.makedirs(temp_dir, exist_ok=True)

        checkpoint_file = f"{temp_dir}/checkpoint.json"

        # Check for existing checkpoint
        if os.path.exists(checkpoint_file):
            print("📦 Found checkpoint, resuming from saved progress...")
            with open(checkpoint_file, 'r') as f:
                content_data = json.load(f)
        else:
            # 1. Single AI call for everything
            content_data = self._generate_complete_content(niche, num_images)
            # Save checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump(content_data, f)
            print("✅ AI Content generated & checkpointed")

        print(f"🎨 Style: AI-driven per image")
        print(f"🎵 Mood: {content_data['mood']}")

        # 2. Sort images by hook score (highest first)
        sorted_prompts = self._sort_by_hook_score(content_data['prompts'])
        print(f"🎯 Hook-first sequencing applied")

        # 3. Download smart music
        music_path = self._download_smart_music(content_data['mood'])

        # 4. Generate all images
        image_files = []

        for i, prompt_data in enumerate(sorted_prompts):
            img_path = f"{temp_dir}/img_{i:03d}.jpg"

            # Skip if image already exists
            if os.path.exists(img_path):
                print(f"✓ Image {i+1}/{num_images} already exists, skipping...")
                image_files.append(img_path)
                continue

            print(f"🎨 Generating image {i+1}/{num_images}...")
            image_data = self._generate_image(prompt_data['prompt'])

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
            content_data['text_designs']
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
        """Single AI call for ALL content decisions with Gemini/OpenAI fallback"""
        
        prompt = f"""You are a professional Instagram content creator. Generate COMPLETE content package for a {niche} Instagram Reel with {count} images.

IMPORTANT: Return ONLY valid JSON, no markdown, no explanations.

Generate:
1. {count} diverse, ultra-detailed image prompts (realistic, cohesive for slideshow/Reel)
2. Each prompt gets a hook_score (1-10) rating its visual impact/attention-grabbing power
3. For EACH image, generate text overlay with its own style settings:
   - text: 2-4 simple words (no special characters)
   - position: top/center/bottom (based on image content)
   - font_size: 40-60 (readable size)
   - box: true/false (whether to add background box)
   - color: #FFFFFF or #FFD700 or #FF0000 (contrasting with image)
4. Viral catchy caption + 20 hashtags
5. Overall mood for music selection

CRITICAL TEXT RULES:
- Simple words only (no punctuation)
- Position should complement the image subject
- Use box=true for busy backgrounds, box=false for clean backgrounds
- Vary the designs across images for visual interest

Return this EXACT JSON structure:
{{
    "prompts": [
        {{"prompt": "detailed prompt here...", "hook_score": 8}}
    ],
    "text_designs": [
        {{"text": "Push Harder", "position": "top", "font_size": 50, "box": true, "color": "#FFFFFF"}},
        {{"text": "Never Quit", "position": "bottom", "font_size": 55, "box": false, "color": "#FFD700"}}
    ],
    "caption": "viral caption here",
    "hashtags": ["#tag1", "#tag2", ..., "#reels", "#viral"],
    "mood": "energetic or calm or upbeat or intense or chill"
}}
"""

        json_str = None

        # Try Gemini first
        if self.google_api_key:
            try:
                print("🤖 Using Gemini API...")
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                response = model.generate_content(prompt)
                json_str = response.text.strip()
                print("✅ Gemini response received")
            except Exception as e:
                print(f"⚠️ Gemini failed: {str(e)[:100]}")

        # Fallback to OpenAI if Gemini failed
        if (not json_str or len(json_str) < 50) and self.openai_api_key:
            try:
                print("🔄 Switching to OpenAI fallback...")
                response = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {self.openai_api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'gpt-4o-mini',
                        'messages': [
                            {'role': 'system', 'content': 'You are a professional Instagram content creator. Return only valid JSON.'},
                            {'role': 'user', 'content': prompt}
                        ],
                        'temperature': 0.8,
                        'max_tokens': 2000
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    json_str = data['choices'][0]['message']['content'].strip()
                    print("✅ OpenAI response received")
                else:
                    print(f"⚠️ OpenAI failed: {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ OpenAI also failed: {str(e)[:100]}")

        # Parse response
        if not json_str or len(json_str) < 50:
            print("⚠️ No valid AI response, using fallback defaults")
            return self._get_fallback_content(niche, count)

        return self._parse_ai_response(json_str, niche, count)

    def _parse_ai_response(self, json_str: str, niche: str, count: int) -> dict:
        """Parse AI response and apply fallbacks"""
        # Extract JSON from markdown code blocks
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0].strip()
        elif '{' in json_str:
            json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]

        try:
            data = json.loads(json_str)

            # Validate mood (for music)
            if 'mood' not in data:
                data['mood'] = 'energetic'

            # Ensure text_designs exist and are valid
            if 'text_designs' not in data or len(data['text_designs']) < count:
                print("⚠️ AI didn't provide text_designs, using defaults")
                data['text_designs'] = [
                    {
                        'text': f'Moment {i+1}',
                        'position': 'center',
                        'font_size': 50,
                        'box': True,
                        'color': '#FFFFFF'
                    } for i in range(count)
                ]
            else:
                # Sanitize each text design
                for i, design in enumerate(data['text_designs'][:count]):
                    # Clean text
                    design['text'] = ''.join(c for c in design.get('text', f'Text {i+1}') if c.isalnum() or c.isspace())[:30]
                    # Validate position
                    design['position'] = design.get('position', 'center')
                    if design['position'] not in ['top', 'center', 'bottom']:
                        design['position'] = 'center'
                    # Clamp font size
                    design['font_size'] = max(35, min(int(design.get('font_size', 50)), 60))
                    # Ensure box is boolean
                    design['box'] = bool(design.get('box', True))
                    # Validate color
                    color = design.get('color', '#FFFFFF')
                    if not color.startswith('#'):
                        color = '#FFFFFF'
                    design['color'] = color

            # Ensure prompts exist
            if 'prompts' not in data or len(data['prompts']) < count:
                data['prompts'] = [{'prompt': f'Beautiful {niche} scene {i+1}, professional photography', 'hook_score': 5} for i in range(count)]

            return data

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            return self._get_fallback_content(niche, count)

    def _get_fallback_content(self, niche: str, count: int) -> dict:
        """Return safe fallback content structure"""
        return {
            'prompts': [{'prompt': f'Beautiful {niche} scene {i+1}, professional photography, high quality', 'hook_score': 5} for i in range(count)],
            'text_designs': [
                {
                    'text': f'Moment {i+1}',
                    'position': 'center',
                    'font_size': 50,
                    'box': True,
                    'color': '#FFFFFF'
                } for i in range(count)
            ],
            'caption': f'Amazing {niche} content! 🔥',
            'hashtags': ['#reels', '#viral', '#instagram', '#explore', '#trending'],
            'mood': 'energetic'
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
                               text_designs: list):
        """Create video with AI-driven text overlays per image"""
        temp_dir = os.path.dirname(image_files[0])

        # Step 1: Add AI-designed text overlays to each image
        overlay_images = []
        for i, (img_path, design) in enumerate(zip(image_files, text_designs)):
            overlay_path = f"{temp_dir}/overlay_{i:03d}.jpg"
            print(f"📝 Image {i+1}: '{design['text']}' at {design['position']} (box={design['box']})")
            self._add_ai_driven_text(img_path, overlay_path, design)
            overlay_images.append(overlay_path)

        # Step 2: Create video with dynamic transitions
        self._create_video_with_transitions(
            overlay_images,
            music_path,
            output_path,
            duration_per_image,
            total_duration
        )

    def _add_ai_driven_text(self, input_path: str, output_path: str, design: dict):
        """Add text overlay based on AI-generated design for this specific image"""
        # Extract design parameters (already sanitized in _parse_ai_response)
        text = design['text']
        position = design['position']
        font_size = design['font_size']
        use_box = design['box']
        color = design['color'].replace('#', '0x')

        # Position mapping
        position_map = {
            'top': 'x=(w-text_w)/2:y=120',
            'center': 'x=(w-text_w)/2:y=(h-text_h)/2',
            'bottom': 'x=(w-text_w)/2:y=h-text_h-120'
        }
        pos_str = position_map.get(position, position_map['center'])

        # Build FFmpeg command based on AI's design choice
        if use_box:
            # With box background
            vf = f"drawtext=text='{text}':fontsize={font_size}:fontcolor={color}:{pos_str}:box=1:boxcolor=black@0.5:boxborderw=10"
        else:
            # No box, just text
            vf = f"drawtext=text='{text}':fontsize={font_size}:fontcolor={color}:{pos_str}"

        cmd = ['ffmpeg', '-y', '-i', input_path, '-vf', vf, '-q:v', '2', output_path]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # Single fallback: copy image without text
            print(f"   ⚠️ Text overlay failed for this image, copying without text")
            subprocess.run(['ffmpeg', '-y', '-i', input_path, '-q:v', '2', output_path], check=True, capture_output=True)

    def _create_video_with_transitions(self, image_files: list, music_path: str,
                                       output_path: str, duration_per_image: float,
                                       total_duration: int):
        """Create video with dynamic Ken Burns zoom effect to look like a real Reel"""
        temp_dir = os.path.dirname(image_files[0])

        # Create individual video clips with varied zoom effects
        clips = []
        for i, img in enumerate(image_files):
            clip_path = f"{temp_dir}/clip_{i:03d}.mp4"

            # Alternate between zoom in and zoom out for variety
            if i % 2 == 0:
                # Zoom in effect
                zoom_filter = f"zoompan=z='min(zoom+0.002,1.3)':d={int(30*duration_per_image)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920"
            else:
                # Zoom out effect
                zoom_filter = f"zoompan=z='if(lte(zoom,1.0),1.3,max(1.0,zoom-0.002))':d={int(30*duration_per_image)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920"

            # Add slight panning for more dynamic feel
            if i % 3 == 0:
                # Pan left to right
                zoom_filter = f"zoompan=z='min(zoom+0.0015,1.2)':d={int(30*duration_per_image)}:x='if(gte(on,1),x+2,0)':y='ih/2-(ih/zoom/2)':s=1080x1920"

            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', img,
                '-vf', f"{zoom_filter},format=yuv420p,fade=t=in:st=0:d=0.3,fade=t=out:st={duration_per_image-0.3}:d=0.3",
                '-t', str(duration_per_image),
                '-r', '30',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '20',
                clip_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            clips.append(clip_path)

        # Concatenate all clips
        concat_file = f"{temp_dir}/concat_clips.txt"
        with open(concat_file, 'w') as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")

        # Merge with music
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-i', music_path,
            '-c:v', 'copy',
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
    openai_api_key = os.getenv('OPENAI_API_KEY')
    cloudinary_cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    cloudinary_upload_preset = os.getenv('CLOUDINARY_UPLOAD_PRESET')
    cloudinary_api_key = os.getenv('CLOUDINARY_API_KEY')
    cloudinary_api_secret = os.getenv('CLOUDINARY_API_SECRET')
    instagram_account_id = os.getenv('INSTAGRAM_ACCOUNT_ID')
    instagram_access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')

    # Check if at least one AI API key is available
    if not google_api_key and not openai_api_key:
        print("❌ Missing: At least one of GOOGLE_API_KEY or OPENAI_API_KEY is required")
        return

    required = {
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
        num_images = int(os.getenv('REEL_IMAGES', '20'))
        duration = int(os.getenv('REEL_DURATION', '15'))

        print(f"🎯 Niche: {niche}")
        print(f"📸 Images: {num_images}")
        print(f"⏱️ Duration: {duration} seconds")
        if google_api_key:
            print("🤖 Primary AI: Gemini")
        if openai_api_key:
            print("🔄 Fallback AI: OpenAI")
        print("=" * 50)

        # Generate enhanced reel
        generator = EnhancedReelGenerator(google_api_key, openai_api_key)
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
