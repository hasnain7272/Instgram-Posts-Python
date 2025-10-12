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

class TrulyAIReelGenerator:
    def __init__(self, google_api_key: str = None, openai_api_key: str = None):
        self.google_api_key = google_api_key
        self.openai_api_key = openai_api_key

        if google_api_key:
            genai.configure(api_key=google_api_key)

        # Music library organized by mood
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
        """Generate truly AI-driven Instagram Reel"""
        print(f"🎬 Generating AI-controlled {num_images}-image reel for {niche} ({duration}s)...")

        # Use persistent temp directory for checkpointing
        base_temp = tempfile.gettempdir()
        session_id = f"reel_{niche.replace(' ', '_')[:30]}_{num_images}_{duration}"
        temp_dir = f"{base_temp}/{session_id}"
        os.makedirs(temp_dir, exist_ok=True)

        checkpoint_file = f"{temp_dir}/checkpoint.json"

        # Check for existing checkpoint
        if os.path.exists(checkpoint_file):
            print("📦 Resuming from checkpoint...")
            with open(checkpoint_file, 'r') as f:
                content_data = json.load(f)
        else:
            content_data = self._generate_ai_complete_package(niche, num_images, duration)
            with open(checkpoint_file, 'w') as f:
                json.dump(content_data, f)
            print("✅ AI package generated & checkpointed")

        print(f"🎵 Mood: {content_data['mood']}")
        print(f"🎬 Transitions: AI-controlled per clip")

        # Download music
        music_path = self._download_music(content_data['mood'])

        # Generate images (DON'T sort yet - keep in sync with generation)
        image_files = []
        for i, clip in enumerate(content_data['clips']):
            img_path = f"{temp_dir}/img_{i:03d}.jpg"

            if os.path.exists(img_path):
                print(f"✓ Image {i+1}/{num_images} exists")
                image_files.append(img_path)
                continue

            print(f"🎨 Generating image {i+1}/{num_images}...")
            image_data = self._generate_image(clip['prompt'])

            with open(img_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
            image_files.append(img_path)
            time.sleep(0.5)

        # NOW sort clips AND images together by hook score
        print("📊 Sorting clips by hook score...")
        clips_with_images = list(zip(content_data['clips'], image_files))
        clips_with_images_sorted = sorted(clips_with_images, key=lambda x: x[0]['hook_score'], reverse=True)
        sorted_clips, sorted_images = zip(*clips_with_images_sorted)
        sorted_clips = list(sorted_clips)
        sorted_images = list(sorted_images)

        # Create AI-controlled video
        video_path = f"{temp_dir}/reel.mp4"
        duration_per_clip = duration / num_images
        self._create_ai_driven_video(
            sorted_images,
            sorted_clips,
            music_path,
            video_path,
            duration_per_clip,
            duration
        )

        # Read video
        with open(video_path, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode()

        print("✅ AI-driven Reel complete!")
        return {
            'video_base64': video_base64,
            'caption': content_data['caption'],
            'hashtags': content_data['hashtags']
        }

    def _generate_ai_complete_package(self, niche: str, count: int, duration: int) -> dict:
        """AI generates complete video package with FFmpeg-ready instructions"""

        prompt = f"""You are a professional video editor and Instagram content creator. Design a complete {duration}-second Reel with {count} clips for the niche: {niche}.

For EACH clip, provide detailed FFmpeg-compatible specifications:

1. Image prompt (detailed, realistic)
2. hook_score (1-10)
3. text_overlay with ONLY alphanumeric characters (2-4 words, NO special chars)
4. text_position: "top" | "center" | "bottom"
5. text_color: "white" | "yellow" | "red" | "cyan" (simple color names)
6. text_size: 40-60 (px)
7. text_style: "plain" | "box" | "shadow"
8. zoom_effect: "zoom_in" | "zoom_out" | "pan_left" | "pan_right" | "static"
9. transition: "fade" | "slide" | "dissolve" (for entering this clip)

Design each clip uniquely - vary positions, colors, zoom effects, and transitions for visual interest.

Return ONLY this JSON (no markdown):
{{
  "clips": [
    {{
      "prompt": "professional stock market trading floor...",
      "hook_score": 9,
      "text_overlay": "Bull Market",
      "text_position": "top",
      "text_color": "yellow",
      "text_size": 55,
      "text_style": "box",
      "zoom_effect": "zoom_in",
      "transition": "fade"
    }}
  ],
  "caption": "viral caption",
  "hashtags": ["#tag1", "#tag2", "#reels", "#viral"],
  "mood": "upbeat"
}}
"""

        json_str = None

        # Try Gemini first
        if self.google_api_key:
            try:
                print("🤖 AI designing video package...")
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                response = model.generate_content(prompt)
                json_str = response.text.strip()
                print("✅ AI design complete")
            except Exception as e:
                print(f"⚠️ Gemini failed: {str(e)[:100]}")

        # Fallback to OpenAI
        if (not json_str or len(json_str) < 50) and self.openai_api_key:
            try:
                print("🔄 Using OpenAI...")
                response = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {self.openai_api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'gpt-4o-mini',
                        'messages': [
                            {'role': 'system', 'content': 'You are a video editor. Return only valid JSON.'},
                            {'role': 'user', 'content': prompt}
                        ],
                        'temperature': 0.9,
                        'max_tokens': 3000
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    json_str = response.json()['choices'][0]['message']['content'].strip()
                    print("✅ OpenAI design complete")

            except Exception as e:
                print(f"⚠️ OpenAI failed: {str(e)[:100]}")

        if not json_str:
            raise Exception("AI generation failed")

        return self._parse_and_validate(json_str, niche, count)

    def _parse_and_validate(self, json_str: str, niche: str, count: int) -> dict:
        """Parse and ensure AI output is FFmpeg-compatible"""
        # Extract JSON
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0].strip()
        elif '{' in json_str:
            json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]

        try:
            data = json.loads(json_str)

            # Validate and sanitize each clip
            if 'clips' in data and len(data['clips']) >= count:
                for clip in data['clips'][:count]:
                    # Clean text (CRITICAL)
                    clip['text_overlay'] = ''.join(c for c in clip.get('text_overlay', 'Text') if c.isalnum() or c.isspace()).strip()[:25]

                    # Validate enums
                    if clip.get('text_position') not in ['top', 'center', 'bottom']:
                        clip['text_position'] = 'center'
                    if clip.get('text_color') not in ['white', 'yellow', 'red', 'cyan', 'green']:
                        clip['text_color'] = 'white'
                    if clip.get('text_style') not in ['plain', 'box', 'shadow']:
                        clip['text_style'] = 'box'
                    if clip.get('zoom_effect') not in ['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'static']:
                        clip['zoom_effect'] = 'zoom_in'
                    if clip.get('transition') not in ['fade', 'slide', 'dissolve']:
                        clip['transition'] = 'fade'

                    # Clamp text size
                    clip['text_size'] = max(35, min(int(clip.get('text_size', 50)), 60))
                    clip['hook_score'] = clip.get('hook_score', 5)

                data['clips'] = data['clips'][:count]
            else:
                raise ValueError("Invalid clips structure")

            data['mood'] = data.get('mood', 'upbeat')
            return data

        except Exception as e:
            print(f"⚠️ Parse error: {e}, using minimal fallback")
            # Absolute minimal fallback
            return {
                'clips': [{
                    'prompt': f'{niche} professional scene {i+1}',
                    'hook_score': 5,
                    'text_overlay': f'Scene {i+1}',
                    'text_position': 'center',
                    'text_color': 'white',
                    'text_size': 50,
                    'text_style': 'plain',
                    'zoom_effect': 'zoom_in' if i % 2 == 0 else 'zoom_out',
                    'transition': 'fade'
                } for i in range(count)],
                'caption': f'Amazing {niche} content',
                'hashtags': ['#reels', '#viral'],
                'mood': 'upbeat'
            }

    def _download_music(self, mood: str) -> str:
        """Download music"""
        music_urls = self.music_library.get(mood.lower(), self.music_library['upbeat'])
        music_url = random.choice(music_urls)

        try:
            response = requests.get(music_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
            if response.status_code == 200:
                music_path = f"{tempfile.gettempdir()}/music.mp3"
                with open(music_path, 'wb') as f:
                    f.write(response.content)
                return music_path
        except:
            pass

        return self._generate_silent_audio()

    def _generate_image(self, prompt: str) -> str:
        """Generate image"""
        enhanced = f"{prompt}, ultra detailed, professional, vibrant, 4K"
        encoded = quote(enhanced)

        attempts = [
            f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&model=flux",
            f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&model=turbo",
        ]

        for url in attempts:
            try:
                response = requests.get(url, timeout=260)
                if response.status_code == 200:
                    return base64.b64encode(response.content).decode()
            except:
                pass
            time.sleep(15)

        raise Exception("Image generation failed")

    def _generate_silent_audio(self) -> str:
        """Generate silent audio"""
        silent_path = f"{tempfile.gettempdir()}/silent.mp3"
        cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
               '-t', '60', '-q:a', '9', '-acodec', 'libmp3lame', silent_path]
        subprocess.run(cmd, check=True, capture_output=True)
        return silent_path

    def __init__(self, google_api_key: str = None, openai_api_key: str = None):
        self.google_api_key = google_api_key
        self.openai_api_key = openai_api_key
        self.has_drawtext = self._check_drawtext_support()  # Check FFmpeg capabilities

        if google_api_key:
            genai.configure(api_key=google_api_key)

        # Music library organized by mood
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

    def _check_drawtext_support(self) -> bool:
        """Check if FFmpeg has drawtext filter support"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-filters'],
                capture_output=True,
                text=True,
                timeout=5
            )
            has_support = 'drawtext' in result.stdout
            if has_support:
                print("✅ FFmpeg has drawtext support")
            else:
                print("⚠️ FFmpeg missing drawtext - text overlays will use PIL")
            return has_support
        except:
            print("⚠️ Could not check FFmpeg filters - assuming no drawtext")
            return False

    def _create_ai_driven_video(self, images: list, clips: list, music_path: str,
                                output_path: str, duration_per: float, total: int):
        """Create video with AI-designed effects per clip - FIXED FILTER CHAIN"""
        temp_dir = os.path.dirname(images[0])

        print(f"\n🎞️ Creating video from {len(images)} clips...")

        # Process each clip with AI specs
        video_clips = []
        for i, (img, clip) in enumerate(zip(images, clips)):
            clip_path = f"{temp_dir}/clip_{i:03d}.mp4"

            print(f"\n🎬 Clip {i+1}/{len(images)}: '{clip['text_overlay']}' | {clip['zoom_effect']} | {clip['transition']}")
            print(f"   Image: {os.path.basename(img)}")

            # If no drawtext support, add text to image first using PIL
            working_img = img
            if not self.has_drawtext and clip['text_overlay']:
                working_img = f"{temp_dir}/img_text_{i:03d}.jpg"
                self._add_text_with_pil(img, working_img, clip)

            # Build filter chain WITHOUT drawtext (since we handle it with PIL)
            filter_parts = []

            # 1. ZOOM/PAN EFFECT FIRST
            zoom_filter = self._build_zoom_filter(clip['zoom_effect'], duration_per)
            filter_parts.append(zoom_filter)

            # 2. TEXT OVERLAY (only if FFmpeg supports it)
            if self.has_drawtext:
                text_filter = self._build_text_filter(clip)
                if text_filter:
                    filter_parts.append(text_filter)

            # 3. TRANSITION EFFECTS
            transition_filter = self._build_transition_filter(clip['transition'], duration_per)
            if transition_filter:
                filter_parts.append(transition_filter)

            # 4. FORMAT (always last)
            filter_parts.append('format=yuv420p')

            # Combine all filters
            vf_chain = ','.join(filter_parts)

            # Build command
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-framerate', '30',  # Input framerate
                '-t', str(duration_per),
                '-i', working_img,
                '-vf', vf_chain,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '20',
                '-pix_fmt', 'yuv420p',
                '-r', '30',  # Output framerate
                '-vsync', 'cfr',  # Constant frame rate
                clip_path
            ]

            print(f"   Filter: {vf_chain[:100]}...")

            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                video_clips.append(clip_path)
                
                # Verify clip duration
                verify_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', clip_path]
                try:
                    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
                    clip_duration = float(verify_result.stdout.strip())
                    print(f"   ✅ Clip {i+1} created successfully ({clip_duration:.2f}s)")
                except:
                    print(f"   ✅ Clip {i+1} created successfully")
                    
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️ FFmpeg error for clip {i+1}:")
                print(f"   STDERR: {e.stderr[-500:]}")
                
                # Fallback: super simple filter without any text
                print(f"   🔄 Trying simple fallback...")
                simple_vf = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,format=yuv420p"
                
                cmd_simple = [
                    'ffmpeg', '-y',
                    '-loop', '1',
                    '-framerate', '30',
                    '-t', str(duration_per),
                    '-i', working_img,
                    '-vf', simple_vf,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-r', '30',
                    '-vsync', 'cfr',
                    clip_path
                ]
                
                subprocess.run(cmd_simple, check=True, capture_output=True, text=True)
                video_clips.append(clip_path)
                print(f"   ✅ Fallback succeeded")

        # Concatenate all clips with music
        print(f"\n🔗 Concatenating {len(video_clips)} clips...")
        concat_file = f"{temp_dir}/concat.txt"
        with open(concat_file, 'w') as f:
            for clip in video_clips:
                f.write(f"file '{clip}'\n")
        
        # Debug: show concat file content
        with open(concat_file, 'r') as f:
            print(f"📝 Concat list:\n{f.read()}")

        # First, concatenate video clips WITHOUT music (to ensure proper duration)
        video_only_path = f"{temp_dir}/video_only.mp4"
        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264',  # Re-encode instead of copy for consistency
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            video_only_path
        ]
        
        print("🎬 Concatenating video clips...")
        subprocess.run(concat_cmd, check=True, capture_output=True)
        
        # Verify concatenated video duration
        verify_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_only_path]
        try:
            result = subprocess.run(verify_cmd, capture_output=True, text=True)
            actual_duration = float(result.stdout.strip())
            print(f"📏 Concatenated video duration: {actual_duration:.2f}s (expected: {total}s)")
        except:
            pass

        # Now add music and trim to exact duration
        final_cmd = [
            'ffmpeg', '-y',
            '-i', video_only_path,
            '-i', music_path,
            '-t', str(total),  # Exact duration
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            output_path
        ]
        
        print("🎵 Adding music...")
        subprocess.run(final_cmd, check=True, capture_output=True)
        print(f"✅ Final video assembled: {output_path}")
        
        # Verify final video
        try:
            result = subprocess.run(verify_cmd[:-1] + [output_path], capture_output=True, text=True)
            duration_check = float(result.stdout.strip())
            print(f"✅ Final video duration: {duration_check:.2f}s (target: {total}s)")
        except:
            pass

    def _build_text_filter(self, clip: dict) -> str:
        """Build FFmpeg text filter - FIXED escaping"""
        text = clip['text_overlay']
        if not text or len(text) < 2:
            return None

        # Escape single quotes for FFmpeg
        text_escaped = text.replace("'", "\\'")

        # Position mapping
        pos_map = {
            'top': 'x=(w-text_w)/2:y=100',
            'center': 'x=(w-text_w)/2:y=(h-text_h)/2',
            'bottom': 'x=(w-text_w)/2:y=h-text_h-100'
        }
        pos = pos_map.get(clip['text_position'], 'x=(w-text_w)/2:y=(h-text_h)/2')

        # Color mapping
        color_map = {
            'white': 'white',
            'yellow': 'yellow',
            'red': 'red',
            'cyan': 'cyan',
            'green': 'green'
        }
        color = color_map.get(clip['text_color'], 'white')

        size = clip['text_size']

        # Build drawtext filter based on style (like working example)
        if clip['text_style'] == 'box':
            return f"drawtext=text='{text_escaped}':fontsize={size}:fontcolor={color}:{pos}:box=1:boxcolor=black@0.7:boxborderw=5"
        elif clip['text_style'] == 'shadow':
            return f"drawtext=text='{text_escaped}':fontsize={size}:fontcolor={color}:{pos}:shadowcolor=black:shadowx=3:shadowy=3"
        else:  # plain
            return f"drawtext=text='{text_escaped}':fontsize={size}:fontcolor={color}:{pos}"

    def _build_zoom_filter(self, effect: str, duration: float) -> str:
        """Build FFmpeg zoom/pan filter - FIXED for proper duration"""
        # Calculate frames (30 fps)
        frames = int(30 * duration)
        
        # IMPORTANT: Use output resolution (1080x1920 for portrait)
        output_res = "s=1080x1920"

        if effect == 'zoom_in':
            # Zoom from 1.0 to 1.5 smoothly
            # CRITICAL: Use frames-1 to ensure exact duration
            return f"zoompan=z='min(1.0+zoom*0.002,1.5)':d={frames}:{output_res}:fps=30"
        
        elif effect == 'zoom_out':
            # Zoom from 1.5 to 1.0
            return f"zoompan=z='max(1.5-on*0.002,1.0)':d={frames}:{output_res}:fps=30"
        
        elif effect == 'pan_left':
            # Pan from right to left
            return f"zoompan=z='1.2':x='iw/2-(iw/zoom/2)+on*2':y='ih/2-(ih/zoom/2)':d={frames}:{output_res}:fps=30"
        
        elif effect == 'pan_right':
            # Pan from left to right
            return f"zoompan=z='1.2':x='iw/2-(iw/zoom/2)-on*2':y='ih/2-(ih/zoom/2)':d={frames}:{output_res}:fps=30"
        
        else:  # static
            # Simple scale and crop (no zoom)
            return f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"

    def _build_transition_filter(self, transition: str, duration: float) -> str:
        """Build FFmpeg transition filter - simplified"""
        # Fade duration (0.3 seconds or 1/3 of clip duration, whichever is smaller)
        fade_dur = min(0.3, duration / 3)
        fade_out_start = duration - fade_dur

        if transition == 'fade':
            # Fade in at start, fade out at end
            return f"fade=t=in:st=0:d={fade_dur},fade=t=out:st={fade_out_start}:d={fade_dur}"
        
        elif transition == 'dissolve':
            # Similar to fade but softer
            return f"fade=t=in:st=0:d={fade_dur}:alpha=1"
        
        # For 'slide' or other transitions, return None (not directly supported)
        # Slide transitions require complex xfade between clips
        return None

    def _add_text_with_pil(self, input_img: str, output_img: str, clip: dict):
        """Add text overlay using PIL (Pillow) as fallback for FFmpeg drawtext"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Open image
            img = Image.open(input_img)
            draw = ImageDraw.Draw(img)
            
            text = clip['text_overlay']
            size = clip['text_size']
            color = clip['text_color']
            position = clip['text_position']
            style = clip['text_style']
            
            # Color mapping
            color_map = {
                'white': (255, 255, 255),
                'yellow': (255, 255, 0),
                'red': (255, 0, 0),
                'cyan': (0, 255, 255),
                'green': (0, 255, 0)
            }
            text_color = color_map.get(color, (255, 255, 255))
            
            # Try to load a bold font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", size)
                except:
                    font = ImageFont.load_default()
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position
            img_width, img_height = img.size
            if position == 'top':
                x = (img_width - text_width) // 2
                y = 100
            elif position == 'bottom':
                x = (img_width - text_width) // 2
                y = img_height - text_height - 100
            else:  # center
                x = (img_width - text_width) // 2
                y = (img_height - text_height) // 2
            
            # Draw text with style
            if style == 'box':
                # Draw background box
                padding = 10
                box_coords = [
                    x - padding,
                    y - padding,
                    x + text_width + padding,
                    y + text_height + padding
                ]
                draw.rectangle(box_coords, fill=(0, 0, 0, 180))
                draw.text((x, y), text, font=font, fill=text_color)
                
            elif style == 'shadow':
                # Draw shadow
                shadow_offset = 3
                draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0))
                draw.text((x, y), text, font=font, fill=text_color)
                
            else:  # plain
                draw.text((x, y), text, font=font, fill=text_color)
            
            # Save
            img.save(output_img, quality=95)
            print(f"   📝 Text added with PIL: '{text}'")
            
        except ImportError:
            print(f"   ⚠️ PIL not installed - text overlay skipped")
            # Just copy the original image
            import shutil
            shutil.copy(input_img, output_img)
        except Exception as e:
            print(f"   ⚠️ PIL text overlay failed: {e}")
            # Just copy the original image
            import shutil
            shutil.copy(input_img, output_img)


class CloudinaryVideoUploader:
    @staticmethod
    def upload_video(video_base64: str, cloud_name: str, upload_preset: str,
                     api_key: str, api_secret: str) -> str:
        import hashlib
        timestamp = int(time.time())
        params = {'api_key': api_key, 'timestamp': timestamp, 'upload_preset': upload_preset}
        filtered = {k: v for k, v in params.items() if k not in {'file', 'cloud_name', 'resource_type', 'api_key'}}
        sorted_params = sorted(filtered.items())
        string_to_sign = '&'.join(f"{k}={v}" for k, v in sorted_params) + api_secret
        signature = hashlib.sha1(string_to_sign.encode()).hexdigest()

        url = f"https://api.cloudinary.com/v1_1/{cloud_name}/video/upload"
        files = {'file': f"data:video/mp4;base64,{video_base64}"}
        data = {'api_key': api_key, 'signature': signature, 'timestamp': timestamp, 'upload_preset': upload_preset}

        print("☁️ Uploading...")
        response = requests.post(url, files=files, data=data, timeout=120)
        response_data = response.json()

        if response.status_code != 200:
            raise Exception(f"Upload failed: {response_data.get('error', {}).get('message')}")
        return response_data['secure_url']


class InstagramReelPublisher:
    def __init__(self):
        self.api_version = 'v20.0'
        self.base_url = f'https://graph.facebook.com/{self.api_version}'

    def publish_reel(self, account_id: str, access_token: str, video_url: str, caption: str) -> str:
        create_url = f"{self.base_url}/{account_id}/media"
        create_params = {
            'media_type': 'REELS', 'video_url': video_url,
            'caption': caption, 'share_to_feed': True, 'access_token': access_token
        }

        print("📱 Creating container...")
        response = requests.post(create_url, data=create_params, timeout=30)
        data = response.json()

        if response.status_code != 200:
            raise Exception(data.get('error', {}).get('message'))

        container_id = data['id']
        print(f"✅ Container: {container_id}")

        print("⏳ Processing...")
        for i in range(30):
            status_response = requests.get(
                f"https://graph.facebook.com/{container_id}",
                params={'fields': 'status_code', 'access_token': access_token},
                timeout=10
            )
            status_code = status_response.json().get('status_code')
            print(f"Status: {status_code}")

            if status_code == 'FINISHED':
                break
            elif status_code in ['ERROR', 'EXPIRED']:
                raise Exception(f"Processing failed: {status_code}")
            time.sleep(5)

        print("🚀 Publishing...")
        publish_response = requests.post(
            f"{self.base_url}/{account_id}/media_publish",
            data={'creation_id': container_id, 'access_token': access_token},
            timeout=30
        )
        publish_data = publish_response.json()

        if publish_response.status_code != 200:
            raise Exception(publish_data.get('error', {}).get('message'))

        print(f"✅ Published: {publish_data['id']}")
        return publish_data['id']


def main():
    print("🎬 TRULY AI-DRIVEN Instagram Reel Generator")
    print("=" * 50)

    google_api_key = os.getenv('GOOGLE_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    cloudinary_cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    cloudinary_upload_preset = os.getenv('CLOUDINARY_UPLOAD_PRESET')
    cloudinary_api_key = os.getenv('CLOUDINARY_API_KEY')
    cloudinary_api_secret = os.getenv('CLOUDINARY_API_SECRET')
    instagram_account_id = os.getenv('INSTAGRAM_ACCOUNT_ID')
    instagram_access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')

    if not (google_api_key or openai_api_key):
        print("❌ Need at least one AI API key")
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
        niche = os.getenv('REEL_NICHE', 'Motivation')
        num_images = int(os.getenv('REEL_IMAGES', '20'))
        duration = int(os.getenv('REEL_DURATION', '15'))

        print(f"🎯 Niche: {niche}")
        print(f"📸 Clips: {num_images}")
        print(f"⏱️ Duration: {duration}s")
        print("=" * 50)

        generator = TrulyAIReelGenerator(google_api_key, openai_api_key)
        reel_data = generator.generate_reel(niche, num_images, duration)

        uploader = CloudinaryVideoUploader()
        video_url = uploader.upload_video(
            reel_data['video_base64'],
            cloudinary_cloud_name,
            cloudinary_upload_preset,
            cloudinary_api_key,
            cloudinary_api_secret
        )

        print(f"✅ Video: {video_url}")

        publisher = InstagramReelPublisher()
        full_caption = f"{reel_data['caption']}\n\n{' '.join(reel_data['hashtags'])}"

        post_id = publisher.publish_reel(
            instagram_account_id,
            instagram_access_token,
            video_url,
            full_caption
        )

        print("=" * 50)
        print("🎉 SUCCESS!")
        print(f"📝 Caption: {reel_data['caption']}")
        print(f"🆔 Post: {post_id}")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()