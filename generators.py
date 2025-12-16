import os
import requests
import base64
import time
import random
import asyncio
import tempfile
import json
import subprocess
import io
from urllib.parse import quote
from PIL import Image, ImageDraw, ImageFont

# Libraries
from huggingface_hub import InferenceClient
import edge_tts
import google.generativeai as genai
from config import MUSIC_LIBRARY, VOICE_ID

class TrulyAIReelGenerator:
    def __init__(self, keys: dict):
        """Initialize with Pro Features (HF Client, EdgeTTS, PIL)"""
        self.keys = keys
        
        # Initialize HF Client (Robust Image Gen)
        if self.keys.get("HUGGINGFACE_API_TOKEN"):
            self.hf_client = InferenceClient(token=self.keys["HUGGINGFACE_API_TOKEN"])
        
        if self.keys.get("GOOGLE_API_KEY"):
            genai.configure(api_key=self.keys["GOOGLE_API_KEY"])

    # --- 1. AI Script Generation (Viral Optimized) ---
    def _generate_ai_script(self, niche: str, count: int) -> dict:
        """Generate a viral script with Hooks and 30 Hashtags"""
        print(f"🤖 Generating Viral Script for: {niche}...")
        
        prompt = f"""You are an Instagram algorithm expert. Write a viral Reel script for: {niche}.
        
        CRITICAL REQUIREMENTS:
        1. Start with a "Visual Hook" (text that makes people stop scrolling).
        2. Generate 30 high-traffic hashtags (mix of broad and niche).
        3. Create {count} segments.
        4. Mood must be one of: energetic, calm, upbeat, intense, chill.
        
        Return STRICT JSON:
        {{
            "segments": [
                {{
                    "voiceover": "Spoken text (keep it under 10 words per clip)",
                    "visual_prompt": "Detailed photorealistic image prompt, 8k, cinematic",
                    "text_overlay": "Short punchy text (max 5 words)"
                }}
            ],
            "title": "Clickbait YouTube Title (under 100 chars)",
            "caption": "Engaging caption with questions",
            "hashtags": ["#tag1", "#tag2", ... "#tag30"],
            "description": "YouTube description",
            "tags": "tag1, tag2, tag3",
            "mood": "energetic", 
            "category_id": "22"
        }}
        """
        
        json_str = None
        
        # Try Gemini First
        if self.keys.get("GOOGLE_API_KEY"):
            try:
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                response = model.generate_content(prompt)
                json_str = response.text.strip()
            except Exception as e:
                print(f"⚠️ Gemini Failed: {e}")
        
        # Fallback to HF Text
        if not json_str and self.keys.get("HUGGINGFACE_API_TOKEN"):
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.hf_client.chat_completion(
                    messages, model="Qwen/Qwen2.5-72B-Instruct", max_tokens=2000
                )
                json_str = response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ HF Text Failed: {e}")

        if not json_str:
            raise Exception("❌ All AI Script Generators Failed")

        return self._parse_json(json_str)

    # --- 2. Image Generation (Updated HF Client) ---
    def _generate_image(self, prompt: str) -> str:
        """Generate image using HF Client (Flux)"""
        print(f"   🎨 Generating Image...")
        
        # Strategy 1: Flux Schnell via InferenceClient
        try:
            image = self.hf_client.text_to_image(
                prompt + ", vertical 9:16 aspect ratio, high quality, 4k",
                model="black-forest-labs/FLUX.1-schnell"
            )
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            print(f"   ⚠️ Flux failed ({e}), trying fallback...")
            
        # Strategy 2: Pollinations Fallback
        try:
            encoded = quote(prompt + " vertical aspect ratio 9:16")
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&model=flux"
            return base64.b64encode(requests.get(url, timeout=30).content).decode()
        except Exception as e:
            raise Exception("All image generators failed")

    # --- 3. Voiceover (Edge TTS) ---
    def _generate_voiceover(self, text: str, output_path: str):
        if not text: return False
        try:
            asyncio.run(edge_tts.Communicate(text, VOICE_ID).save(output_path))
            return True
        except Exception as e: 
            print(f"⚠️ TTS Failed: {e}")
            return False

    # --- 4. The Core Pipeline ---
    def generate_reel(self, niche: str, num_images: int = 5):
        base_temp = tempfile.gettempdir()
        session_id = f"reel_{int(time.time())}"
        temp_dir = f"{base_temp}/{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. Get Script
        data = self._generate_ai_script(niche, num_images)
        print(f"📝 Title: {data['title']}")
        
        # 2. Build Clips
        clips = []
        for i, seg in enumerate(data['segments']):
            print(f"\n🎞️ Processing Segment {i+1}: '{seg['text_overlay']}'")
            
            img_path = f"{temp_dir}/img_{i}.jpg"
            voice_path = f"{temp_dir}/voice_{i}.mp3"
            clip_path = f"{temp_dir}/clip_{i}.mp4"
            
            # Generate Image
            img_b64 = self._generate_image(seg['visual_prompt'])
            with open(img_path, "wb") as f: f.write(base64.b64decode(img_b64))
            
            # BURN TEXT (PIL)
            self._burn_text_into_image(img_path, seg['text_overlay'])
            
            # Voice
            has_voice = self._generate_voiceover(seg['voiceover'], voice_path)
            
            # Calculate Duration
            duration = 3.0
            if has_voice:
                duration = self._get_audio_duration(voice_path) + 0.2
            
            # Render Clip
            self._render_clip_ffmpeg(img_path, voice_path if has_voice else None, duration, clip_path)
            clips.append(clip_path)
            
        # 3. Stitch & Mix
        final_path = f"{temp_dir}/reel.mp4"
        music_url = random.choice(MUSIC_LIBRARY.get(data.get('mood', 'upbeat'), MUSIC_LIBRARY['upbeat']))
        music_path = self._download_file(music_url, f"{temp_dir}/music.mp3")
        
        self._stitch_videos(clips, music_path, final_path, temp_dir)
        
        with open(final_path, 'rb') as f:
            video_b64 = base64.b64encode(f.read()).decode()
            
        return {
            'video_base64': video_b64,
            'caption': data['caption'],
            'hashtags': data['hashtags'],
            'title': data['title'],
            'description': data['description'],
            'tags': data['tags'],
            'category_id': data['category_id'],
            'temp_dir': temp_dir
        }

    # --- Helpers ---
    def _burn_text_into_image(self, img_path: str, text: str):
        """Uses PIL to burn text permanently onto the image"""
        if not text: return
        try:
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            W, H = img.size
            
            try:
                # Try standard fonts
                font = ImageFont.truetype("arial.ttf", 70)
            except:
                font = ImageFont.load_default()

            # Text Wrapping
            lines = []
            words = text.upper().split()
            current_line = []
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 15: 
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            lines.append(' '.join(current_line))
            
            # Draw (Bottom Center)
            y_text = H - 450 
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                x_pos = (W - w) / 2
                
                # Outline
                for off_x in [-3, 0, 3]:
                    for off_y in [-3, 0, 3]:
                        draw.text((x_pos+off_x, y_text+off_y), line, font=font, fill="black")
                
                # Fill
                draw.text((x_pos, y_text), line, font=font, fill="white")
                y_text += h + 15
                
            img.save(img_path)
            print(f"   ✍️ Text burned")
        except Exception as e:
            print(f"   ⚠️ Text Burn Failed: {e}")

    def _render_clip_ffmpeg(self, img, audio, dur, out):
        cmd = ['ffmpeg', '-y', '-loop', '1', '-i', img]
        if audio: cmd.extend(['-i', audio])
        
        # Zoom + Scale
        vf = "scale=1080:1920,zoompan=z='min(zoom+0.0015,1.5)':d=700:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920:fps=30"
        
        cmd.extend(['-vf', vf, '-c:v', 'libx264', '-t', str(dur), '-pix_fmt', 'yuv420p'])
        if audio: cmd.append('-shortest')
        cmd.append(out)
        
        subprocess.run(cmd, check=True, capture_output=True)

    def _stitch_videos(self, clips, music, out, temp_dir):
        list_path = f"{temp_dir}/list.txt"
        with open(list_path, 'w') as f:
            for c in clips: f.write(f"file '{c}'\n")
            
        vid_only = f"{temp_dir}/vid.mp4"
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', vid_only], check=True, capture_output=True)
        
        # Mix Audio (Duck music volume to 10%)
        cmd = [
            'ffmpeg', '-y', '-i', vid_only, '-i', music,
            '-filter_complex', '[1:a]volume=0.1[bg];[0:a][bg]amix=inputs=2:duration=first',
            '-c:v', 'copy', '-shortest', out
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _get_audio_duration(self, path):
        try:
            o = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path])
            return float(o.strip())
        except: return 3.0

    def _download_file(self, url, path):
        try:
            with open(path, 'wb') as f:
                f.write(requests.get(url, timeout=10).content)
            return path
        except:
            # Silent fallback
            subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc', '-t', '10', path], capture_output=True)
            return path

    def _parse_json(self, text):
        if '```' in text: text = text.split('```json')[1].split('```')[0] if '```json' in text else text.split('```')[1]
        try:
            return json.loads(text)
        except:
            print("⚠️ JSON Parse Failed. Returning empty.")
            return {"segments": [], "title": "Error", "hashtags": []}
