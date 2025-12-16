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
import edge_tts
from groq import Groq  # High-speed free LLM
from config import MUSIC_LIBRARY, VOICE_ID

class TrulyAIReelGenerator:
    def __init__(self, keys: dict):
        """
        Initialize with Free/Open Source Powerhouses.
        keys: {
            "GROQ_API_KEY": "gsk_...",  # Get free at console.groq.com
            "HORDE_API_KEY": "..."      # Optional: '0000000000' works but is slower
        }
        """
        self.keys = keys
        
        # 1. Initialize Groq (The Script Brain)
        if self.keys.get("GROQ_API_KEY"):
            self.groq_client = Groq(api_key=self.keys["GROQ_API_KEY"])
        else:
            print("⚠️ Warning: GROQ_API_KEY missing. Script generation may fail.")

        # 2. Horde Configuration (The Image Engine)
        self.horde_api_key = self.keys.get("HORDE_API_KEY", "0000000000")
        self.client_agent = "TrulyAI_Bot:v2.0:github.com/zero-budget"

    # =========================================================================
    # 1. VIRAL SCRIPT GENERATION (Groq / Llama 3)
    # =========================================================================
    def _generate_ai_script(self, niche: str, count: int) -> dict:
        print(f"🤖 Generating Viral Script for: {niche}...")
        
        prompt = f"""
        You are an elite Instagram algorithm expert. Write a viral Reel script for: {niche}.
        
        CRITICAL FORMATTING RULES:
        1. Output ONLY valid JSON. Do not write "Here is the JSON".
        2. Strict JSON Structure:
        {{
            "segments": [
                {{
                    "voiceover": "Spoken text (keep it under 15 words)",
                    "visual_prompt": "Cinematic, photorealistic, 8k, detailed description of...",
                    "text_overlay": "Punchy 3-word Hook"
                }}
            ],
            "title": "Clickbait YouTube Title",
            "caption": "Engaging caption with questions",
            "hashtags": ["#tag1", "#tag2", "#tag30"],
            "mood": "energetic"
        }}
        
        CONTENT STRATEGY:
        - Generate exactly {count} segments.
        - Segment 1 MUST be a "Visual Hook" (stop the scroll).
        - Use simple, 4th-grade reading level English for voiceovers.
        """

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile", # Free, fast, smart
                temperature=0.7,
            )
            json_str = chat_completion.choices[0].message.content.strip()
            print(json_str)
            return self._parse_json(json_str)
            
        except Exception as e:
            raise Exception(f"❌ Script Generation Failed: {e}")

    # =========================================================================
    # 2. ROBUST IMAGE GENERATION (The Waterfall Strategy)
    # =========================================================================
    def _generate_image(self, prompt: str) -> str:
        """
        Tries AI Horde twice. If that fails, tries Pollinations twice.
        Returns: Base64 string of the image.
        """
        print(f"   🎨 Generating Image for: '{prompt[:30]}...'")
        
        # --- ATTEMPT 1 & 2: AI HORDE (High Quality, Distributed) ---
        for attempt in range(1, 3):
            try:
                print(f"      🔹 Horde Attempt {attempt}/2...")
                return self._generate_image_horde(prompt)
            except Exception as e:
                print(f"      ⚠️ Horde Attempt {attempt} failed: {e}")
                time.sleep(2) # Cooldown
        
        print("      ⚠️ Horde failed completely. Switching to fallback...")

        # --- ATTEMPT 3 & 4: POLLINATIONS (Fast, Unstable Backup) ---
        for attempt in range(1, 3):
            try:
                print(f"      🔸 Pollinations Attempt {attempt}/2...")
                return self._generate_image_pollinations(prompt)
            except Exception as e:
                print(f"      ⚠️ Pollinations Attempt {attempt} failed: {e}")
                time.sleep(2)

        raise Exception("❌ ALL Image Generators Failed. Check your network or API status.")

    def _generate_image_horde(self, prompt: str) -> str:
        """Async worker polling for AI Horde"""
        submit_url = "https://stablehorde.net/api/v2/generate/async"
        headers = {"apikey": self.horde_api_key, "Client-Agent": self.client_agent}
        
        payload = {
            "prompt": prompt + " ### vertical, 9:16 aspect ratio, cinematic, 8k, highly detailed, photorealistic",
            "params": {
                "sampler_name": "k_euler",
                "toggles": [1, 4], 
                "cfg_scale": 7,
                "steps": 25,
                "width": 576, # Safe width for free workers
                "height": 1024
            },
            "nsfw": False,
            "censor_nsfw": True,
            "models": ["AlbedoBase XL (SDXL)", "SDXL 1.0"] # Preferred models
        }

        # 1. Submit
        req = requests.post(submit_url, json=payload, headers=headers)
        if req.status_code != 202:
            raise Exception(f"Horde Submission Error: {req.text}")
        
        job_id = req.json()['id']
        
        # 2. Poll (Wait)
        start_t = time.time()
        while True:
            # Timeout after 90 seconds
            if time.time() - start_t > 90:
                raise Exception("Horde Timeout (90s)")
                
            status = requests.get(f"https://stablehorde.net/api/v2/generate/check/{job_id}").json()
            
            if status['done'] == 1:
                break
            if status['faulted']:
                raise Exception("Horde Worker Faulted")
                
            time.sleep(4) # Check every 4s

        # 3. Retrieve
        final = requests.get(f"https://stablehorde.net/api/v2/generate/status/{job_id}").json()
        if not final.get('generations'):
            raise Exception("Horde returned no images")
            
        img_url = final['generations'][0]['img']
        return base64.b64encode(requests.get(img_url).content).decode()

    def _generate_image_pollinations(self, prompt: str) -> str:
        """Simple GET request fallback"""
        encoded = quote(prompt + " vertical aspect ratio 9:16 cinematic 8k")
        # Added random seed to force variation on retry
        seed = random.randint(1, 99999) 
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1280&nologo=true&seed={seed}&model=flux"
        
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"Pollinations Error: {resp.status_code}")
            
        return base64.b64encode(resp.content).decode()

    # =========================================================================
    # 3. AUDIO & VIDEO PIPELINE (Unchanged but solid)
    # =========================================================================
    def _generate_voiceover(self, text: str, output_path: str):
        if not text: return False
        try:
            # Using a high quality male voice. Change to 'en-US-AnaNeural' for female.
            asyncio.run(edge_tts.Communicate(text, "en-US-GuyNeural").save(output_path))
            return True
        except Exception as e: 
            print(f"⚠️ TTS Failed: {e}")
            return False

    def generate_reel(self, niche: str, num_images: int = 5):
        base_temp = tempfile.gettempdir()
        session_id = f"reel_{int(time.time())}"
        temp_dir = f"{base_temp}/{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. Get Script
        data = self._generate_ai_script(niche, num_images)
        print(f"📝 Title: {data.get('title', 'Untitled')}")
        
        # 2. Build Clips
        clips = []
        for i, seg in enumerate(data['segments']):
            print(f"\n🎞️ Processing Segment {i+1}...")
            
            img_path = f"{temp_dir}/img_{i}.jpg"
            voice_path = f"{temp_dir}/voice_{i}.mp3"
            clip_path = f"{temp_dir}/clip_{i}.mp4"
            
            # Image + Burn Text
            try:
                img_b64 = self._generate_image(seg['visual_prompt'])
                with open(img_path, "wb") as f: f.write(base64.b64decode(img_b64))
                self._burn_text_into_image(img_path, seg.get('text_overlay', ''))
            except Exception as e:
                print(f"❌ Failed to create visual for segment {i}: {e}")
                continue # Skip bad segments

            # Voice
            has_voice = self._generate_voiceover(seg['voiceover'], voice_path)
            
            # Duration
            duration = 3.0
            if has_voice:
                duration = self._get_audio_duration(voice_path) + 0.3 # +0.3s padding
            
            # Render
            self._render_clip_ffmpeg(img_path, voice_path if has_voice else None, duration, clip_path)
            clips.append(clip_path)
            
        if not clips:
            raise Exception("No clips were generated successfully.")

        # 3. Stitch & Mix
        final_path = f"{temp_dir}/reel.mp4"
        
        # Music Logic
        mood = data.get('mood', 'upbeat')
        music_url = random.choice(MUSIC_LIBRARY.get(mood, MUSIC_LIBRARY['upbeat']))
        music_path = self._download_file(music_url, f"{temp_dir}/music.mp3")
        
        self._stitch_videos(clips, music_path, final_path, temp_dir)
        
        # Return Result
        with open(final_path, 'rb') as f:
            video_b64 = base64.b64encode(f.read()).decode()
            
        return {
            'video_base64': video_b64,
            'meta': data,
            'temp_dir': temp_dir
        }

    # =========================================================================
    # 4. HELPERS (Text Burn, FFmpeg, etc)
    # =========================================================================
    def _burn_text_into_image(self, img_path, text):
        if not text: return
        try:
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            W, H = img.size
            
            # Simple font loading
            try:
                font_size = int(H * 0.05) # Responsive font size
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            # Wrap text
            lines = []
            words = text.upper().split()
            current = []
            for w in words:
                current.append(w)
                if len(' '.join(current)) > 15:
                    lines.append(' '.join(current[:-1]))
                    current = [w]
            lines.append(' '.join(current))
            
            # Draw
            y = H - (len(lines) * font_size * 1.5) - 200 # Bottom offset
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                w_line = bbox[2] - bbox[0]
                x = (W - w_line) / 2
                
                # Shadow
                draw.text((x+4, y+4), line, font=font, fill="black")
                # Text
                draw.text((x, y), line, font=font, fill="white")
                y += font_size * 1.3
                
            img.save(img_path)
        except Exception as e:
            print(f"   ⚠️ Text Burn Warning: {e}")

    def _render_clip_ffmpeg(self, img, audio, dur, out):
        # Zoompan effect for 30fps vertical video
        vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0015,1.5)':d=700:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920:fps=30"
        
        cmd = ['ffmpeg', '-y', '-loop', '1', '-i', img]
        if audio: cmd.extend(['-i', audio])
        
        cmd.extend(['-vf', vf, '-c:v', 'libx264', '-t', str(dur), '-pix_fmt', 'yuv420p', '-preset', 'fast'])
        
        if audio: cmd.append('-shortest')
        cmd.append(out)
        
        # Suppress output unless error
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def _stitch_videos(self, clips, music, out, temp_dir):
        list_path = f"{temp_dir}/list.txt"
        with open(list_path, 'w') as f:
            for c in clips: f.write(f"file '{c}'\n")
            
        vid_only = f"{temp_dir}/vid.mp4"
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', vid_only], check=True, stdout=subprocess.DEVNULL)
        
        # Add Music (Ducking)
        cmd = [
            'ffmpeg', '-y', '-i', vid_only, '-i', music,
            '-filter_complex', '[1:a]volume=0.15[bg];[0:a][bg]amix=inputs=2:duration=first',
            '-c:v', 'copy', '-shortest', out
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

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
            # Create silent audio if download fails
            subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc', '-t', '10', path], stdout=subprocess.DEVNULL)
            return path

    def _parse_json(self, text):
        if '```' in text: 
            text = text.split('```json')[1].split('```')[0] if '```json' in text else text.split('```')[1]
        return json.loads(text)

# # Example Usage
# if __name__ == "__main__":
#     # Test Config
#     keys = {
#         "GROQ_API_KEY": "gsk_...", # Replace with actual key
#         # "HORDE_API_KEY": "..." # Optional
#     }
    
#     try:
#         gen = TrulyAIReelGenerator(keys)
#         # result = gen.generate_reel("Mysterious Ancient Egypt Facts", 3)
#         # print(f"✅ Success! Video at: {result['temp_dir']}/reel.mp4")
#     except Exception as e:
#         print(f"🔥 Critical Error: {e}")
