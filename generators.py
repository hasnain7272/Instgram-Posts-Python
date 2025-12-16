import os
import requests
import base64
import time
import random
import asyncio
import tempfile
import json
import subprocess
import concurrent.futures
from urllib.parse import quote
from PIL import Image, ImageDraw, ImageFont

# Libraries
import edge_tts
from groq import Groq
from config import MUSIC_LIBRARY, VOICE_ID

class TrulyAIReelGenerator:
    def __init__(self, keys: dict):
        """
        Initialize the Zero-Budget Production Engine.
        keys: {
            "GROQ_API_KEY": "gsk_...", 
            "HORDE_API_KEY": "..." (Optional, defaults to '0000000000')
        }
        """
        self.keys = keys
        
        # 1. Initialize Groq (The Script Brain)
        if self.keys.get("GROQ_API_KEY"):
            self.groq_client = Groq(api_key=self.keys["GROQ_API_KEY"])
        else:
            print("⚠️ Warning: GROQ_API_KEY missing. Script generation will fail.")

        # 2. Horde Configuration (The Image Engine)
        self.horde_api_key = self.keys.get("HORDE_API_KEY", "0000000000")
        self.client_agent = "TrulyAI_Bot:v7.0:production-quality"

    # =========================================================================
    # 1. HIGH-RETENTION SCRIPT GENERATION (Groq / Llama 3)
    # =========================================================================
    def _generate_ai_script(self, niche: str, count: int) -> dict:
        print(f"🤖 Generating Retention-Optimized Script for: {niche} ({count} segments)...")
        
        # Dynamic instructions based on video length
        retention_logic = ""
        if count > 12:
            retention_logic = f"""
            LONG-FORM STRUCTURE (CRITICAL):
            1. Segment 1: THE HOOK. Use a visual pattern interrupt (something weird/shocking).
            2. Segment {int(count/3)}: THE RE-HOOK. Ask a controversial question or say "But wait..."
            3. Segment {int(count*0.75)}: THE CLIMAX. Reveal the main secret/tip.
            4. VISUAL PACING: Alternating camera angles every 3 segments (e.g., Drone Shot -> Extreme Close Up -> POV).
            """
        else:
            retention_logic = "SHORT-FORM STRUCTURE: Start fast, deliver value immediately, end with a question."

        prompt = f"""
        You are an elite YouTube/Instagram Showrunner. Write a viral script for: {niche}.
        
        {retention_logic}
        
        STRICT JSON OUTPUT ONLY. No markdown.
        {{
            "segments": [
                {{
                    "voiceover": "Spoken text (Simple English, max 15 words, conversational)",
                    "visual_prompt": "Cinematic 8k prompt, SPECIFY CAMERA ANGLE (e.g. 'Low angle shot of...')",
                    "text_overlay": "Punchy Text (Max 4 words)"
                }}
            ],
            "title": "Clickbait Title (High CTR)",
            "caption": "Engaging caption with hooks",
            "hashtags": ["#tag1", "#tag2", ...],
            "mood": "energetic"
        }}
        
        Generate exactly {count} segments. Ensure visual prompts are distinct and not repetitive.
        """

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile", # Best free model for logic
                temperature=0.7,
            )
            json_str = chat_completion.choices[0].message.content.strip()
            return self._parse_json(json_str)
            
        except Exception as e:
            raise Exception(f"❌ Script Generation Failed: {e}")

    # =========================================================================
    # 2. HIGH-FIDELITY IMAGE ENGINE (Horde -> Dynamic Wait -> Pollinations)
    # =========================================================================
    def _submit_to_horde(self, prompt):
        url = "https://stablehorde.net/api/v2/generate/async"
        headers = {"apikey": self.horde_api_key, "Client-Agent": self.client_agent}
        
        # 1. THE MAGIC SAUCE: Better Keywords for Photorealism
        quality_boost = " ### masterpiece, cinematic lighting, 8k, hyperrealistic, highly detailed, sharp focus, 35mm photography"
        full_prompt = prompt + quality_boost

        # 2. THE NEGATIVE PROMPT (Removes "AI Look")
        negative_prompt = "cartoon, anime, painting, illustration, ugly, deformed, blurry, low quality, pixelated, distorted faces, bad anatomy, watermark, text, signature"

        # 3. TOP-TIER MODEL LIST (Prioritized)
        high_quality_models = [
            "Juggernaut XL",        # #1 for cinematic realism
            "RealVisXL V4.0",       # #2 for photo-realism
            "DreamShaper XL",       # Great artistic realism
            "AlbedoBase XL (SDXL)", # Solid all-rounder
            "Realistic Vision V6.0 B1" 
        ]

        payload = {
            "prompt": full_prompt + " ### " + negative_prompt,
            "params": {
                "sampler_name": "k_dpmpp_2m", # Sharper details
                "toggles": [1, 4],            # Downloadable
                "cfg_scale": 6,               # Lower scale = more realistic
                "steps": 30,                  # Cleaner image
                "width": 576,                 # 9:16 safe width
                "height": 1024
            },
            "nsfw": False,
            "censor_nsfw": True,
            "models": high_quality_models,
            "r2": True,
            "shared": True
        }
        
        resp = requests.post(url, json=payload, headers=headers)
        
        if resp.status_code != 202: 
            # Fallback if specific models are overloaded
            print(f"   ⚠️ Pro models busy ({resp.status_code}). Retrying with standard SDXL...")
            payload["models"] = ["SDXL 1.0"] 
            resp = requests.post(url, json=payload, headers=headers)
            
            if resp.status_code != 202:
                raise Exception(f"Horde Error: {resp.text}")
                
        return resp.json()['id']

    def _generate_all_images(self, segments, temp_dir):
        """
        Orchestrates the image generation strategy.
        """
        num_images = len(segments)
        results = {i: None for i in range(num_images)}
        horde_jobs = {} 
        
        # Dynamic Timeout: 60s per image (quality takes time), max 20 mins.
        MAX_WAIT = min(1200, max(200, num_images * 60))
        
        print(f"🚀 Phase 1: Submitting {num_images} High-Fidelity jobs... (Max Wait: {MAX_WAIT}s)")

        # A. Submit to Horde
        for i, seg in enumerate(segments):
            try:
                # Sanitize prompt to avoid confusing the photorealistic models
                clean_prompt = seg['visual_prompt'].replace("illustration", "photo").replace("vector", "photo")
                job_id = self._submit_to_horde(clean_prompt)
                horde_jobs[i] = job_id
                print(f"   🔹 [Seg {i}] Submitted (ID: {job_id})")
            except Exception as e:
                print(f"   ⚠️ [Seg {i}] Submit Error: {e}")
                horde_jobs[i] = None

        # B. Smart Wait Loop
        start_time = time.time()
        completed = 0
        
        while time.time() - start_time < MAX_WAIT:
            pending = [i for i in range(num_images) if results[i] is None and horde_jobs[i] is not None]
            
            if not pending:
                print("   ✨ All images finished via Horde!")
                break
                
            for i in pending:
                status, img_b64 = self._check_horde_status(horde_jobs[i])
                
                if status == 'DONE':
                    completed += 1
                    print(f"   ✅ [Seg {i}] Horde Delivered! ({completed}/{num_images})")
                    path = f"{temp_dir}/img_{i}.jpg"
                    with open(path, "wb") as f: f.write(base64.b64decode(img_b64))
                    results[i] = path
                elif status == 'FAILED':
                    print(f"   ❌ [Seg {i}] Horde Job Failed. Queuing for fallback.")
                    horde_jobs[i] = None 
            
            # Sleep longer to avoid rate limits
            time.sleep(8) 

        # C. Pollinations Fallback (The "Rescue" Phase)
        missing = [i for i, path in results.items() if path is None]
        if missing:
            print(f"💨 Phase 3: {len(missing)} images missing. Rush ordering via Pollinations...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_idx = {
                    executor.submit(self._generate_pollinations, segments[i]['visual_prompt']): i 
                    for i in missing
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        img_b64 = future.result()
                        path = f"{temp_dir}/img_{idx}.jpg"
                        with open(path, "wb") as f: f.write(base64.b64decode(img_b64))
                        results[idx] = path
                        print(f"   🐇 [Seg {idx}] Recovered via Pollinations.")
                    except Exception as e:
                        print(f"   💀 [Seg {idx}] Pollinations also failed: {e}")

        return results

    # --- Internal API Helpers ---
    def _check_horde_status(self, job_id):
        try:
            stat = requests.get(f"https://stablehorde.net/api/v2/generate/check/{job_id}").json()
            if stat.get('faulted') or stat.get('is_possible') == False: return 'FAILED', None
            if stat['done'] == 1:
                final = requests.get(f"https://stablehorde.net/api/v2/generate/status/{job_id}").json()
                img_url = final['generations'][0]['img']
                return 'DONE', base64.b64encode(requests.get(img_url).content).decode()
            return 'WAITING', None
        except: return 'FAILED', None

    def _generate_pollinations(self, prompt):
        # Fallback uses Flux model via Pollinations
        encoded = quote(prompt + " vertical cinematic 8k")
        seed = random.randint(1, 999999)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1280&nologo=true&seed={seed}&model=flux"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200: return base64.b64encode(resp.content).decode()
        raise Exception("Pollinations Error")

    # =========================================================================
    # 3. VIDEO ASSEMBLY PIPELINE (Stitch & Render)
    # =========================================================================
    def generate_reel(self, niche: str, num_images: int = 5):
        base_temp = tempfile.gettempdir()
        session_id = f"reel_{int(time.time())}"
        temp_dir = f"{base_temp}/{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. Get Script
        data = self._generate_ai_script(niche, num_images)
        print(f"📝 Title: {data.get('title', 'Untitled')}")
        
        # 2. Batch Generate Images
        image_paths = self._generate_all_images(data['segments'], temp_dir)
        
        # 3. Build Clips (Audio + Image + Burn Text)
        clips = []
        for i, seg in enumerate(data['segments']):
            if not image_paths.get(i): 
                print(f"❌ Skipping Seg {i} (Image Missing)")
                continue 
            
            voice_path = f"{temp_dir}/voice_{i}.mp3"
            clip_path = f"{temp_dir}/clip_{i}.mp4"
            
            # Burn Text
            self._burn_text_into_image(image_paths[i], seg.get('text_overlay', ''))
            
            # Voice Generation
            try:
                # Using a neutral, clear storytelling voice
                asyncio.run(edge_tts.Communicate(seg['voiceover'], "en-US-GuyNeural").save(voice_path))
                has_voice = True
            except: has_voice = False
            
            # Render Clip
            dur = self._get_audio_duration(voice_path) + 0.2 if has_voice else 4.0
            self._render_clip_ffmpeg(image_paths[i], voice_path if has_voice else None, dur, clip_path)
            clips.append(clip_path)
            
        if not clips: raise Exception("No clips generated.")

        # 4. Final Stitch
        final_path = f"{temp_dir}/reel.mp4"
        
        # Music Selection
        mood = data.get('mood', 'upbeat')
        # Use fallback if mood not found in library
        music_url = random.choice(MUSIC_LIBRARY.get(mood, list(MUSIC_LIBRARY.values())[0]))
        music_path = self._download_file(music_url, f"{temp_dir}/music.mp3")
        
        self._stitch_videos(clips, music_path, final_path, temp_dir)
        
        with open(final_path, 'rb') as f:
            video_b64 = base64.b64encode(f.read()).decode()
            
        return {
            'video_base64': video_b64,
            'caption': data.get('caption', ''),
            'hashtags': data.get('hashtags', []),
            'title': data.get('title', ''),
            'description': data.get('description', ''),
            'tags': data.get('tags', ''),
            'category_id': data.get('category_id', '22'),
            'temp_dir': temp_dir
        }

    # --- Video Helpers ---
    def _burn_text_into_image(self, img_path, text):
        if not text: return
        try:
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            # Dynamic Font Size
            W, H = img.size
            font_size = int(W * 0.08) # Text is 8% of screen width
            try: font = ImageFont.truetype("arial.ttf", font_size)
            except: font = ImageFont.load_default()
            
            # Text Wrapping
            lines = []
            words = text.upper().split()
            current = []
            for w in words:
                current.append(w)
                if len(' '.join(current)) > 15: # Break every 15 chars
                    lines.append(' '.join(current[:-1]))
                    current = [w]
            lines.append(' '.join(current))
            
            # Draw (Bottom Center)
            y = H - (len(lines) * font_size * 1.3) - (H * 0.15) # 15% from bottom
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                w_line = bbox[2] - bbox[0]
                x = (W - w_line) / 2
                
                # Thick Black Stroke
                for off in [-2, 0, 2]:
                    draw.text((x+off, y-2), line, font=font, fill="black")
                    draw.text((x+off, y+2), line, font=font, fill="black")
                    
                draw.text((x, y), line, font=font, fill="white")
                y += font_size * 1.2
            img.save(img_path)
        except: pass

    def _render_clip_ffmpeg(self, img, audio, dur, out):
        # Standard Vertical Video 1080x1920
        # Zoompan effect: Slow zoom in (1.0 -> 1.5)
        vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:-1:-1,zoompan=z='min(zoom+0.0015,1.5)':d=700:s=1080x1920:fps=30"
        
        cmd = ['ffmpeg', '-y', '-loop', '1', '-i', img]
        if audio: cmd.extend(['-i', audio])
        cmd.extend(['-vf', vf, '-c:v', 'libx264', '-t', str(dur), '-pix_fmt', 'yuv420p', '-preset', 'ultrafast'])
        if audio: cmd.append('-shortest')
        cmd.append(out)
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def _stitch_videos(self, clips, music, out, temp_dir):
        list_path = f"{temp_dir}/list.txt"
        with open(list_path, 'w') as f:
            for c in clips: f.write(f"file '{c}'\n")
            
        vid = f"{temp_dir}/vid.mp4"
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', vid], check=True, stdout=subprocess.DEVNULL)
        
        # Audio Mixing: Music volume 0.15, Voice volume 1.0
        cmd = [
            'ffmpeg', '-y', '-i', vid, '-i', music, 
            '-filter_complex', '[1:a]volume=0.15[bg];[0:a][bg]amix=inputs=2:duration=first', 
            '-c:v', 'copy', out
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    def _get_audio_duration(self, path):
        try: return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path]))
        except: return 4.0

    def _download_file(self, url, path):
        try:
            with open(path, 'wb') as f: f.write(requests.get(url, timeout=10).content)
        except:
            # Fallback: create silent audio
            subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc', '-t', '10', path], stdout=subprocess.DEVNULL)
        return path

    def _parse_json(self, text):
        if '```' in text: text = text.split('```json')[1].split('```')[0] if '```json' in text else text.split('```')[1]
        try: return json.loads(text)
        except: return {"segments": [], "title": "Error"}
