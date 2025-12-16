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

class BatchTrulyAIReelGenerator:
    def __init__(self, keys: dict):
        self.keys = keys
        if self.keys.get("GROQ_API_KEY"):
            self.groq_client = Groq(api_key=self.keys["GROQ_API_KEY"])
        
        self.horde_api_key = self.keys.get("HORDE_API_KEY", "0000000000")
        self.client_agent = "TrulyAI_Bot:v4.0:github.com/batch-mode"

    # --- 1. SCRIPT GENERATION (Unchanged) ---
    def _generate_ai_script(self, niche: str, count: int) -> dict:
        print(f"🤖 Generating Viral Script for: {niche}...")
        prompt = f"""
        You are a viral content expert. Create a Reel script for: {niche}.
        Strict JSON output only.
        Structure:
        {{
            "segments": [
                {{
                    "voiceover": "Spoken text (short)",
                    "visual_prompt": "Cinematic 8k photo of...",
                    "text_overlay": "Punchy Hook"
                }}
            ],
            "title": "Title",
            "mood": "energetic"
        }}
        Generate exactly {count} segments.
        """
        try:
            chat = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
            )
            return self._parse_json(chat.choices[0].message.content.strip())
        except Exception as e:
            raise Exception(f"Script Gen Failed: {e}")

    # =========================================================================
    # 2. THE SMART BATCH IMAGE GENERATOR (Quality First -> Speed Fallback)
    # =========================================================================
    def _generate_all_images(self, segments, temp_dir):
        """
        1. Submits ALL prompts to Horde.
        2. Waits X minutes.
        3. If any fail/timeout, fills gaps with Pollinations.
        """
        num_images = len(segments)
        results = {i: None for i in range(num_images)} # Stores paths: {0: 'path', 1: None...}
        horde_jobs = {} # {index: job_id}
        
        print(f"🚀 Phase 1: Submitting {num_images} jobs to AI Horde...")

        # --- STEP A: SUBMIT TO HORDE ---
        # We do this sequentially or parallel, submitting is fast.
        for i, seg in enumerate(segments):
            try:
                job_id = self._submit_to_horde(seg['visual_prompt'])
                horde_jobs[i] = job_id
                print(f"   🔹 [Seg {i}] Submitted to Horde (ID: {job_id})")
            except Exception as e:
                print(f"   ⚠️ [Seg {i}] Horde Submit Failed ({e}). Will wait for fallback.")
                horde_jobs[i] = None

        # --- STEP B: POLL HORDE (The "Wait" Phase) ---
        # We wait up to 180 seconds (3 mins) for high quality
        MAX_WAIT = 180 
        start_time = time.time()
        
        print(f"⏳ Phase 2: Waiting up to {MAX_WAIT}s for Horde workers...")
        
        while time.time() - start_time < MAX_WAIT:
            pending = [i for i in range(num_images) if results[i] is None and horde_jobs[i] is not None]
            
            if not pending:
                print("   ✨ All images finished via Horde!")
                break
                
            for i in pending:
                status, img_b64 = self._check_horde_status(horde_jobs[i])
                
                if status == 'DONE':
                    print(f"   ✅ [Seg {i}] Horde Delivered!")
                    # Save immediately
                    path = f"{temp_dir}/img_{i}.jpg"
                    with open(path, "wb") as f: f.write(base64.b64decode(img_b64))
                    results[i] = path
                elif status == 'FAILED':
                    print(f"   ❌ [Seg {i}] Horde Job Faulted. Queuing for fallback.")
                    horde_jobs[i] = None # Stop checking this one
            
            time.sleep(5) # Poll every 5 seconds

        # --- STEP C: FILL GAPS WITH POLLINATIONS (The "Cleanup" Phase) ---
        missing_indices = [i for i, path in results.items() if path is None]
        
        if missing_indices:
            print(f"💨 Phase 3: {len(missing_indices)} images missing. Rush ordering via Pollinations...")
            
            # Run Pollinations in parallel for speed
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_idx = {
                    executor.submit(self._generate_pollinations, segments[i]['visual_prompt']): i 
                    for i in missing_indices
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

    # --- HORDE HELPER METHODS ---
    def _submit_to_horde(self, prompt):
        url = "https://stablehorde.net/api/v2/generate/async"
        headers = {"apikey": self.horde_api_key, "Client-Agent": self.client_agent}
        payload = {
            "prompt": prompt + " ### vertical, 9:16 aspect ratio, cinematic, 8k",
            "params": {"steps": 25, "width": 576, "height": 1024, "toggles": [1, 4]}, # Toggles ensure we get the URL
            "models": ["AlbedoBase XL (SDXL)", "SDXL 1.0"],
            "nsfw": False, "censor_nsfw": True
        }
        resp = requests.post(url, json=payload, headers=headers)
        if resp.status_code != 202: raise Exception(f"Status {resp.status_code}")
        return resp.json()['id']

    def _check_horde_status(self, job_id):
        try:
            # Check Status
            stat_url = f"https://stablehorde.net/api/v2/generate/check/{job_id}"
            stat = requests.get(stat_url).json()
            
            if stat.get('faulted', False) or stat.get('is_possible', True) == False:
                return 'FAILED', None
                
            if stat['done'] == 1:
                # Retrieve
                final_url = f"https://stablehorde.net/api/v2/generate/status/{job_id}"
                final = requests.get(final_url).json()
                img_url = final['generations'][0]['img']
                return 'DONE', base64.b64encode(requests.get(img_url).content).decode()
            
            return 'WAITING', None
        except:
            return 'FAILED', None

    # --- POLLINATIONS HELPER METHOD ---
    def _generate_pollinations(self, prompt):
        encoded = quote(prompt + " vertical cinematic 8k")
        seed = random.randint(1, 999999)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1280&nologo=true&seed={seed}&model=flux"
        resp = requests.get(url, timeout=25)
        if resp.status_code == 200:
            return base64.b64encode(resp.content).decode()
        raise Exception("Pollinations API Error")

    # =========================================================================
    # 3. MAIN PIPELINE (Simplified to use Batch Image Gen)
    # =========================================================================
    def generate_reel(self, niche: str, num_images: int = 5):
        base_temp = tempfile.gettempdir()
        session_id = f"reel_{int(time.time())}"
        temp_dir = f"{base_temp}/{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. Script
        data = self._generate_ai_script(niche, num_images)
        print(f"📝 Script Ready: {len(data['segments'])} segments")
        
        # 2. BATCH IMAGES (The new logic)
        image_paths = self._generate_all_images(data['segments'], temp_dir)
        
        # 3. Process Clips (Audio + Stitch)
        clips = []
        for i, seg in enumerate(data['segments']):
            if not image_paths.get(i):
                print(f"❌ Skipping segment {i} (No image generated)")
                continue
                
            voice_path = f"{temp_dir}/voice_{i}.mp3"
            clip_path = f"{temp_dir}/clip_{i}.mp4"
            
            # Burn Text
            self._burn_text_into_image(image_paths[i], seg.get('text_overlay', ''))
            
            # Audio
            try:
                asyncio.run(edge_tts.Communicate(seg['voiceover'], "en-US-GuyNeural").save(voice_path))
                has_voice = True
            except: has_voice = False
            
            # Render
            dur = self._get_audio_duration(voice_path) + 0.2 if has_voice else 3.0
            self._render_clip_ffmpeg(image_paths[i], voice_path if has_voice else None, dur, clip_path)
            clips.append(clip_path)

        if not clips: raise Exception("No clips generated")

        # 4. Stitch
        final_path = f"{temp_dir}/reel.mp4"
        music_url = random.choice(MUSIC_LIBRARY.get(data.get('mood', 'upbeat'), MUSIC_LIBRARY['upbeat']))
        music_path = self._download_file(music_url, f"{temp_dir}/music.mp3")
        self._stitch_videos(clips, music_path, final_path, temp_dir)
        
        with open(final_path, 'rb') as f:
            return {'video_base64': base64.b64encode(f.read()).decode(), 'temp_dir': temp_dir}

    # --- HELPERS (Standard) ---
    def _burn_text_into_image(self, img_path, text):
        if not text: return
        try:
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            try: font = ImageFont.truetype("arial.ttf", 60)
            except: font = ImageFont.load_default()
            
            # Simple bottom-center text
            W, H = img.size
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            x, y = (W - w)/2, H - 400
            
            draw.text((x+3, y+3), text, font=font, fill="black")
            draw.text((x, y), text, font=font, fill="white")
            img.save(img_path)
        except: pass

    def _render_clip_ffmpeg(self, img, audio, dur, out):
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
        cmd = ['ffmpeg', '-y', '-i', vid, '-i', music, '-filter_complex', '[1:a]volume=0.1[bg];[0:a][bg]amix=inputs=2:duration=first', '-c:v', 'copy', out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    def _get_audio_duration(self, path):
        try: return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path]))
        except: return 3.0

    def _download_file(self, url, path):
        with open(path, 'wb') as f: f.write(requests.get(url).content)
        return path

    def _parse_json(self, text):
        if '```' in text: text = text.split('```json')[1].split('```')[0] if '```json' in text else text.split('```')[1]
        return json.loads(text)
