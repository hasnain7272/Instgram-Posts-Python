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
import io
from urllib.parse import quote
from PIL import Image, ImageDraw, ImageFont

# Libraries
import edge_tts
from groq import Groq
from huggingface_hub import InferenceClient
from config import MUSIC_LIBRARY, VOICE_ID

class TrulyAIReelGenerator:
    def __init__(self, keys: dict):
        """
        Initialize the Triple-Layer Engine.
        keys: {
            "GROQ_API_KEY": "gsk_...",
            "HORDE_API_KEY": "...",     # Recommended for high quality
            "HUGGINGFACE_API_TOKEN": "..." # Needed for Level 3 fallback
        }
        """
        self.keys = keys
        
        # 1. Groq (Primary Scripting)
        if self.keys.get("GROQ_API_KEY"):
            self.groq_client = Groq(api_key=self.keys["GROQ_API_KEY"])
            
        # 2. Hugging Face (Backup Scripting & Last Resort Images)
        if self.keys.get("HUGGINGFACE_API_TOKEN"):
            self.hf_client = InferenceClient(token=self.keys["HUGGINGFACE_API_TOKEN"])
        else:
            print("⚠️ Warning: HF Token missing. Level 3 fallback disabled.")

        # 3. Horde (Primary Images)
        self.horde_api_key = self.keys.get("HORDE_API_KEY", "0000000000")
        self.client_agent = "TrulyAI_Bot:v9.0:triple-layer"

    # =========================================================================
    # 1. TRIPLE-LAYER SCRIPT GENERATION
    # =========================================================================
    def _generate_ai_script(self, niche: str, count: int) -> dict:
        print(f"🤖 Generating Script for: {niche}...")
        
        # 1. Define Prompt (Reusable)
        prompt = f"""
        You are a viral content expert. Write a Reel script for: {niche}.
        
        STRICT JSON OUTPUT ONLY. No markdown.
        Structure:
        {{
            "segments": [
                {{
                    "voiceover": "Spoken text (Conversational, under 15 words)",
                    "visual_prompt": "Cinematic 8k detailed prompt, distinct camera angle",
                    "text_overlay": "Punchy Hook (Max 4 words)"
                }}
            ],
            "title": "Clickbait Title",
            "caption": "Engaging caption",
            "hashtags": ["#tag1", "#tag2"](atleast 20+),
            "mood": "energetic"
        }}
        
        Generate exactly {count} segments. Segment 1 MUST be a Visual Hook.
        """

        # 2. Attempt 1: Groq (Llama 3)
        if self.keys.get("GROQ_API_KEY"):
            try:
                print("   🧠 Strategy A: Using Groq (Llama 3)...")
                chat = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                )
                return self._parse_json(chat.choices[0].message.content)
            except Exception as e:
                print(f"   ⚠️ Groq Failed: {e}. Switching to Strategy B...")

        # 3. Attempt 2: Hugging Face (Qwen 2.5)
        if self.keys.get("HUGGINGFACE_API_TOKEN"):
            try:
                print("   🧠 Strategy B: Using HF (Qwen 2.5)...")
                chat = self.hf_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model="Qwen/Qwen2.5-72B-Instruct", 
                    max_tokens=2000
                )
                return self._parse_json(chat.choices[0].message.content)
            except Exception as e:
                print(f"   ⚠️ HF Scripting Failed: {e}")

        raise Exception("❌ All Script Generators Failed.")

    # =========================================================================
    # 2. TRIPLE-LAYER IMAGE ENGINE (Horde -> Pollinations -> HF)
    # =========================================================================
    def _generate_all_images(self, segments, temp_dir):
        num_images = len(segments)
        results = {i: None for i in range(num_images)}
        
        # --- PHASE 1: AI HORDE (Best Quality) ---
        print(f"🚀 Phase 1: AI Horde (High Fidelity)...")
        horde_jobs = {}
        
        # Submit All
        for i, seg in enumerate(segments):
            print(seg['visual_prompt'])
            try:
                job_id = self._submit_to_horde(seg['visual_prompt'])
                horde_jobs[i] = job_id
            except Exception as e:
                print(f"   ⚠️ [Seg {i}] Horde Submit Error: {e}")
                horde_jobs[i] = None

        # Wait Loop (Max 120s - keep it tight)
        start_time = time.time()
        while time.time() - start_time < 120:
            pending = [i for i in range(num_images) if results[i] is None and horde_jobs[i] is not None]
            if not pending: break
            
            for i in pending:
                status, img_b64 = self._check_horde_status(horde_jobs[i])
                if status == 'DONE':
                    print(f"   ✅ [Seg {i}] Horde Delivered!")
                    self._save_b64(img_b64, i, temp_dir, results)
                elif status == 'FAILED':
                    horde_jobs[i] = None # Mark for fallback
            time.sleep(5)

        # --- PHASE 2: POLLINATIONS (Fast Backup) ---
        missing = [i for i, path in results.items() if path is None]
        if missing:
            print(f"💨 Phase 2: Pollinations ({len(missing)} missing)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_map = {executor.submit(self._gen_pollinations, segments[i]['visual_prompt']): i for i in missing}
                for future in concurrent.futures.as_completed(future_map):
                    idx = future_map[future]
                    try:
                        self._save_b64(future.result(), idx, temp_dir, results)
                        print(f"   🐇 [Seg {idx}] Saved by Pollinations")
                    except: pass

        # --- PHASE 3: HUGGING FACE (Last Resort) ---
        missing = [i for i, path in results.items() if path is None]
        if missing and self.keys.get("HUGGINGFACE_API_TOKEN"):
            print(f"🛡️ Phase 3: Hugging Face ({len(missing)} critical)...")
            for i in missing:
                try:
                    img_b64 = self._gen_hf_flux(segments[i]['visual_prompt'])
                    self._save_b64(img_b64, i, temp_dir, results)
                    print(f"   🏰 [Seg {i}] Saved by HF Flux")
                except Exception as e:
                    print(f"   ❌ [Seg {i}] Failed on all 3 layers: {e}")

        return results

    # --- Image Worker Methods ---
    def _submit_to_horde(self, prompt):
        url = "https://stablehorde.net/api/v2/generate/async"
        headers = {"apikey": self.horde_api_key, "Client-Agent": self.client_agent}
        
        # We ask for Top Tier models since we have an API Key
        full_prompt = prompt + " ### masterpiece, cinematic, 8k, photorealistic, sharp focus"
        negative = "cartoon, anime, painting, illustration, ugly, deformed, blurry, text, watermark"
        
        payload = {
            "prompt": full_prompt + " ### " + negative,
            "params": {
                "sampler_name": "k_dpmpp_2m", 
                "steps": 30,  
                "width": 576, "height": 1024,
                "cfg_scale": 6
            },
            "models": [
                "Juggernaut XL", 
                "RealVisXL V4.0", 
                "AlbedoBase XL (SDXL)"
            ],
            "nsfw": False, "censor_nsfw": True, "shared": True
        }
        
        resp = requests.post(url, json=payload, headers=headers)
        if resp.status_code != 202: raise Exception(f"Horde {resp.status_code}")
        return resp.json()['id']

    def _check_horde_status(self, job_id):
        try:
            r = requests.get(f"https://stablehorde.net/api/v2/generate/check/{job_id}").json()
            if r.get('faulted') or r.get('is_possible') == False: return 'FAILED', None
            if r['done'] == 1:
                final = requests.get(f"https://stablehorde.net/api/v2/generate/status/{job_id}").json()
                return 'DONE', base64.b64encode(requests.get(final['generations'][0]['img']).content).decode()
            return 'WAITING', None
        except: return 'FAILED', None

    def _gen_pollinations(self, prompt):
        encoded = quote(prompt + " vertical cinematic 8k")
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1280&nologo=true&seed={random.randint(1,999)}&model=flux"
        return base64.b64encode(requests.get(url, timeout=20).content).decode()

    def _gen_hf_flux(self, prompt):
        # Uses Flux Schnell via HF Inference Client
        try:
            image = self.hf_client.text_to_image(
                prompt + ", vertical 9:16 aspect ratio, high quality, 4k",
                model="black-forest-labs/FLUX.1-schnell"
            )
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            raise Exception(f"HF Flux Failed: {e}")

    def _save_b64(self, b64_str, idx, temp_dir, results):
        path = f"{temp_dir}/img_{idx}.jpg"
        with open(path, "wb") as f: f.write(base64.b64decode(b64_str))
        results[idx] = path

    # =========================================================================
    # 3. VIDEO ASSEMBLY (Standard Pipeline)
    # =========================================================================
    def generate_reel(self, niche: str, num_images: int = 5):
        base_temp = tempfile.gettempdir()
        session_id = f"reel_{int(time.time())}"
        temp_dir = f"{base_temp}/{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. Script
        data = self._generate_ai_script(niche, num_images)
        print(f"📝 Title: {data.get('title', 'Untitled')}")
        
        # 2. Images (The Triple Layer Engine)
        image_paths = self._generate_all_images(data['segments'], temp_dir)
        
        # 3. Clips
        clips = []
        for i, seg in enumerate(data['segments']):
            if not image_paths.get(i): continue 
            
            voice_path = f"{temp_dir}/voice_{i}.mp3"
            clip_path = f"{temp_dir}/clip_{i}.mp4"
            
            self._burn_text_into_image(image_paths[i], seg.get('text_overlay', ''))
            
            try:
                asyncio.run(edge_tts.Communicate(seg['voiceover'], "en-US-GuyNeural").save(voice_path))
                has_voice = True
            except: has_voice = False
            
            dur = self._get_audio_duration(voice_path) + 0.2 if has_voice else 4.0
            self._render_clip_ffmpeg(image_paths[i], voice_path if has_voice else None, dur, clip_path)
            clips.append(clip_path)
            
        if not clips: raise Exception("No clips generated.")

        # 4. Final Output
        final_path = f"{temp_dir}/reel.mp4"
        music_url = random.choice(MUSIC_LIBRARY.get(data.get('mood', 'upbeat'), list(MUSIC_LIBRARY.values())[0]))
        music_path = self._download_file(music_url, f"{temp_dir}/music.mp3")
        self._stitch_videos(clips, music_path, final_path, temp_dir)
        
        with open(final_path, 'rb') as f:
            return {
                'video_base64': base64.b64encode(f.read()).decode(),
                'caption': data.get('caption', ''),
                'hashtags': data.get('hashtags', []),
                'title': data.get('title', ''),
                'temp_dir': temp_dir
            }

    # --- Helpers (Unchanged) ---
    def _burn_text_into_image(self, img_path, text):
        if not text: return
        try:
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            W, H = img.size
            font_size = int(W * 0.08)
            try: font = ImageFont.truetype("arial.ttf", font_size)
            except: font = ImageFont.load_default()
            
            lines = []
            words = text.upper().split()
            current = []
            for w in words:
                current.append(w)
                if len(' '.join(current)) > 15: 
                    lines.append(' '.join(current[:-1]))
                    current = [w]
            lines.append(' '.join(current))
            
            y = H - (len(lines) * font_size * 1.3) - (H * 0.15)
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                w_line = bbox[2] - bbox[0]
                x = (W - w_line) / 2
                for off in [-2, 0, 2]:
                    draw.text((x+off, y-2), line, font=font, fill="black")
                    draw.text((x+off, y+2), line, font=font, fill="black")
                draw.text((x, y), line, font=font, fill="white")
                y += font_size * 1.2
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
        cmd = ['ffmpeg', '-y', '-i', vid, '-i', music, '-filter_complex', '[1:a]volume=0.15[bg];[0:a][bg]amix=inputs=2:duration=first', '-c:v', 'copy', out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    def _get_audio_duration(self, path):
        try: return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path]))
        except: return 4.0

    def _download_file(self, url, path):
        try:
            with open(path, 'wb') as f: f.write(requests.get(url, timeout=10).content)
        except:
            subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc', '-t', '10', path], stdout=subprocess.DEVNULL)
        return path

    def _parse_json(self, text):
        if '```' in text: text = text.split('```json')[1].split('```')[0] if '```json' in text else text.split('```')[1]
        try: return json.loads(text)
        except: return {"segments": [], "title": "Error"}
