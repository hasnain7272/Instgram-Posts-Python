import os
import requests
import base64
import time
from urllib.parse import quote
import google.generativeai as genai
import subprocess
import tempfile
import json
import random

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp

class TrulyAIReelGenerator:
    def __init__(self, google_api_key: str = None, openai_api_key: str = None,
                 cloudinary_cloud_name: str = None, cloudinary_api_key: str = None, 
                 cloudinary_api_secret: str = None, cloudinary_upload_preset: str = None,
                 replicate_api_token: str = None, huggingface_api_token: str = None):
        """Initialize AI Reel Generator with comprehensive setup"""
        self.google_api_key = google_api_key
        self.openai_api_key = openai_api_key
        
        # Cloudinary credentials (optional for enhancement)
        self.cloudinary_cloud_name = cloudinary_cloud_name
        self.cloudinary_api_key = cloudinary_api_key
        self.cloudinary_api_secret = cloudinary_api_secret
        self.cloudinary_upload_preset = cloudinary_upload_preset
        
        # Image generation API tokens (optional - works without them too)
        self.replicate_api_token = replicate_api_token
        self.huggingface_api_token = huggingface_api_token
        
        # Check if enhancement is enabled
        self.enable_enhancement = os.getenv('ENABLE_CLOUDINARY_ENHANCE', 'false').lower() == 'true'
        if self.enable_enhancement:
            if all([cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret]):
                print("✨ Cloudinary AI enhancement: ENABLED")
            else:
                print("⚠️ Cloudinary AI enhancement disabled (missing credentials)")
                self.enable_enhancement = False
        
        if google_api_key:
            genai.configure(api_key=google_api_key)
        
        self.has_drawtext = self._check_drawtext_support()
        
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
            result = subprocess.run(['ffmpeg', '-filters'], capture_output=True, text=True, timeout=5)
            has_support = 'drawtext' in result.stdout
            print("✅ FFmpeg has drawtext support" if has_support else "⚠️ FFmpeg missing drawtext - text overlays will use PIL")
            return has_support
        except:
            print("⚠️ Could not check FFmpeg filters - assuming no drawtext")
            return False

    def _huggingface_text_generate(self, prompt: str) -> str:
        """Generate JSON package using Hugging Face Inference API (OpenAI compatible)"""
        if not self.huggingface_api_token:
            raise Exception("Hugging Face API token required")
        
        # This is the new, correct endpoint
        API_URL = "https://router.huggingface.co/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.huggingface_api_token}",
            "Content-Type": "application/json"
        }
        
        # The new API uses an OpenAI-compatible payload
        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.1", # Specify the model here
            "messages": [
                {"role": "system", "content": "You are a video editor. You must return ONLY the valid JSON object, with no other text before or after."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.7
        }
        
        # Retry logic for model loading (same as your image generator)
        for attempt in range(3):
            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        # Extract the text from the OpenAI-compatible response
                        return data['choices'][0]['message']['content']
                    else:
                        raise Exception(f"Unexpected HF response format: {str(data)[:200]}")

                elif response.status_code == 503: # Service Unavailable / Model loading
                    print(f"    ⏳ HF Text Model loading (503), waiting {5 * (attempt + 1)}s...")
                    time.sleep(5 * (attempt + 1))
                    continue
                else:
                    # Don't retry on other errors (like 400 Bad Request)
                    raise Exception(f"HuggingFace text API error: {response.status_code} {response.text[:100]}")
            
            except requests.exceptions.ReadTimeout:
                print(f"    ⏳ HF Text request timed out, retrying ({attempt + 1}/3)...")
                time.sleep(5 * (attempt + 1))
                continue
            except Exception as e:
                 # Catch other exceptions and re-raise
                 raise e 
        
        raise Exception("HuggingFace text generation failed after 3 retries")
        
    def generate_reel(self, niche: str, num_images: int = 20, duration: int = 15):
        """Generate truly AI-driven Instagram Reel"""
        print(f"🎬 Generating AI-controlled {num_images}-image reel for {niche} ({duration}s)...")

        base_temp = tempfile.gettempdir()
        session_id = f"reel_{niche.replace(' ', '_')[:30]}_{num_images}_{duration}"
        temp_dir = f"{base_temp}/{session_id}"
        os.makedirs(temp_dir, exist_ok=True)

        checkpoint_file = f"{temp_dir}/checkpoint.json"

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

        music_path = self._download_music(content_data['mood'])

        # Generate images in original order
        image_files = []
        for i, clip in enumerate(content_data['clips']):
            img_path = f"{temp_dir}/img_{i:03d}.jpg"

            if os.path.exists(img_path):
                print(f"✓ Image {i+1}/{num_images} exists")
                image_files.append(img_path)
                continue

            print(f"\n🎨 Generating image {i+1}/{num_images}...")
            image_data = self._generate_image(clip['image_prompt'])

            # Save original image (SAFETY: always have this)
            with open(img_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
            print(f"   💾 Image saved locally: {img_path}")
            
            # NEW: Try optional Cloudinary enhancement
            img_path = self._enhance_image_optional(img_path, temp_dir, i)
            
            image_files.append(img_path)
            time.sleep(0.5)

        # Sort clips AND images together by hook score
        print("\n📊 Sorting clips by hook score...")
        clips_with_images = list(zip(content_data['clips'], image_files))
        clips_with_images_sorted = sorted(clips_with_images, key=lambda x: x[0]['hook_score'], reverse=True)
        sorted_clips, sorted_images = zip(*clips_with_images_sorted)

        video_path = f"{temp_dir}/reel.mp4"
        duration_per_clip = duration / num_images
        self._create_smart_video(list(sorted_images), list(sorted_clips), music_path, video_path, duration_per_clip, duration, temp_dir)

        with open(video_path, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode()

        print("\n✅ AI-driven Reel complete!")
        return {
            'video_base64': video_base64,
            'caption': content_data['caption'],
            'hashtags': content_data['hashtags'],
            'title' : content_data['title'],
            'description': content_data['description'],
            'tags': content_data['tags'],
            'category_id' : content_data['category_id'],
            'temp_dir': temp_dir
        }

    def _generate_image(self, prompt: str) -> str:
        """
        Multi-source image generation with cascading fallback
        Priority: Replicate → Hugging Face → Pollinations
        """
        
        # Source 1: Try Replicate (Best quality)
        print("   🔄 Trying Replicate...")
        try:
            image_data = self._replicate_generate(prompt)
            if image_data:
                print("   ✅ Image generated via Replicate")
                return image_data
        except Exception as e:
            print(f"   ⚠️ Replicate failed: {str(e)[:80]}")
        
        # Source 2: Try Hugging Face (Good quality)
        print("   🔄 Trying Hugging Face...")
        try:
            image_data = self._huggingface_generate(prompt)
            if image_data:
                print("   ✅ Image generated via Hugging Face")
                return image_data
        except Exception as e:
            print(f"   ⚠️ Hugging Face failed: {str(e)[:80]}")
        
        # Source 3: Try Pollinations (Reliable fallback)
        print("   🔄 Trying Pollinations...")
        try:
            image_data = self._pollinations_generate(prompt)
            if image_data:
                print("   ✅ Image generated via Pollinations")
                return image_data
        except Exception as e:
            print(f"   ⚠️ Pollinations failed: {str(e)[:80]}")
        
        # All sources failed
        raise Exception("❌ All image generation sources failed")

    def _replicate_generate(self, prompt: str) -> str:
        """Generate image using Replicate (Flux model) - matches working implementation"""
        enhanced = f"{prompt}, ultra detailed, professional, vibrant, 4K, high quality"
        
        # Check if API token is available
        if not self.replicate_api_token:
            raise Exception("Replicate API token required")
        
        # The model endpoint for Flux 1.1 Pro
        model_url = "https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro/predictions"
        
        # Input data for image generation
        input_data = {
            "input": {
                "prompt": enhanced,
                "aspect_ratio": "9:16",  # Instagram Reels format
                "output_format": "jpg",  # Request JPG instead of WebP
                "output_quality": 90,
                "prompt_upsampling": True  # Better quality
            }
        }
        
        # Headers with Authorization token
        headers = {
            "Authorization": f"Token {self.replicate_api_token}",
            "Content-Type": "application/json",
            "Prefer": "wait",  # Wait for result
        }
        
        # Step 1: Send POST request to create the prediction
        response = requests.post(model_url, headers=headers, json=input_data, timeout=30)
        
        if response.status_code == 201:
            prediction = response.json()
            
            # Get the URL to check status of the prediction
            prediction_url = prediction.get("urls", {}).get("get")
            
            if not prediction_url:
                raise Exception("No prediction URL returned")
            
            # Step 2: Poll the prediction status
            for attempt in range(12):  # 12 * 5 = 60 seconds max
                time.sleep(5)  # Wait before checking status
                status_response = requests.get(prediction_url, headers=headers, timeout=10)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status")
                    
                    if status == "succeeded":
                        # The image generation succeeded, get the image URL
                        image_url = status_data.get("output")
                        
                        if not image_url:
                            raise Exception("No output URL in response")
                        
                        # If output is a list, take first element
                        if isinstance(image_url, list) and len(image_url) > 0:
                            image_url = image_url[0]
                        
                        print(f"      📥 Downloading from: {image_url[:60]}...")
                        
                        # Step 3: Download the generated image
                        img_response = requests.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            # Return base64 encoded image
                            return base64.b64encode(img_response.content).decode()
                        else:
                            raise Exception(f"Failed to download image: {img_response.status_code}")
                    
                    elif status in ["failed", "canceled"]:
                        raise Exception(f"Prediction {status}")
                    
                    # Still processing, continue polling
                    print(f"      ⏳ Status: {status}, waiting...")
                else:
                    raise Exception(f"Failed to get status: {status_response.status_code}")
            
            # Timeout after all retries
            raise Exception("Replicate generation timeout after 60s")
        else:
            error_msg = response.text[:200] if response.text else "Unknown error"
            raise Exception(f"Failed to create prediction ({response.status_code}): {error_msg}")

    def _huggingface_generate(self, prompt: str) -> str:
        """Generate image using Hugging Face Inference API (Flux)"""
        enhanced = f"{prompt}, ultra detailed, professional, vibrant, 4K"
        
        # Using FLUX.1-schnell (fast, Apache 2.0 license, free)
        API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        
        headers = {}
        if self.huggingface_api_token:
            headers['Authorization'] = f'Bearer {self.huggingface_api_token}'
        
        payload = {"inputs": enhanced}
        
        # Hugging Face may queue requests, retry with backoff
        for attempt in range(3):
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                # Check if it's actual image data
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type:
                    return base64.b64encode(response.content).decode()
                else:
                    # Might be JSON with error or loading status
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            if 'loading' in error_data['error'].lower():
                                print(f"      ⏳ Model loading, waiting {5 * (attempt + 1)}s...")
                                time.sleep(5 * (attempt + 1))
                                continue
                    except:
                        pass
            elif response.status_code == 503:
                print(f"      ⏳ Service unavailable, retrying in {5 * (attempt + 1)}s...")
                time.sleep(5 * (attempt + 1))
                continue
            
            break
        
        raise Exception(f"HuggingFace API error: {response.status_code}")

    def _pollinations_generate(self, prompt: str) -> str:
        """Generate image using Pollinations AI (Original fallback)"""
        enhanced = f"{prompt}, ultra detailed, professional, vibrant, 4K"
        encoded = quote(enhanced)

        attempts = [
            f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&model=flux",
            f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&model=turbo",
        ]

        for url in attempts:
            try:
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    return base64.b64encode(response.content).decode()
            except Exception as e:
                print(f"      ⚠️ Attempt failed: {str(e)[:50]}")
            time.sleep(2)

        raise Exception("Pollinations generation failed")

    def _enhance_image_optional(self, original_path: str, temp_dir: str, index: int) -> str:
        """
        Optional Cloudinary AI enhancement with full fallback safety
        
        Returns:
            - Enhanced image path if successful
            - Original image path if enhancement fails or disabled
        """
        
        # Skip if enhancement disabled
        if not self.enable_enhancement:
            print("   ⚠️ Enhancement disabled, using original")
            print(f"   ✅ Final: Using original version")
            return original_path
        
        try:
            print(f"   ✨ Attempting Cloudinary AI enhancement...")
            
            # Step 1: Upload to Cloudinary
            with open(original_path, 'rb') as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode()
            
            import hashlib
            timestamp = int(time.time())
            
            params_for_signature = {
                'timestamp': timestamp,
            }
            
            # Add upload_preset if available
            if self.cloudinary_upload_preset:
                params_for_signature['upload_preset'] = self.cloudinary_upload_preset
            
            string_to_sign = '&'.join(f"{k}={v}" for k, v in sorted(params_for_signature.items())) + self.cloudinary_api_secret
            signature = hashlib.sha1(string_to_sign.encode()).hexdigest()

            url = f"https://api.cloudinary.com/v1_1/{self.cloudinary_cloud_name}/image/upload"
            files = {'file': f"data:image/jpeg;base64,{image_base64}"}
            data = {
                'api_key': self.cloudinary_api_key,
                'signature': signature,
                'timestamp': timestamp,
            }
            
            if self.cloudinary_upload_preset:
                data['upload_preset'] = self.cloudinary_upload_preset

            response = requests.post(url, files=files, data=data, timeout=60)
            
            if response.status_code != 200:
                print(f"   ⚠️ Upload failed, using original")
                print(f"   ✅ Final: Using original version")
                return original_path
            
            response_data = response.json()
            base_url = response_data['secure_url']
            public_id = response_data['public_id']
            
            print(f"   ☁️ Uploaded to Cloudinary")
            
            # Step 2: Build enhanced URL with AI transformations (100% FREE features)
            transformations = [
                'q_auto',           # Auto quality
                'f_auto',           # Auto format (WebP/AVIF)
                'e_improve:outdoor', # AI color enhancement
                'e_sharpen:80',     # AI sharpening
                'c_fill',           # Fill crop mode
                'g_auto',           # AI smart gravity (face/object detection)
                'ar_9:16',          # 9:16 aspect ratio
                'w_1080',           # Width
                'h_1920'            # Height
            ]
            
            transformation_string = ','.join(transformations)
            
            # Parse URL and insert transformations
            parts = base_url.split('/upload/')
            if len(parts) == 2:
                enhanced_url = f"{parts[0]}/upload/{transformation_string}/{parts[1]}"
            else:
                enhanced_url = base_url
            
            print(f"   ✨ AI enhancement applied: quality↑, format↑, color↑, crop↑")
            
            # Step 3: Download enhanced image
            enhanced_response = requests.get(enhanced_url, timeout=30)
            
            if enhanced_response.status_code == 200:
                enhanced_path = f"{temp_dir}/img_enhanced_{index:03d}.jpg"
                with open(enhanced_path, 'wb') as f:
                    f.write(enhanced_response.content)
                
                print(f"   ⬇️ Enhanced image downloaded")
                
                # Cleanup: Delete from Cloudinary to save storage
                self._delete_cloudinary_image(public_id)
                print(f"   🗑️ Deleted from Cloudinary")
                
                print(f"   ✅ Final: Using enhanced version")
                return enhanced_path
            else:
                print(f"   ⚠️ Download failed, using original")
                print(f"   ✅ Final: Using original version")
                return original_path
                
        except Exception as e:
            print(f"   ⚠️ Enhancement failed ({str(e)[:50]}), using original")
            print(f"   ✅ Final: Using original version")
            return original_path
    
    def _delete_cloudinary_image(self, public_id: str):
        """Silently delete image from Cloudinary (cleanup)"""
        try:
            auth = (self.cloudinary_api_key, self.cloudinary_api_secret)
            url = f"https://api.cloudinary.com/v1_1/{self.cloudinary_cloud_name}/resources/image/upload"
            data = {'public_ids[]': public_id}
            requests.delete(url, auth=auth, data=data, timeout=10)
        except:
            pass  # Silent fail - not critical

    def _generate_ai_complete_package(self, niche: str, count: int, duration: int) -> dict:
        """AI generates complete video package"""
        prompt = f"""You are an Professional expert Instagram/Youtube content creator. Design a {duration}-second Reel with {count} clips for: {niche}.
15-20 hashtags and quality trending caption

For EACH clip, provide:
1. **image_prompt**: Detailed, realistic image description (4K quality)
2. **hook_score**: 1-10 (higher = more engaging, decides clip sequence)
3. **text_overlay**: 4-8 words, alphanumeric only
4. **text_position**: "top" | "center" | "bottom"
5. **text_color**: "white" | "yellow" | "red" | "cyan" | "green"
6. **text_size**: 40-60 (pixels)
7. **text_style**: "plain" | "box" | "shadow"
8. **motion_effect**: "zoom_in" | "zoom_out" | "pan_left" | "pan_right" | "static"
9. **clip_fade**: "fade_in_out" | "fade_in" | "fade_out" | "none"

Return ONLY valid JSON:
{{
  "clips": [
    {{
      "image_prompt": "Detailed scene description...",
      "hook_score": 9,
      "text_overlay": "Unlock Your Potential",
      "text_position": "top",
      "text_color": "yellow",
      "text_size": 55,
      "text_style": "box",
      "motion_effect": "zoom_in",
      "clip_fade": "fade_in_out"
    }}
  ],
  "caption": "Viral caption with call-to-action",
  "hashtags": ["#tag1", "#tag2", "#reels", "#viral"],
  "mood": "upbeat" // Choose from: "energetic", "calm", "upbeat", "intense", "chill",
  "title": "Quality title for youtube to shorts/video (have clear and easy catchy via search)",
  "description": "Youtube Vedio description",
  "tags": "Youtube Tags",
  "category_id" : "Youtube category id (as per niche content e.g 22)"
}}"""

        json_str = None

        if self.google_api_key:
            try:
                print("🤖 AI designing video package...")
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                response = model.generate_content(prompt)
                json_str = response.text.strip()
                print("✅ AI design complete")
            except Exception as e:
                print(f"⚠️ Gemini failed: {str(e)[:100]}")
                try:
                    print("🔄 Using Hugging Face...")
                    json_str = self._huggingface_text_generate(prompt)
                    print("✅ Hugging Face design complete")
                except Exception as e:
                    print(f"⚠️ Hugging Face failed: {str(e)}")

        # if (not json_str or len(json_str) < 50) and self.openai_api_key:
        #     try:
        #         print("🔄 Using OpenAI...")
        #         response = requests.post(
        #             'https://api.openai.com/v1/chat/completions',
        #             headers={'Authorization': f'Bearer {self.openai_api_key}', 'Content-Type': 'application/json'},
        #             json={
        #                 'model': 'gpt-4o-mini',
        #                 'messages': [
        #                     {'role': 'system', 'content': 'You are a video editor. Return only valid JSON.'},
        #                     {'role': 'user', 'content': prompt}
        #                 ],
        #                 'temperature': 0.9,
        #                 'max_tokens': 3000
        #             },
        #             timeout=30
        #         )
        #         if response.status_code == 200:
        #             json_str = response.json()['choices'][0]['message']['content'].strip()
        #             print("✅ OpenAI design complete")
        #     except Exception as e:
        #         print(f"⚠️ OpenAI failed: {str(e)[:100]}")

        # --- NEW HUGGING FACE FALLBACK ---
        # if (not json_str or len(json_str) < 50) and self.huggingface_api_token:
        #     try:
        #         print("🔄 Using Hugging Face...")
        #         json_str = self._huggingface_text_generate(prompt)
        #         print("✅ Hugging Face design complete")
        #     except Exception as e:
        #         print(f"⚠️ Hugging Face failed: {str(e)[:100]}")
        # --- END OF NEW BLOCK ---

        if not json_str:
            raise Exception("AI generation failed")

        return self._parse_and_validate(json_str, niche, count)

    def _parse_and_validate(self, json_str: str, niche: str, count: int) -> dict:
        """Parse and validate AI output"""
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0].strip()
        elif '{' in json_str:
            json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]

        try:
            data = json.loads(json_str)

            if 'clips' in data and len(data['clips']) >= count:
                for clip in data['clips'][:count]:
                    clip['text_overlay'] = ''.join(c for c in clip.get('text_overlay', 'Text') if c.isalnum() or c.isspace()).strip()[:25]
                    
                    if clip.get('text_position') not in ['top', 'center', 'bottom']:
                        clip['text_position'] = 'center'
                    if clip.get('text_color') not in ['white', 'yellow', 'red', 'cyan', 'green']:
                        clip['text_color'] = 'white'
                    if clip.get('text_style') not in ['plain', 'box', 'shadow']:
                        clip['text_style'] = 'box'
                    if clip.get('motion_effect') not in ['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'static']:
                        clip['motion_effect'] = 'zoom_in'
                    if clip.get('clip_fade') not in ['fade_in_out', 'fade_in', 'fade_out', 'none']:
                        clip['clip_fade'] = 'fade_in_out'

                    clip['text_size'] = max(35, min(int(clip.get('text_size', 50)), 60))
                    clip['hook_score'] = clip.get('hook_score', 5)

                data['clips'] = data['clips'][:count]
            else:
                raise ValueError("Invalid clips structure")

            data['mood'] = data.get('mood', 'upbeat')
            return data

        except Exception as e:
            print(f"⚠️ Parse error: {e}, using fallback")
            return {
                'clips': [{
                    'image_prompt': f'{niche} professional scene {i+1}',
                    'hook_score': 5,
                    'text_overlay': f'Scene {i+1}',
                    'text_position': 'center',
                    'text_color': 'white',
                    'text_size': 50,
                    'text_style': 'plain',
                    'motion_effect': 'zoom_in' if i % 2 == 0 else 'zoom_out',
                    'clip_fade': 'fade_in_out'
                } for i in range(count)],
                'caption': f'Amazing {niche} content',
                'hashtags': ['#reels', '#viral'],
                'mood': 'upbeat',
                'title': f'Amazing {niche} content',
                'description': f'Amazing {niche} content',
                'tags': ['#reels', '#viral'],
                'category_id': '22'
            }

    def _download_music(self, mood: str) -> str:
        """Download music or generate silent audio"""
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

    def _generate_silent_audio(self) -> str:
        """Generate silent audio fallback"""
        silent_path = f"{tempfile.gettempdir()}/silent.mp3"
        cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
               '-t', '60', '-q:a', '9', '-acodec', 'libmp3lame', silent_path]
        subprocess.run(cmd, check=True, capture_output=True)
        return silent_path

    def _create_smart_video(self, images: list, clips: list, music_path: str,
                           output_path: str, duration_per: float, total_duration: int, temp_dir: str):
        """Create video with calibrated effects per clip"""
        print(f"\n🎞️ Creating {len(images)} video clips (each {duration_per:.2f}s)...")

        processed_clips = []
        fps = 30

        for i, (img, clip) in enumerate(zip(images, clips)):
            processed_clip_path = f"{temp_dir}/processed_clip_{i:03d}.mp4"
            print(f"\n🎬 Clip {i+1}/{len(images)}: '{clip['text_overlay']}'")

            working_img = img
            if not self.has_drawtext and clip['text_overlay']:
                working_img = f"{temp_dir}/img_text_{i:03d}.jpg"
                self._add_text_with_pil(img, working_img, clip)

            vf_filters = []

            zoom_filter = self._build_calibrated_zoom_filter(clip['motion_effect'], duration_per)
            if zoom_filter:
                vf_filters.append(zoom_filter)
            else:
                vf_filters.append(f"trim=duration={duration_per},setpts=PTS-STARTPTS")

            vf_filters.append(f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps={fps}")

            if self.has_drawtext:
                text_filter = self._build_text_filter(clip)
                if text_filter:
                    vf_filters.append(text_filter)

            fade_effect = clip.get('clip_fade', 'fade_in_out')
            if fade_effect == "fade_in_out":
                fade_dur = min(0.3, duration_per / 3)
                fade_out_start = duration_per - fade_dur
                vf_filters.append(f"fade=t=in:st=0:d={fade_dur},fade=t=out:st={fade_out_start}:d={fade_dur}")
            elif fade_effect == "fade_in":
                fade_dur = min(0.3, duration_per / 3)
                vf_filters.append(f"fade=t=in:st=0:d={fade_dur}")
            elif fade_effect == "fade_out":
                fade_dur = min(0.3, duration_per / 3)
                fade_out_start = duration_per - fade_dur
                vf_filters.append(f"fade=t=out:st={fade_out_start}:d={fade_dur}")

            vf_filters.append('format=yuv420p')
            vf_chain = ','.join(vf_filters)

            cmd = [
                'ffmpeg', '-y', '-loop', '1', '-i', working_img,
                '-vf', vf_chain,
                '-t', str(duration_per + 0.1),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p', '-an',
                processed_clip_path
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                processed_clips.append(processed_clip_path)
                print(f"   ✅ Clip {i+1} processed successfully")
                self._verify_duration(processed_clip_path, duration_per, f"Clip {i+1}")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed: {e.stderr[-200:]}")
                raise

        print(f"\n🔗 Concatenating {len(processed_clips)} clips...")
        concat_file = f"{temp_dir}/concat.txt"
        with open(concat_file, 'w') as f:
            for clip_path in processed_clips:
                f.write(f"file '{clip_path}'\n")

        video_only_path = f"{temp_dir}/video_only.mp4"
        concat_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file, '-c', 'copy', video_only_path]

        subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
        self._verify_duration(video_only_path, total_duration, "concatenated video")

        print("🎵 Adding music...")
        final_cmd = [
            'ffmpeg', '-y', '-i', video_only_path, '-i', music_path,
            '-t', str(total_duration), '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
            '-shortest', output_path
        ]

        subprocess.run(final_cmd, check=True, capture_output=True, text=True)
        self._verify_duration(output_path, total_duration, "final video")
        print(f"✅ Complete video: {output_path}")

    def _build_calibrated_zoom_filter(self, effect: str, duration: float) -> str:
        """Build calibrated zoom/pan filter spanning exact duration"""
        if effect == 'static':
            return None

        fps = 30
        total_frames = int(duration * fps)

        if effect == 'zoom_in':
            return f"zoompan=z='1.0+(0.5*on/{total_frames})':d={total_frames}:s=1080x1920:fps={fps}"
        elif effect == 'zoom_out':
            return f"zoompan=z='1.5-(0.5*on/{total_frames})':d={total_frames}:s=1080x1920:fps={fps}"
        elif effect == 'pan_left':
            return f"zoompan=z='1.2':x='iw/2-(iw/zoom/2)-(iw*0.15*on/{total_frames})':y='ih/2-(ih/zoom/2)':d={total_frames}:s=1080x1920:fps={fps}"
        elif effect == 'pan_right':
            return f"zoompan=z='1.2':x='iw/2-(iw/zoom/2)+(iw*0.15*on/{total_frames})':y='ih/2-(ih/zoom/2)':d={total_frames}:s=1080x1920:fps={fps}"

        return None

    def _build_text_filter(self, clip: dict) -> str:
        """Build FFmpeg drawtext filter"""
        text = clip['text_overlay']
        if not text or len(text) < 2:
            return None

        text_escaped = text.replace("'", "\\'")

        pos_map = {
            'top': 'x=(w-text_w)/2:y=100',
            'center': 'x=(w-text_w)/2:y=(h-text_h)/2',
            'bottom': 'x=(w-text_w)/2:y=h-text_h-100'
        }
        pos = pos_map.get(clip['text_position'], 'x=(w-text_w)/2:y=(h-text_h)/2')

        color_map = {'white': 'white', 'yellow': 'yellow', 'red': 'red', 'cyan': 'cyan', 'green': 'green'}
        color = color_map.get(clip['text_color'], 'white')
        size = clip['text_size']

        if clip['text_style'] == 'box':
            return f"drawtext=text='{text_escaped}':fontsize={size}:fontcolor={color}:{pos}:box=1:boxcolor=black@0.7:boxborderw=5"
        elif clip['text_style'] == 'shadow':
            return f"drawtext=text='{text_escaped}':fontsize={size}:fontcolor={color}:{pos}:shadowcolor=black:shadowx=3:shadowy=3"
        else:
            return f"drawtext=text='{text_escaped}':fontsize={size}:fontcolor={color}:{pos}"

    def _add_text_with_pil(self, input_img: str, output_img: str, clip: dict):
        """Add text overlay using PIL (Pillow)"""
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.open(input_img)
            draw = ImageDraw.Draw(img)

            text = clip['text_overlay']
            size = clip['text_size']
            color_map = {'white': (255, 255, 255), 'yellow': (255, 255, 0), 'red': (255, 0, 0), 'cyan': (0, 255, 255), 'green': (0, 255, 0)}
            text_color = color_map.get(clip['text_color'], (255, 255, 255))

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", size)
                except:
                    font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            img_width, img_height = img.size

            if clip['text_position'] == 'top':
                x, y = (img_width - text_width) // 2, 100
            elif clip['text_position'] == 'bottom':
                x, y = (img_width - text_width) // 2, img_height - text_height - 100
            else:
                x, y = (img_width - text_width) // 2, (img_height - text_height) // 2

            if clip['text_style'] == 'box':
                padding = 10
                draw.rectangle([x - padding, y - padding, x + text_width + padding, y + text_height + padding], fill=(0, 0, 0, 180))
                draw.text((x, y), text, font=font, fill=text_color)
            elif clip['text_style'] == 'shadow':
                draw.text((x + 3, y + 3), text, font=font, fill=(0, 0, 0))
                draw.text((x, y), text, font=font, fill=text_color)
            else:
                draw.text((x, y), text, font=font, fill=text_color)

            img.save(output_img, quality=95)
            print(f"   📝 Text added: '{text}'")

        except ImportError:
            print(f"   ⚠️ PIL not installed")
            import shutil
            shutil.copy(input_img, output_img)
        except Exception as e:
            print(f"   ⚠️ PIL failed: {e}")
            import shutil
            shutil.copy(input_img, output_img)

    def _verify_duration(self, video_path: str, expected: float, label: str):
        """Verify video duration matches expected"""
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            actual = float(result.stdout.strip())
            status = "✅" if abs(actual - expected) < 0.5 else "⚠️"
            print(f"{status} {label}: {actual:.2f}s (expected: {expected:.2f}s)")
        except:
            pass


class CloudinaryVideoUploader:
    @staticmethod
    def upload_video(video_base64: str, cloud_name: str, upload_preset: str, 
                     api_key: str, api_secret: str) -> dict:
        """Upload video to Cloudinary CDN
        
        Returns dict with 'secure_url' and 'public_id' for later deletion
        """
        import hashlib
        
        timestamp = int(time.time())
        
        # Parameters for signature
        params_for_signature = {
            'timestamp': timestamp,
            'upload_preset': upload_preset,
        }
        
        # Create signature string (sorted alphabetically)
        string_to_sign = '&'.join(f"{k}={v}" for k, v in sorted(params_for_signature.items())) + api_secret
        signature = hashlib.sha1(string_to_sign.encode()).hexdigest()

        url = f"https://api.cloudinary.com/v1_1/{cloud_name}/video/upload"
        files = {'file': f"data:video/mp4;base64,{video_base64}"}
        data = {
            'api_key': api_key,
            'signature': signature,
            'timestamp': timestamp,
            'upload_preset': upload_preset,
        }

        print("\n☁️ Uploading video to Cloudinary...")
        response = requests.post(url, files=files, data=data, timeout=120)
        response_data = response.json()

        if response.status_code != 200:
            raise Exception(f"Upload failed: {response_data.get('error', {}).get('message')}")
        
        print(f"✅ Video uploaded: {response_data['secure_url']}")
        print(f"📋 Public ID: {response_data['public_id']}")
        
        # Return both URL and public_id for potential deletion
        return {
            'secure_url': response_data['secure_url'],
            'public_id': response_data['public_id'],
            'created_at': response_data.get('created_at')
        }
    
    @staticmethod
    def delete_video(cloud_name: str, api_key: str, api_secret: str, public_id: str) -> bool:
        """Delete a video from Cloudinary using Admin API"""
        
        # Use HTTP Basic Authentication (correct method for Admin API)
        auth = (api_key, api_secret)
        
        # Delete endpoint
        url = f"https://api.cloudinary.com/v1_1/{cloud_name}/resources/video/upload"
        
        # Data payload
        data = {
            'public_ids[]': public_id,
            'invalidate': True  # Remove from CDN cache immediately
        }
        
        print(f"\n🗑️ Deleting video from Cloudinary: {public_id}")
        
        try:
            response = requests.delete(url, auth=auth, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                deleted_status = result.get('deleted', {}).get(public_id)
                
                if deleted_status == 'deleted':
                    print(f"✅ Video deleted from Cloudinary")
                    return True
                elif deleted_status == 'not_found':
                    print(f"⚠️ Video not found (may already be deleted)")
                    return False
                else:
                    print(f"⚠️ Unexpected status: {deleted_status}")
                    return False
            else:
                print(f"⚠️ Delete failed (HTTP {response.status_code}): {response.text}")
                return False
                
        except Exception as e:
            print(f"⚠️ Delete error: {str(e)}")
            return False


class InstagramReelPublisher:
    def __init__(self):
        """Initialize Instagram Graph API publisher"""
        self.api_version = 'v20.0'
        self.base_url = f'https://graph.facebook.com/{self.api_version}'

    def publish_reel(self, account_id: str, access_token: str, video_url: str, caption: str) -> str:
        """Publish reel to Instagram"""
        print("\n📱 Publishing to Instagram...")
        
        create_url = f"{self.base_url}/{account_id}/media"
        create_params = {
            'media_type': 'REELS',
            'video_url': video_url,
            'caption': caption,
            'share_to_feed': True,
            'access_token': access_token
        }

        print("📱 Creating container...")
        response = requests.post(create_url, data=create_params, timeout=30)
        data = response.json()

        if response.status_code != 200:
            raise Exception(data.get('error', {}).get('message'))

        container_id = data['id']
        print(f"✅ Container created: {container_id}")

        print("⏳ Processing video...")
        for i in range(30):
            status_response = requests.get(
                f"https://graph.facebook.com/{container_id}",
                params={'fields': 'status_code', 'access_token': access_token},
                timeout=10
            )
            status_code = status_response.json().get('status_code')
            print(f"   Status: {status_code}")

            if status_code == 'FINISHED':
                break
            elif status_code in ['ERROR', 'EXPIRED']:
                raise Exception(f"Processing failed: {status_code}")
            time.sleep(5)

        print("🚀 Publishing to feed...")
        publish_response = requests.post(
            f"{self.base_url}/{account_id}/media_publish",
            data={'creation_id': container_id, 'access_token': access_token},
            timeout=30
        )
        publish_data = publish_response.json()

        if publish_response.status_code != 200:
            raise Exception(publish_data.get('error', {}).get('message'))

        print(f"✅ Published to Instagram: {publish_data['id']}")
        return publish_data['id']


class YouTubePublisher:
    """Publish videos to YouTube via Data API v3"""
    
    def __init__(self, client_id=None, client_secret=None, refresh_token=None, token_uri=None):
        """Initialize YouTube API client with proper validation"""
        print("\n📺 Initializing YouTube Publisher...")
        
        # Get credentials with fallback to environment variables
        self.client_id = client_id or os.getenv("CLIENT_ID_YOUTUBE")
        self.client_secret = client_secret or os.getenv("CLIENT_SECRET_YOUTUBE")
        self.refresh_token = refresh_token or os.getenv("REFRESH_TOKEN_YOUTUBE")
        self.token_uri = token_uri or os.getenv("YT_TOKEN_URI", "https://oauth2.googleapis.com/token")
        
        # Validate all required credentials are present
        missing = []
        if not self.client_id:
            missing.append("CLIENT_ID_YOUTUBE")
        if not self.client_secret:
            missing.append("CLIENT_SECRET_YOUTUBE")
        if not self.refresh_token:
            missing.append("REFRESH_TOKEN_YOUTUBE")
        
        if missing:
            raise ValueError(f"Missing required YouTube credentials: {', '.join(missing)}")
        
        # Validate credential types (catch the tuple bug)
        if not isinstance(self.client_id, str):
            raise TypeError(f"client_id must be str, got {type(self.client_id)}")
        if not isinstance(self.client_secret, str):
            raise TypeError(f"client_secret must be str, got {type(self.client_secret)}")
        if not isinstance(self.refresh_token, str):
            raise TypeError(f"refresh_token must be str, got {type(self.refresh_token)}")
        
        # Validate credential format (basic sanity checks)
        if len(self.client_id) < 10:
            raise ValueError("client_id appears invalid (too short)")
        if len(self.client_secret) < 10:
            raise ValueError("client_secret appears invalid (too short)")
        if len(self.refresh_token) < 20:
            raise ValueError("refresh_token appears invalid (too short)")
        
        print("   ✅ All credentials validated")
        
        try:
            # Create credentials object
            self.creds = Credentials(
                token=None,  # Will be refreshed automatically
                refresh_token=self.refresh_token,
                token_uri=self.token_uri,
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=['https://www.googleapis.com/auth/youtube.upload']
            )
            
            # Test token refresh immediately to catch auth errors early
            print("   🔄 Testing token refresh...")
            from google.auth.transport.requests import Request
            self.creds.refresh(Request())
            print("   ✅ Token refresh successful")
            
            # Create authorized HTTP client
            # self.http = AuthorizedHttp(self.creds)
            
            # Build YouTube API client
            self.youtube = build("youtube", "v3", credentials=self.creds)
            
            print("✅ YouTube client initialized successfully")
            
        except Exception as e:
            error_msg = str(e)
            if "invalid_client" in error_msg.lower():
                raise ValueError(
                    "OAuth client is invalid. Please verify:\n"
                    "  1. CLIENT_ID_YOUTUBE and CLIENT_SECRET_YOUTUBE are correct\n"
                    "  2. OAuth client exists in Google Cloud Console\n"
                    "  3. OAuth client is not deleted or disabled"
                )
            elif "invalid_grant" in error_msg.lower():
                raise ValueError(
                    "Refresh token is invalid or expired. Please:\n"
                    "  1. Generate a new refresh token\n"
                    "  2. Ensure the token hasn't been revoked\n"
                    "  3. Check that refresh token matches the client credentials"
                )
            else:
                raise Exception(f"Failed to initialize YouTube client: {e}")
    
    def publish_video(self, video_path: str, title: str, description: str,
                      tags=None, privacy="public", category_id="22") -> str:
        """Upload a video to YouTube with robust error handling"""
        
        print(f"\n🎬 Preparing to upload: {video_path}")
        
        # Validate video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Validate file size (YouTube limit: 256GB, but warn for large files)
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"   📦 Video size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 10000:  # 10GB warning
            print(f"   ⚠️ Large file detected. Upload may take a while...")
        
        # Validate title and description
        if not title or len(title.strip()) == 0:
            raise ValueError("Video title cannot be empty")
        if len(title) > 100:
            print(f"   ⚠️ Title truncated to 100 chars (was {len(title)})")
            title = title[:97] + "..."
        
        if description and len(description) > 5000:
            print(f"   ⚠️ Description truncated to 5000 chars (was {len(description)})")
            description = description[:4997] + "..."
        
        # Validate tags
        if tags:
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',')]
            if len(tags) > 500:
                print(f"   ⚠️ Too many tags, using first 500")
                tags = tags[:500]
        
        # Validate category ID
        valid_categories = ["1", "2", "10", "15", "17", "18", "19", "20", "21", "22", 
                           "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", 
                           "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44"]
        if category_id not in valid_categories:
            print(f"   ⚠️ Invalid category_id '{category_id}', using '22' (People & Blogs)")
            category_id = "22"
        
        # Create upload media object (256KB chunks for better progress reporting)
        media = MediaFileUpload(
            video_path, 
            chunksize=256*1024,  # 256KB chunks
            resumable=True,
            mimetype='video/mp4'
        )
        
        # Prepare video metadata
        body = {
            "snippet": {
                "title": title,
                "description": description or "",
                "tags": tags or [],
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": False  # Important: declare this
            },
        }
        
        print("⬆️ Starting upload...")
        
        # Create upload request
        request = self.youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )
        
        response = None
        retry = 0
        max_retries = 5
        
        while response is None:
            try:
                status, response = request.next_chunk()
                
                if status:
                    progress = int(status.progress() * 100)
                    print(f"📤 Upload progress: {progress}%")
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle specific error types
                if "invalid_client" in error_str or "invalid_grant" in error_str:
                    raise Exception(
                        f"Authentication failed: {e}\n"
                        "Your OAuth credentials may be invalid or expired.\n"
                        "Please regenerate your refresh token."
                    )
                
                elif "quota" in error_str:
                    raise Exception(
                        f"YouTube API quota exceeded: {e}\n"
                        "Your daily upload quota is exhausted. Try again tomorrow."
                    )
                
                elif "file" in error_str and "size" in error_str:
                    raise Exception(
                        f"File size error: {e}\n"
                        "Video may be too large or corrupted."
                    )
                
                # Retry for transient errors
                retry += 1
                if retry > max_retries:
                    raise Exception(f"Upload failed after {max_retries} retries: {e}")
                
                wait_time = min(2 ** retry, 32)  # Exponential backoff, max 32s
                print(f"⚠️ Error: {str(e)[:100]}")
                print(f"   Retrying in {wait_time}s... (attempt {retry}/{max_retries})")
                time.sleep(wait_time)
        
        # Extract video ID from response
        video_id = response.get("id")
        
        if not video_id:
            raise Exception("Upload completed but no video ID returned")
        
        print(f"✅ Upload completed! Video ID: {video_id}")
        print(f"🔗 Watch at: https://www.youtube.com/watch?v={video_id}")
        
        # Video processing status (optional)
        print("⏳ Video is now being processed by YouTube...")
        print("   Processing may take several minutes depending on video length")
        
        return video_id
    
    def get_video_status(self, video_id: str) -> dict:
        """Check the processing status of an uploaded video"""
        try:
            request = self.youtube.videos().list(
                part="status,processingDetails",
                id=video_id
            )
            response = request.execute()
            
            if not response.get("items"):
                return {"error": "Video not found"}
            
            video = response["items"][0]
            status = video.get("status", {})
            processing = video.get("processingDetails", {})
            
            return {
                "upload_status": status.get("uploadStatus"),
                "privacy_status": status.get("privacyStatus"),
                "processing_status": processing.get("processingStatus"),
                "processing_progress": processing.get("processingProgress", {})
            }
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main execution function"""
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
    replicate_api_token = os.getenv('REPLICATE_API_TOKEN')
    huggingface_api_token = os.getenv('HUGGINGFACE_API_TOKEN')
    refresh_token = os.getenv("REFRESH_TOKEN_YOUTUBE")
    token_uri = os.getenv("YT_TOKEN_URI", "https://oauth2.googleapis.com/token")
    client_id = os.getenv("CLIENT_ID_YOUTUBE")
    client_secret = os.getenv("CLIENT_SECRET_YOUTUBE")

    # Debug: Print credential status (without exposing secrets)
    print("\n🔑 YouTube Credentials Check:")
    print(f"   Client ID: {'✅ Set' if client_id else '❌ Missing'}")
    print(f"   Client Secret: {'✅ Set' if client_secret else '❌ Missing'}")
    print(f"   Refresh Token: {'✅ Set' if refresh_token else '❌ Missing'}")
    print(f"   Token URI: {token_uri}")
    
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
        niche = os.getenv('REEL_NICHE', 'any niche tranding now')
        num_images = int(os.getenv('REEL_IMAGES', '5'))
        duration = int(os.getenv('REEL_DURATION', '15'))

        print(f"🎯 Niche: {niche}")
        print(f"📸 Clips: {num_images}")
        print(f"⏱️ Duration: {duration}s")
        print("=" * 50)

        # Step 1: Generate Reel (with multi-source generation + optional enhancement)
        generator = TrulyAIReelGenerator(
            google_api_key, 
            openai_api_key,
            cloudinary_cloud_name,
            cloudinary_api_key,
            cloudinary_api_secret,
            cloudinary_upload_preset,
            replicate_api_token,
            huggingface_api_token
        )
        reel_data = generator.generate_reel(niche, num_images, duration)

        # Step 2: Upload to Cloudinary
        uploader = CloudinaryVideoUploader()
        upload_result = uploader.upload_video(
            reel_data['video_base64'],
            cloudinary_cloud_name,
            cloudinary_upload_preset,
            cloudinary_api_key,
            cloudinary_api_secret
        )

        video_url = upload_result['secure_url']
        public_id = upload_result['public_id']

        # Step 3: Publish to Instagram
        publisher = InstagramReelPublisher()
        full_caption = f"{reel_data['caption']}\n\n{' '.join(reel_data['hashtags'])}"

        post_id = publisher.publish_reel(
            instagram_account_id,
            instagram_access_token,
            video_url,
            full_caption
        )

        print("\n" + "=" * 50)
        print("🎉 SUCCESS!")
        print(f"📝 Caption: {reel_data['caption']}")
        print(f"🆔 Post ID: {post_id}")
        print(f"🔗 Video URL: {video_url}")
        
        # Step 4: Delete from Cloudinary (after successful Instagram upload)
        print("=" * 50)
        print("🧹 Cleaning up Cloudinary...")
        
        # Wait to ensure Instagram has processed the video
        print("⏳ Waiting 10 seconds for Instagram to cache video...")
        time.sleep(10)
        
        delete_success = uploader.delete_video(
            cloudinary_cloud_name,
            cloudinary_api_key,
            cloudinary_api_secret,
            public_id
        )
        
        if delete_success:
            print("✅ Cleanup complete - Video deleted from Cloudinary")
        else:
            print("⚠️ Could not delete from Cloudinary (but Instagram post is live)")
        
        print("=" * 50)
        print("✅ REEL GENERATION COMPLETE!")
        print("=" * 50)

        # Youtube Upload

        video_path = f"{reel_data['temp_dir']}/reel.mp4"
        yt_publisher = YouTubePublisher(
            client_id=client_id, 
            client_secret=client_secret, 
            refresh_token=refresh_token, 
            token_uri=token_uri)
        
        video_id = yt_publisher.publish_video(
            video_path=video_path,
            title=reel_data['title'],
            description=reel_data['description'],
            tags=reel_data['tags'],
            privacy="public",
            category_id = reel_data['category_id']
        )
        print(f"📺 YouTube Video: https://www.youtube.com/watch?v={video_id}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
