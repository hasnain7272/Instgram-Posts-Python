import json
import time
import base64
import requests
import os
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import google.generativeai as genai
# Import necessary for video upload to Cloudinary
from io import BytesIO 
# Using a running Gradio Space ID for the Text-to-Video task 
# NOTE: This ID points to a live user-deployed Space, which is often more stable.
HF_VIDEO_MODEL_ID = "camenduru/Modelscope-text-to-video" 



@dataclass
class InspirationPost:
    id: str
    username: str
    caption: str
    imageDescription: str


@dataclass
class GeneratedPost:
    video_url: str 
    caption: str
    hashtags: List[str]


@dataclass
class PostMetadata:
    id: str
    timestamp: str
    inspiration_source: str
    image_description_hash: str
    caption_keywords: List[str]
    hashtags_used: List[str]
    engagement_niche: str


class CloudinaryVideoUploader:
    def __init__(self, cloud_name: str, upload_preset: str, api_key: str, api_secret: str):
        self.cloud_name = cloud_name
        self.upload_preset = upload_preset
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = f"https://api.cloudinary.com/v1_1/{self.cloud_name}/video/upload"

    def generate_signature(self, params: dict) -> str:
        # Re-importing here for completeness, though it's already at the top
        import hashlib 
        
        excluded_params = {'file', 'cloud_name', 'resource_type', 'api_key'}
        filtered_params = {k: v for k, v in params.items() if k not in excluded_params}
        sorted_params = sorted(filtered_params.items())
        string_to_sign = '&'.join(f"{k}={v}" for k, v in sorted_params)
        string_to_sign += self.api_secret
        signature = hashlib.sha1(string_to_sign.encode('utf-8')).hexdigest()

        return signature
    
    # NEW METHOD: Uploads binary video data
    def upload_video_from_bytes(self, video_bytes: bytes, filename: str) -> str:
        timestamp = int(time.time())
        public_id = f"reel_{filename}_{timestamp}"
        
        params = {
            'api_key': self.api_key,
            'timestamp': timestamp,
            'upload_preset': self.upload_preset,
            'resource_type': 'video',
            'public_id': public_id
        }
        
        signature = self.generate_signature(params)
        
        data = {
            'api_key': self.api_key,
            'timestamp': timestamp,
            'upload_preset': self.upload_preset,
            'signature': signature,
            'public_id': public_id
        }
        
        # Use BytesIO to simulate a file for the requests library
        files = {'file': (f'{filename}.mp4', BytesIO(video_bytes), 'video/mp4')}
        
        print("☁️ Uploading video to Cloudinary...")
        response = requests.post(self.base_url, files=files, data=data, timeout=60)

        if response.status_code != 200:
            raise Exception(f"Cloudinary upload failed: {response.text}")

        video_data = response.json()
        
        # Return the secure HTTPS URL
        return video_data['secure_url'] 


class PostHistoryManager:
    # Initializing with Cloudinary credentials to manage history
    def __init__(self, cloudinary_cloud_name: str, cloudinary_upload_preset: str, api_key: str, api_secret: str):
        self.cloud_name = cloudinary_cloud_name
        self.upload_preset = cloudinary_upload_preset
        self.history_file = "post_history.json"
        self.api_key = api_key
        self.api_secret = api_secret

    # NOTE: The generate_signature and other methods are inherited from the original script
    # and were verified as correct for the Cloudinary raw file API.
    def generate_signature(self, params: dict) -> str:
        # Re-importing here for completeness, though it's already at the top
        import hashlib 

        excluded_params = {'file', 'cloud_name', 'resource_type', 'api_key'}
        filtered_params = {k: v for k, v in params.items() if k not in excluded_params}
        sorted_params = sorted(filtered_params.items())
        string_to_sign = '&'.join(f"{k}={v}" for k, v in sorted_params)
        string_to_sign += self.api_secret
        signature = hashlib.sha1(string_to_sign.encode('utf-8')).hexdigest()

        return signature

    def download_history(self) -> List[PostMetadata]:
        try:
            timestamp = int(time.time())
            params = {
                'api_key': self.api_key,
                'timestamp': timestamp,
            }
            signature = self.generate_signature(params)

            cache_buster = int(time.time())
            signed_url = f"https://res.cloudinary.com/{self.cloud_name}/raw/upload/v{cache_buster}/{self.history_file}?api_key={self.api_key}&timestamp={timestamp}&signature={signature}"

            headers = {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }

            response = requests.get(signed_url, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                posts = data.get('posts', [])
                return [PostMetadata(**post) for post in posts]
            else:
                # 404 is expected on first run
                print("No existing post history found, starting fresh")
                return []

        except Exception as e:
            print(f"Error downloading history: {e}")
            return []

    def upload_history(self, history: List[PostMetadata]) -> None:
        # (History upload logic is unchanged)
        try:
            history_data = {
                'posts': [asdict(post) for post in history],
                'updated_at': datetime.now().isoformat()
            }
            json_string = json.dumps(history_data, indent=2)

            timestamp = int(time.time())
            params = {
                'api_key': self.api_key,
                'public_id': self.history_file,
                'timestamp': timestamp,
                'upload_preset': self.upload_preset,
                'resource_type': 'raw' # Important for non-image files
            }

            signature = self.generate_signature(params)

            url = f"https://api.cloudinary.com/v1_1/{self.cloud_name}/raw/upload"
            files = {'file': ('post_history.json', json_string, 'application/json')}
            data = {
                'api_key': self.api_key,
                'public_id': self.history_file,
                'signature': signature,
                'timestamp': timestamp,
                'upload_preset': self.upload_preset,
            }

            response = requests.post(url, files=files, data=data, timeout=30)

            if response.status_code != 200:
                print(f"Warning: Failed to upload history: {response.text}")

        except Exception as e:
            print(f"Error uploading history: {e}")

    # (Duplicate content and niche logic remains unchanged)
    def is_duplicate_content(self, new_description: str, new_keywords: List[str], 
                           new_hashtags: List[str], history: List[PostMetadata]) -> bool:
        new_desc_hash = hashlib.md5(new_description.lower().encode()).hexdigest()
        recent_posts = [p for p in history if self._is_recent(p.timestamp, days=7)]

        for post in recent_posts:
            if post.image_description_hash == new_desc_hash:
                return True

            common_keywords = set(new_keywords) & set(post.caption_keywords)
            if len(common_keywords) / max(len(new_keywords), 1) > 0.6:
                return True

            common_hashtags = set(new_hashtags) & set(post.hashtags_used)
            if len(common_hashtags) >= 5:
                return True

        return False

    def get_next_niche(self, history: List[PostMetadata]) -> str:
        niches = ['fitness', 'motivation', 'food', 'travel', 'lifestyle', 'aesthetic', 'fashion', 'tech', 'productivity'] 
        recent_posts = history[-5:]
        recent_niches = [p.engagement_niche for p in recent_posts]

        for niche in niches:
            if niche not in recent_niches:
                return niche

        return niches[0]

    def _is_recent(self, timestamp_str: str, days: int) -> bool:
        try:
            post_time = datetime.fromisoformat(timestamp_str)
            return datetime.now() - post_time < timedelta(days=days)
        except:
            return False


class InstagramPostGenerator:
    def __init__(self, google_api_key: str, hf_token: str, cloudinary_uploader: CloudinaryVideoUploader):
        genai.configure(api_key=google_api_key)
        self.google_api_key = google_api_key
        self.hf_token = hf_token
        self.cloudinary_uploader = cloudinary_uploader
        # New URL structure for calling the Gradio-based Space API
self.hf_video_url = f"https://huggingface.co/spaces/{HF_VIDEO_MODEL_ID}/api/predict"
        self.search_templates = [
            "viral Instagram Reels {niche} high engagement 2025 Current",
            "trending {niche} short-form video content Instagram",
            "popular Instagram {niche} aesthetic vertical videos",
            "{season} Instagram trends {niche} viral Reels",
            "Instagram growth {niche} content formats trending"
        ]

    # (fetch_inspiration_posts logic is unchanged)
    def fetch_inspiration_posts(self, niche: str, season: str = None) -> List[InspirationPost]:
        if not season:
            season = self._get_current_season()

        search_queries = []
        for template in self.search_templates:
            query = template.format(niche=niche, season=season)
            search_queries.append(query)

        prompt = f"""
        Using Google Search, find 5 recent and visually compelling Instagram REELS using these search queries:
        {', '.join(search_queries[:3])}
        
        Focus on posts with high engagement potential in the {niche} niche.
        For each post, provide:
        - A detailed, vivid description of the **vertical video content** (imageDescription) 
        - The original username
        - The original caption
        
        The imageDescription should be detailed enough for AI **video** generation, specifying the action, scene, and vertical format (9:16).
        Return as valid JSON array with objects containing: id, username, caption, imageDescription fields.
        """

        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)

            json_string = response.text.strip()
            # Clean up JSON
            if '```json' in json_string:
                start = json_string.find('```json') + 7
                end = json_string.find('```', start)
                json_string = json_string[start:end].strip()
            elif '[' in json_string and ']' in json_string:
                start = json_string.find('[')
                end = json_string.rfind(']') + 1
                json_string = json_string[start:end]

            posts_data = json.loads(json_string)

            inspiration_posts = []
            for post_data in posts_data:
                inspiration_posts.append(InspirationPost(
                    id=post_data.get('id', f"post_{len(inspiration_posts)}"),
                    username=post_data.get('username', 'unknown'),
                    caption=post_data.get('caption', ''),
                    imageDescription=post_data.get('imageDescription', '')
                ))

            return inspiration_posts

        except Exception as error:
            print(f"Error fetching inspiration posts: {error}")
            raise Exception("Failed to find inspiration. Please check your API key and try again.")
    
    # NEW LOGIC: Text-to-Video via free Inference API + Cloudinary Upload
    def generate_ready_post(self, inspiration: InspirationPost, niche: str) -> GeneratedPost:
        try:
            video_prompt = f"""
            High-quality, professional **9:16 vertical video clip** for an Instagram Reel, 
            inspired by: "{inspiration.imageDescription}". The video should be dynamic, 
            aesthetic, and optimized for short-form social media. Max 5 seconds.
            Focus on {niche} content.
            """

            # --- VIDEO GENERATION LOGIC ---
            hf_headers = {"Authorization": f"Bearer {self.hf_token}"}
            # Gradio API expects 'data' containing a list of inputs (just the prompt here)
hf_payload = json.dumps({"data": [video_prompt]})

            print(f"🎬 Generating {niche} vertical video via Hugging Face Inference API...")
            
            # Use a higher timeout for video generation cold start and processing
            hf_response = requests.post(self.hf_video_url, headers=hf_headers, data=hf_payload, timeout=240) 

            if hf_response.status_code != 200:
                 # Check for the common error of model loading timeout
                 if 'loading' in hf_response.text or hf_response.status_code == 503:
                     raise Exception("Hugging Face model is loading or timed out. Try again in a minute.")
                 else:
                    raise Exception(f"Hugging Face Video API failed: {hf_response.text}")
            
            # The free API returns the raw binary video data
            video_bytes = hf_response.content
            print("✅ Raw video data retrieved. Size:", len(video_bytes) / (1024 * 1024), "MB")

            # Upload the video data to Cloudinary to get a public URL
            filename = f"{niche}_reel_{int(time.time())}"
            generated_video_url = self.cloudinary_uploader.upload_video_from_bytes(video_bytes, filename)
            print(f"🔗 Public Video URL created: {generated_video_url}")

            # --- TEXT GENERATION (Using Gemini) ---
            text_model = genai.GenerativeModel('gemini-2.0-flash-exp')

            text_prompt = f"""
            Create engaging Instagram content for a **REEL** in the {niche} niche.
            Inspiration caption: "{inspiration.caption}"
            Video description: "{inspiration.imageDescription}"
            
            Generate:
            1. A catchy caption (max 100 characters) that encourages views and comments
            2. 7 trending hashtags for {niche} **Reels** content
            
            Make it original and optimized for maximum likes and comments.
            
            Return as JSON:
            {{
                "caption": "your caption here",
                "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5", "#tag6", "#tag7"]
            }}
            """

            text_response = text_model.generate_content(
                text_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.8,
                    max_output_tokens=400
                )
            )

            json_string = text_response.text.strip()
            # Clean up JSON
            if '```json' in json_string:
                start = json_string.find('```json') + 7
                end = json_string.find('```', start)
                json_string = json_string[start:end].strip()
            elif '{' in json_string:
                start = json_string.find('{')
                end = json_string.rfind('}') + 1
                json_string = json_string[start:end]

            text_content = json.loads(json_string)

            return GeneratedPost(
                video_url=generated_video_url,
                caption=text_content.get('caption', f'Amazing {niche} content!'),
                hashtags=text_content.get('hashtags', [f'#{niche}', '#reels', '#viralreels', '#trending', '#shortformvideo', '#aesthetic', '#reeloftheday'])
            )

        except Exception as error:
            print(f"Error generating ready post: {error}")
            # Remember not to give false hopes.
            raise Exception("Failed to generate post content for Reel. Check API keys, especially the Hugging Face Token.")


    def _get_current_season(self) -> str:
        month = datetime.now().month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    def _extract_keywords(self, caption: str) -> List[str]:
        words = caption.lower().replace('#', '').replace('@', '').split()
        keywords = [word for word in words if len(word) > 3 and word.isalpha()]
        return keywords[:10]


class InstagramPublisher:
    def __init__(self):
        self.api_version = 'v20.0'
        self.base_url = f'https://graph.facebook.com/{self.api_version}'

    def publish_post(self, account_id: str, access_token: str, media_url: str, caption: str) -> str:
        try:
            create_url = f"{self.base_url}/{account_id}/media"
            create_params = {
                'video_url': media_url,
                'caption': caption,
                'media_type': 'REELS',
                'share_to_feed': 'true',
                'access_token': access_token
            }

            print(f"Creating media container for REEL...")
            # Increased timeout for initial video creation
            container_response = requests.post(create_url, data=create_params, timeout=60) 
            container_data = container_response.json()

            if container_response.status_code != 200 or 'id' not in container_data:
                error_msg = container_data.get('error', {}).get('message', 'Failed to create media container for Reel')
                raise Exception(error_msg)

            creation_id = container_data['id']
            print(f"Media container created: {creation_id}")

            # Polling for video processing status
            max_retries = 20
            for i in range(max_retries):
                status_url = f"https://graph.facebook.com/{creation_id}"
                status_params = {'fields': 'status_code', 'access_token': access_token}

                status_response = requests.get(status_url, params=status_params, timeout=10)
                status_data = status_response.json()

                status_code = status_data.get('status_code')
                print(f"Media status: {status_code} (attempt {i+1}/{max_retries})")

                if status_code == 'FINISHED':
                    break
                elif status_code in ['ERROR', 'EXPIRED']:
                    raise Exception(f"Media processing failed: {status_code}")

                if i == max_retries - 1:
                    raise Exception("Media processing timeout")

                time.sleep(10) # Wait longer (10s) for video processing

            print("Publishing REEL...")
            publish_url = f"{self.base_url}/{account_id}/media_publish"
            publish_params = {'creation_id': creation_id, 'access_token': access_token}

            publish_response = requests.post(publish_url, data=publish_params, timeout=30)
            publish_data = publish_response.json()

            if publish_response.status_code != 200 or 'id' not in publish_data:
                error_msg = publish_data.get('error', {}).get('message', 'Failed to publish Reel')
                raise Exception(error_msg)

            post_id = publish_data['id']
            print(f"🎉 Published successfully! Reel ID: {post_id}")
            return post_id

        except Exception as error:
            print(f"Error publishing Reel: {error}")
            raise Exception(f"Publishing failed: {str(error)}")


def main():
    print("🤖 Enhanced Instagram REEL Generator (Open-Source/Free) Starting...")

    google_api_key = os.getenv('GOOGLE_API_KEY')
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
    cloudinary_cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    cloudinary_upload_preset = os.getenv('CLOUDINARY_UPLOAD_PRESET')
    cloudinary_api_key = os.getenv('CLOUDINARY_API_KEY')
    cloudinary_api_secret = os.getenv('CLOUDINARY_API_SECRET')
    instagram_account_id = os.getenv('INSTAGRAM_ACCOUNT_ID')
    instagram_access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')
    
    # HUGGINGFACE_VIDEO_URL is no longer required as it is hardcoded

    required_vars = {
        'GOOGLE_API_KEY': google_api_key,
        'HUGGINGFACE_TOKEN': huggingface_token, # Needed for authentication
        'CLOUDINARY_CLOUD_NAME': cloudinary_cloud_name,
        'CLOUDINARY_UPLOAD_PRESET': cloudinary_upload_preset,
        'CLOUDINARY_API_KEY': cloudinary_api_key,
        'CLOUDINARY_API_SECRET': cloudinary_api_secret,
        'INSTAGRAM_ACCOUNT_ID': instagram_account_id,
        'INSTAGRAM_ACCESS_TOKEN': instagram_access_token
    }

    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        print(f"❌ Missing critical environment variables: {', '.join(missing_vars)}")
        print("Please set these variables to make the script fully functional.")
        return 

    try:
        # Initialize Cloudinary uploader first, as it's used for both history and video
        cloudinary_uploader = CloudinaryVideoUploader(
            cloudinary_cloud_name, cloudinary_upload_preset, cloudinary_api_key, cloudinary_api_secret
        )
        
        history_manager = PostHistoryManager(
            cloudinary_cloud_name, cloudinary_upload_preset, cloudinary_api_key, cloudinary_api_secret
        )
        
        generator = InstagramPostGenerator(
            google_api_key, huggingface_token, cloudinary_uploader
        )

        print("📥 Loading post history...")
        post_history = history_manager.download_history()
        print(f"Found {len(post_history)} previous posts")

        next_niche = history_manager.get_next_niche(post_history)
        print(f"🎯 Target niche: {next_niche}")

        max_attempts = 3
        generated_post = None
        
        for attempt in range(max_attempts):
            print(f"🔍 Fetching {next_niche} inspiration (attempt {attempt + 1})...")
            inspiration_posts = generator.fetch_inspiration_posts(next_niche)

            if not inspiration_posts:
                if attempt < max_attempts - 1:
                    continue
                raise Exception("No inspiration posts found")

            # The content generation will now perform the video generation/upload
            generated_post = generator.generate_ready_post(inspiration_posts[0], next_niche)

            keywords = generator._extract_keywords(generated_post.caption)
            hashtag_list = [tag.replace('#', '') for tag in generated_post.hashtags]

            if not history_manager.is_duplicate_content(
                inspiration_posts[0].imageDescription, keywords, hashtag_list, post_history):
                print("✅ Original content generated!")
                break
            else:
                print(f"⚠️ Content too similar to recent posts, retrying...")
                if attempt == max_attempts - 1:
                    raise Exception("Unable to generate original content after multiple attempts")

        if not generated_post:
             raise Exception("Failed to generate a valid post object.")

        print(f"📝 Caption: {generated_post.caption}")
        print(f"🏷️ Hashtags: {' '.join(generated_post.hashtags)}")
        media_url = generated_post.video_url
        print(f"🔗 Final Reel Video URL: {media_url}")

        print("📱 Publishing REEL to Instagram...")
        publisher = InstagramPublisher()
        full_caption = f"{generated_post.caption}\n\n{' '.join(generated_post.hashtags)}"

        post_id = publisher.publish_post(
            account_id=instagram_account_id,
            access_token=instagram_access_token,
            media_url=media_url,
            caption=full_caption
        )
        
        print("💾 Updating post history...")
        new_metadata = PostMetadata(
            id=post_id,
            timestamp=datetime.now().isoformat(),
            inspiration_source=f"{next_niche} viral content (Reel)",
            image_description_hash=hashlib.md5(inspiration_posts[0].imageDescription.lower().encode()).hexdigest(),
            caption_keywords=keywords,
            hashtags_used=hashtag_list,
            engagement_niche=next_niche
        )

        post_history.append(new_metadata)
        history_manager.upload_history(post_history)

        print("🎉 Reel published and history updated successfully!")

    except Exception as e:
        print(f"\n\n❌ ERROR: The automated process failed. Please check the error message and setup: {e}")
        # Re-raise the exception to stop execution gracefully
        raise


if __name__ == "__main__":
    main()
