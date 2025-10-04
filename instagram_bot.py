"""
Instagram Reel Auto-Generator using Google Veo 3 for Text-to-Video

Required packages:
pip install google-genai requests

Required Environment Variables:
- GOOGLE_API_KEY: Google Gemini API key (with Veo 3 access)
- CLOUDINARY_CLOUD_NAME: Cloudinary cloud name
- CLOUDINARY_UPLOAD_PRESET: Cloudinary upload preset
- CLOUDINARY_API_KEY: Cloudinary API key
- CLOUDINARY_API_SECRET: Cloudinary API secret
- INSTAGRAM_ACCOUNT_ID: Instagram Business Account ID
- INSTAGRAM_ACCESS_TOKEN: Instagram Graph API access token
"""

import json
import time
import requests
import os
import hashlib
import tempfile
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from io import BytesIO

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ google-genai not installed. Run: pip install google-genai")
    raise


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
        excluded_params = {'file', 'cloud_name', 'resource_type', 'api_key'}
        filtered_params = {k: v for k, v in params.items() if k not in excluded_params}
        sorted_params = sorted(filtered_params.items())
        string_to_sign = '&'.join(f"{k}={v}" for k, v in sorted_params)
        string_to_sign += self.api_secret
        signature = hashlib.sha1(string_to_sign.encode('utf-8')).hexdigest()
        return signature

    def upload_video_from_file(self, video_file_path: str, filename: str) -> str:
        """Upload video from local file to Cloudinary"""
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

        print("☁️ Uploading video to Cloudinary...")
        with open(video_file_path, 'rb') as video_file:
            files = {'file': video_file}
            response = requests.post(self.base_url, files=files, data=data, timeout=120)

        if response.status_code != 200:
            raise Exception(f"Cloudinary upload failed: {response.text}")

        video_data = response.json()
        return video_data['secure_url']

    def upload_video_from_url(self, video_url: str, filename: str) -> str:
        """Upload video from URL to Cloudinary"""
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
            'file': video_url,  # Cloudinary can fetch from URL
            'api_key': self.api_key,
            'timestamp': timestamp,
            'upload_preset': self.upload_preset,
            'signature': signature,
            'public_id': public_id
        }

        print("☁️ Uploading video to Cloudinary...")
        response = requests.post(self.base_url, data=data, timeout=120)

        if response.status_code != 200:
            raise Exception(f"Cloudinary upload failed: {response.text}")

        video_data = response.json()
        return video_data['secure_url']


class PostHistoryManager:
    def __init__(self, cloudinary_cloud_name: str, cloudinary_upload_preset: str, api_key: str, api_secret: str):
        self.cloud_name = cloudinary_cloud_name
        self.upload_preset = cloudinary_upload_preset
        self.history_file = "post_history.json"
        self.api_key = api_key
        self.api_secret = api_secret

    def generate_signature(self, params: dict) -> str:
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
                print("No existing post history found, starting fresh")
                return []

        except Exception as e:
            print(f"Error downloading history: {e}")
            return []

    def upload_history(self, history: List[PostMetadata]) -> None:
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
                'resource_type': 'raw'
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
    def __init__(self, google_api_key: str, cloudinary_uploader: CloudinaryVideoUploader):
        self.google_api_key = google_api_key
        self.cloudinary_uploader = cloudinary_uploader
        # Initialize Veo 3 client
        self.veo_client = genai.Client(api_key=google_api_key)
        
        self.search_templates = [
            "viral Instagram Reels {niche} high engagement 2025 Current",
            "trending {niche} short-form video content Instagram",
            "popular Instagram {niche} aesthetic vertical videos",
            "{season} Instagram trends {niche} viral Reels",
            "Instagram growth {niche} content formats trending"
        ]

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
            # Use the new genai client for text generation
            model = self.veo_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            response_text = model.text

            json_string = response_text.strip()
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

    def generate_ready_post(self, inspiration: InspirationPost, niche: str) -> GeneratedPost:
        try:
            # Create optimized prompt for vertical video using Veo 3
            video_prompt = f"""A high-quality 8-second Instagram Reel in 9:16 vertical format.

{inspiration.imageDescription[:250]}

The video should be {niche}-focused, professional, aesthetic, and Instagram-ready with dynamic camera movement, smooth transitions, and cinematic quality. Energetic, viral-worthy, and attention-grabbing mood. High-resolution with vibrant colors and sharp details optimized for mobile viewing."""

            print(f"🎬 Generating {niche} video using Google Veo 3...")
            print(f"📝 Prompt: {video_prompt[:150]}...")

            # Generate video using Veo 3
            print("🎥 Calling Veo 3 model...")
            
            operation = self.veo_client.models.generate_videos(
                model="veo-3.0-generate-001",
                prompt=video_prompt,
            )
            
            # Poll the operation status until the video is ready
            max_wait_time = 300  # 5 minutes timeout
            start_time = time.time()
            poll_count = 0
            
            while not operation.done:
                if time.time() - start_time > max_wait_time:
                    raise Exception("Video generation timed out after 5 minutes")
                
                poll_count += 1
                print(f"⏳ Waiting for video generation to complete... (poll #{poll_count})")
                time.sleep(10)
                operation = self.veo_client.operations.get(operation)
            
            print("✅ Video generation completed!")
            
            # Download the generated video
            generated_video = operation.response.generated_videos[0]
            
            # Create temp file to save video
            temp_dir = tempfile.mkdtemp()
            temp_video_path = os.path.join(temp_dir, f"{niche}_temp_{int(time.time())}.mp4")
            
            # Download and save video
            print("📥 Downloading generated video...")
            self.veo_client.files.download(file=generated_video.video)
            generated_video.video.save(temp_video_path)
            print(f"✅ Video saved temporarily to {temp_video_path}")
            
            # Upload to Cloudinary for permanent hosting
            filename = f"{niche}_reel_{int(time.time())}"
            final_video_url = self.cloudinary_uploader.upload_video_from_file(temp_video_path, filename)
            print(f"✅ Video uploaded to Cloudinary: {final_video_url}")
            
            # Clean up temp file
            try:
                os.remove(temp_video_path)
                os.rmdir(temp_dir)
            except:
                pass

            # Generate caption and hashtags using Gemini
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

            text_response = self.veo_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=text_prompt
            )

            json_string = text_response.text.strip()
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
                video_url=final_video_url,
                caption=text_content.get('caption', f'Amazing {niche} content!'),
                hashtags=text_content.get('hashtags', [f'#{niche}', '#reels', '#viralreels', '#trending'])
            )

        except Exception as error:
            print(f"Error generating ready post: {error}")
            raise Exception(f"Failed to generate post content for Reel: {str(error)}")
    
    def _generate_from_images_fallback(self, inspiration: InspirationPost, niche: str) -> GeneratedPost:
        """Fallback method: Generate video from multiple AI images if Veo fails"""
        print("📸 Generating image-based video as fallback...")
        raise Exception("Veo 3 video generation is not available yet. Please use image-based generation or wait for API access.")

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
            container_response = requests.post(create_url, data=create_params, timeout=60) 
            container_data = container_response.json()

            if container_response.status_code != 200 or 'id' not in container_data:
                error_msg = container_data.get('error', {}).get('message', 'Failed to create media container for Reel')
                raise Exception(error_msg)

            creation_id = container_data['id']
            print(f"Media container created: {creation_id}")

            # Poll for video processing
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

                time.sleep(10)

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
    print("🤖 Instagram REEL Generator with Google Veo 3 Starting...")

    google_api_key = os.getenv('GOOGLE_API_KEY')
    cloudinary_cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    cloudinary_upload_preset = os.getenv('CLOUDINARY_UPLOAD_PRESET')
    cloudinary_api_key = os.getenv('CLOUDINARY_API_KEY')
    cloudinary_api_secret = os.getenv('CLOUDINARY_API_SECRET')
    instagram_account_id = os.getenv('INSTAGRAM_ACCOUNT_ID')
    instagram_access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')

    required_vars = {
        'GOOGLE_API_KEY': google_api_key,
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
        return 

    try:
        cloudinary_uploader = CloudinaryVideoUploader(
            cloudinary_cloud_name, cloudinary_upload_preset, cloudinary_api_key, cloudinary_api_secret
        )

        history_manager = PostHistoryManager(
            cloudinary_cloud_name, cloudinary_upload_preset, cloudinary_api_key, cloudinary_api_secret
        )

        generator = InstagramPostGenerator(
            google_api_key, cloudinary_uploader
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
        print(f"🔗 Final Reel Video URL: {generated_post.video_url}")

        print("📱 Publishing REEL to Instagram...")
        publisher = InstagramPublisher()
        full_caption = f"{generated_post.caption}\n\n{' '.join(generated_post.hashtags)}"

        post_id = publisher.publish_post(
            account_id=instagram_account_id,
            access_token=instagram_access_token,
            media_url=generated_post.video_url,
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
        print(f"\n\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
