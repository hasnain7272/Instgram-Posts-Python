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


@dataclass
class InspirationPost:
    id: str
    username: str
    caption: str
    imageDescription: str


@dataclass
class GeneratedPost:
    base64Image: str
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


class PostHistoryManager:
    def __init__(self, cloudinary_cloud_name: str, cloudinary_upload_preset: str):
        self.cloud_name = cloudinary_cloud_name
        self.upload_preset = cloudinary_upload_preset
        self.history_file = "post_history.json"

    def download_history(self) -> List[PostMetadata]:
        """Download post history from Cloudinary, using cache busting."""
        try:
            import time
            history_url = f"https://res.cloudinary.com/{self.cloud_name}/raw/upload/{self.history_file}?t={int(time.time())}"
            print(f"[DEBUG] Downloading post history from {history_url}")
            response = requests.get(history_url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                posts = data.get('posts', [])
                print(f"[DEBUG] Downloaded {len(posts)} posts from history.")
                return [PostMetadata(**post) for post in posts]
            else:
                print(f"[DEBUG] No existing post history found (status: {response.status_code}), starting fresh.")
                return []
        except Exception as e:
            print(f"[ERROR] Error downloading history: {e}")
            return []

    def upload_history(self, history: List[PostMetadata]) -> None:
        """Upload updated post history to Cloudinary, force overwrite."""
        try:
            history_data = {
                'posts': [asdict(post) for post in history],
                'updated_at': datetime.now().isoformat()
            }
            json_string = json.dumps(history_data, indent=2)
            url = f"https://api.cloudinary.com/v1_1/{self.cloud_name}/raw/upload"
            files = {'file': ('post_history.json', json_string, 'application/json')}
            data = {
                'upload_preset': self.upload_preset,
                'public_id': 'post_history.json',
                'resource_type': 'raw'
            }
            print(f"[DEBUG] Uploading post history ({len(history)} posts) to Cloudinary...")
            response = requests.post(url, files=files, data=data, timeout=30)
            if response.status_code != 200:
                print(f"[ERROR] Failed to upload history: {response.text}")
                raise Exception("Failed to upload history to Cloudinary")
            print("[DEBUG] History uploaded successfully.")
        except Exception as e:
            print(f"[ERROR] Error uploading history: {e}")
            raise
    
    def is_duplicate_content(self, new_description: str, new_keywords: List[str], 
                           new_hashtags: List[str], history: List[PostMetadata]) -> bool:
        """Check if content is too similar to recent posts"""
        new_desc_hash = hashlib.md5(new_description.lower().encode()).hexdigest()
        recent_posts = [p for p in history if self._is_recent(p.timestamp, days=7)]
        
        for post in recent_posts:
            # Check image description similarity
            if post.image_description_hash == new_desc_hash:
                return True
            
            # Check keyword overlap
            common_keywords = set(new_keywords) & set(post.caption_keywords)
            if len(common_keywords) / max(len(new_keywords), 1) > 0.6:
                return True
            
            # Check hashtag overlap
            common_hashtags = set(new_hashtags) & set(post.hashtags_used)
            if len(common_hashtags) >= 5:  # Too many common hashtags
                return True
                
        return False
    
    def get_next_niche(self, history: List[PostMetadata]) -> str:
        """Get next niche to avoid repetition"""
        niches = ['fitness', 'food', 'travel', 'lifestyle', 'motivation', 'aesthetic', 'fashion', 'tech', 'hot Sensual indulgence']
        recent_niches = [p.engagement_niche for p in history[-5:]]  # Last 5 posts
        
        for niche in niches:
            if niche not in recent_niches:
                return niche
        
        return niches[0]  # Fallback
    
    def _is_recent(self, timestamp_str: str, days: int) -> bool:
        """Check if timestamp is within recent days"""
        try:
            post_time = datetime.fromisoformat(timestamp_str)
            return datetime.now() - post_time < timedelta(days=days)
        except:
            return False


class InstagramPostGenerator:
    def __init__(self, google_api_key: str):
        genai.configure(api_key=google_api_key)
        self.google_api_key = google_api_key
        self.search_templates = [
            "viral Instagram posts {niche} high engagement 2025 Current",
            "trending {niche} content Instagram getting most likes",
            "popular Instagram {niche} aesthetics viral posts",
            "{season} Instagram trends {niche} viral content",
            "Instagram growth {niche} content formats trending"
        ]
    
    def fetch_inspiration_posts(self, niche: str, season: str = None) -> List[InspirationPost]:
        """Fetch inspiration posts with enhanced search strategy"""
        if not season:
            season = self._get_current_season()
            
        search_queries = []
        for template in self.search_templates:
            query = template.format(niche=niche, season=season)
            search_queries.append(query)
        
        prompt = f"""
        Using Google Search, find 5 recent and visually appealing Instagram posts using these search queries:
        {', '.join(search_queries[:3])}
        
        Focus on posts with high engagement potential in the {niche} niche.
        For each post, provide:
        - A detailed, vivid description of the image content (imageDescription) 
        - The original username
        - The original caption
        
        The imageDescription should be detailed enough for AI image generation.
        Return as valid JSON array with objects containing: id, username, caption, imageDescription fields.
        """
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
            
            json_string = response.text.strip()
            
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
        """Generate post content with niche-specific optimization"""
        try:
            image_prompt = f'High-quality, professional {niche} Instagram photo inspired by: "{inspiration.imageDescription}". Aesthetic, engaging, and optimized for social media.'
            
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token:
                raise Exception("HUGGINGFACE_TOKEN environment variable is required")
            
            hf_headers = {"Authorization": f"Bearer {hf_token}"}
            hf_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            
            print(f"🎨 Generating {niche} image with HuggingFace FLUX...")
            response = requests.post(hf_url, headers=hf_headers, json={"inputs": image_prompt}, timeout=60)
            
            if response.status_code == 200:
                base64_image = base64.b64encode(response.content).decode('utf-8')
                print("✅ Image generated successfully")
            else:
                raise Exception(f"Image generation failed: {response.status_code} - {response.text}")
            
            text_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            text_prompt = f"""
            Create engaging Instagram content for the {niche} niche.
            Inspiration caption: "{inspiration.caption}"
            Image description: "{inspiration.imageDescription}"
            
            Generate:
            1. A catchy caption (max 100 characters) that drives engagement
            2. 7 trending hashtags for {niche} content
            
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
                base64Image=base64_image,
                caption=text_content.get('caption', f'Amazing {niche} content!'),
                hashtags=text_content.get('hashtags', [f'#{niche}', '#instagram', '#viral', '#trending', '#aesthetic', '#instagood', '#photooftheday'])
            )
            
        except Exception as error:
            print(f"Error generating ready post: {error}")
            raise Exception("Failed to generate post content. Please try again.")
    
    def _get_current_season(self) -> str:
        """Get current season for seasonal content"""
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
        """Extract keywords from caption for duplicate checking"""
        words = caption.lower().replace('#', '').replace('@', '').split()
        keywords = [word for word in words if len(word) > 3 and word.isalpha()]
        return keywords[:10]  # Top 10 keywords


class CloudinaryUploader:
    @staticmethod
    def upload_image(base64_image: str, cloud_name: str, upload_preset: str) -> str:
        """Upload image to Cloudinary"""
        url = f"https://api.cloudinary.com/v1_1/{cloud_name}/image/upload"
        
        files = {'file': f"data:image/jpeg;base64,{base64_image}"}
        data = {'upload_preset': upload_preset}
        
        try:
            response = requests.post(url, files=files, data=data, timeout=30)
            data = response.json()
            
            if response.status_code != 200 or 'secure_url' not in data:
                error_msg = data.get('error', {}).get('message', f'Upload failed: {response.status_code}')
                raise Exception(error_msg)
            
            return data['secure_url']
            
        except Exception as error:
            print(f"Error uploading to Cloudinary: {error}")
            raise Exception(f"Cloudinary upload failed: {str(error)}")


class InstagramPublisher:
    def __init__(self):
        self.api_version = 'v20.0'
        self.base_url = f'https://graph.facebook.com/{self.api_version}'
    
    def publish_post(self, account_id: str, access_token: str, image_url: str, caption: str) -> str:
        """Publish post to Instagram and return post ID"""
        try:
            create_url = f"{self.base_url}/{account_id}/media"
            create_params = {
                'image_url': image_url,
                'caption': caption,
                'access_token': access_token
            }
            
            print(f"Creating media container...")
            container_response = requests.post(create_url, data=create_params, timeout=30)
            container_data = container_response.json()
            
            if container_response.status_code != 200 or 'id' not in container_data:
                error_msg = container_data.get('error', {}).get('message', 'Failed to create media container')
                raise Exception(error_msg)
            
            creation_id = container_data['id']
            print(f"Media container created: {creation_id}")
            
            # Wait for processing
            max_retries = 10
            for i in range(max_retries):
                status_url = f"https://graph.facebook.com/{creation_id}"
                status_params = {'fields': 'status_code', 'access_token': access_token}
                
                status_response = requests.get(status_url, params=status_params, timeout=10)
                status_data = status_response.json()
                
                status_code = status_data.get('status_code')
                print(f"Media status: {status_code}")
                
                if status_code == 'FINISHED':
                    break
                elif status_code in ['ERROR', 'EXPIRED']:
                    raise Exception(f"Media processing failed: {status_code}")
                
                if i == max_retries - 1:
                    raise Exception("Media processing timeout")
                
                time.sleep(3)
            
            # Publish
            print("Publishing post...")
            publish_url = f"{self.base_url}/{account_id}/media_publish"
            publish_params = {'creation_id': creation_id, 'access_token': access_token}
            
            publish_response = requests.post(publish_url, data=publish_params, timeout=30)
            publish_data = publish_response.json()
            
            if publish_response.status_code != 200 or 'id' not in publish_data:
                error_msg = publish_data.get('error', {}).get('message', 'Failed to publish')
                raise Exception(error_msg)
            
            post_id = publish_data['id']
            print(f"✅ Published successfully! Post ID: {post_id}")
            return post_id
                
        except Exception as error:
            print(f"Error publishing: {error}")
            raise Exception(f"Publishing failed: {str(error)}")


def main():
    print("🤖 Enhanced Instagram Post Generator Starting...")
    
    # Environment variables
    google_api_key = os.getenv('GOOGLE_API_KEY')
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
    cloudinary_cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    cloudinary_upload_preset = os.getenv('CLOUDINARY_UPLOAD_PRESET')
    instagram_account_id = os.getenv('INSTAGRAM_ACCOUNT_ID')
    instagram_access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')
    
    required_vars = {
        'GOOGLE_API_KEY': google_api_key,
        'HUGGINGFACE_TOKEN': huggingface_token,
        'CLOUDINARY_CLOUD_NAME': cloudinary_cloud_name,
        'CLOUDINARY_UPLOAD_PRESET': cloudinary_upload_preset,
        'INSTAGRAM_ACCOUNT_ID': instagram_account_id,
        'INSTAGRAM_ACCESS_TOKEN': instagram_access_token
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return
    
    try:
        # Initialize managers
        history_manager = PostHistoryManager(cloudinary_cloud_name, cloudinary_upload_preset)
        generator = InstagramPostGenerator(google_api_key)
        
        # Get post history
        print("📥 Loading post history...")
        post_history = history_manager.download_history()
        print(f"Found {len(post_history)} previous posts")
        
        # Determine next niche
        next_niche = history_manager.get_next_niche(post_history)
        print(f"🎯 Target niche: {next_niche}")
        
        # Generate content with retry for duplicates
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"🔍 Fetching {next_niche} inspiration (attempt {attempt + 1})...")
            inspiration_posts = generator.fetch_inspiration_posts(next_niche)
            
            if not inspiration_posts:
                if attempt < max_attempts - 1:
                    continue
                raise Exception("No inspiration posts found")
            
            print("🎨 Generating post content...")
            generated_post = generator.generate_ready_post(inspiration_posts[0], next_niche)
            
            # Check for duplicates
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
        
        print(f"📝 Caption: {generated_post.caption}")
        print(f"🏷️ Hashtags: {' '.join(generated_post.hashtags)}")
        
        # Upload image
        print("☁️ Uploading to Cloudinary...")
        uploader = CloudinaryUploader()
        image_url = uploader.upload_image(
            base64_image=generated_post.base64Image,
            cloud_name=cloudinary_cloud_name,
            upload_preset=cloudinary_upload_preset
        )
        print(f"✅ Image uploaded: {image_url}")
        
        # Publish to Instagram
        print("📱 Publishing to Instagram...")
        # publisher = InstagramPublisher()
        # full_caption = f"{generated_post.caption}\n\n{' '.join(generated_post.hashtags)}"
        
        #post_id=publisher.publish_post(account_id=instagram_account_id,access_token=instagram_access_token,image_url=image_url,caption=full_caption)
        
        # Update history
        print("💾 Updating post history...")
        new_metadata = PostMetadata(
            id='1234',
            timestamp=datetime.now().isoformat(),
            inspiration_source=f"{next_niche} viral content",
            image_description_hash=hashlib.md5(inspiration_posts[0].imageDescription.lower().encode()).hexdigest(),
            caption_keywords=keywords,
            hashtags_used=hashtag_list,
            engagement_niche=next_niche
        )
        
        post_history.append(new_metadata)
        history_manager.upload_history(post_history)
        
        print("🎉 Post published and history updated successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
