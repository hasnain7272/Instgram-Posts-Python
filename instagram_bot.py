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
    def __init__(self, cloudinary_cloud_name: str, cloudinary_upload_preset: str, api_key: str, api_secret: str):
        self.cloud_name = cloudinary_cloud_name
        self.upload_preset = cloudinary_upload_preset
        self.history_file = "post_history.json"
        self.api_key = api_key
        self.api_secret = api_secret
        
    def generate_signature(self, params: dict) -> str:
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
        niches = ['fitness', 'motivation', 'food', 'travel', 'lifestyle', 'aesthetic', 'fashion', 'tech', 'hot Sensual indulgence']
        recent_niches = [p.engagement_niche for p in history[-8:]]
        
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

class ImageGenerator:
    """Handles image generation with multiple fallback options"""
    
    def generate_image(self, prompt: str, niche: str) -> str:
        """
        Try multiple image generation methods in order:
        1. HuggingFace Inference API
        2. Pollinations.ai (free, no API key)
        """
        
        # Method 1: HuggingFace
        try:
            return self._generate_huggingface(prompt)
        except Exception as e:
            print(f"⚠️ HuggingFace failed: {e}")
        
        # Method 2: Pollinations.ai (Free, no limits)
        try:
            return self._generate_pollinations(prompt)
        except Exception as e:
            print(f"⚠️ Pollinations.ai failed: {e}")

    def _generate_huggingface(self, image_prompt) -> str:
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            raise Exception("HUGGINGFACE_TOKEN environment variable is required")
        
        hf_headers = {"Authorization": f"Bearer {hf_token}"}
        hf_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        
        print(f"🎨 Generating {image_prompt} image with HuggingFace FLUX...")
        response = requests.post(hf_url, headers=hf_headers, json={"inputs": image_prompt}, timeout=60)
        
        if response.status_code == 200:
            base64_image = base64.b64encode(response.content).decode('utf-8')
            print("✅ Image generated successfully")
            return base64_image
        else:
            raise Exception(f"Image generation failed: {response.status_code} - {response.text}")

    def _generate_pollinations(self, prompt: str) -> str:
        """
        Use Pollinations.ai - Free image generation, no API key needed
        This is a reliable fallback option
        """
        print("🎨 Generating with Pollinations.ai (Free)...")
        
        from urllib.parse import quote
        # URL-encode the prompt
        encoded_prompt = quote(prompt)
        
        # Pollinations.ai free API - supports various models
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&model=flux"
        
        try:
            response = requests.get(image_url, timeout=60)
            if response.status_code == 200:
                base64_image = base64.b64encode(response.content).decode('utf-8')
                print("✅ Generated with Pollinations.ai")
                return base64_image
            else:
                raise Exception(f"Status code: {response.status_code}")
        except Exception as e:
            raise Exception(f"Pollinations.ai failed: {e}")
    
        
class InstagramPostGenerator:
    def __init__(self, google_api_key: str):
        genai.configure(api_key=google_api_key)
        self.google_api_key = google_api_key
        self.image_generator = ImageGenerator()
        self.search_templates = [
            "viral Instagram posts {niche} high engagement 2025 Current",
            "trending {niche} content Instagram getting most likes",
            "{season} Instagram trends {niche} viral content",
            "popular Instagram {niche} aesthetics viral posts",
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
        Using Google Search, find 5 recent and visually appealing Instagram posts using these search queries:
        {', '.join(search_queries[:3])}
        
        Focus on posts with high engagement potential in the {niche} niche.
        For each post, provide:
        - A detailed, vivid description of the image content (imageDescription) 
        - The original username
        - The original caption
        
        The imageDescription should be long and detailed enough for AI image generation.
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
        try:
            image_prompt = f'High-quality, professional {niche} Instagram photo inspired by: "{inspiration.imageDescription}". Aesthetic, engaging, and optimized for social media. Ultra-detailed, vibrant colors, professional photography.'
            
            # Use the new image generator with multiple fallbacks
            print(f"🎨 Generating {niche} image...")
            base64_image = self.image_generator.generate_image(image_prompt, niche)
            print("✅ Image generated successfully")
            
            # hf_token = os.getenv('HUGGINGFACE_TOKEN')
            # if not hf_token:
            #     raise Exception("HUGGINGFACE_TOKEN environment variable is required")
            
            # hf_headers = {"Authorization": f"Bearer {hf_token}"}
            # hf_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            
            # print(f"🎨 Generating {niche} image with HuggingFace FLUX...")
            # response = requests.post(hf_url, headers=hf_headers, json={"inputs": image_prompt}, timeout=60)
            
            # if response.status_code == 200:
            #     base64_image = base64.b64encode(response.content).decode('utf-8')
            #     print("✅ Image generated successfully")
            # else:
            #     raise Exception(f"Image generation failed: {response.status_code} - {response.text}")
            
            text_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            text_prompt = f"""
            Create engaging Instagram content for the {niche} niche.
            Inspiration caption: "{inspiration.caption}"
            Image description: "{inspiration.imageDescription}"
            
            Generate:
            1. A catchy caption (max 100 characters) that drives engagement
            2. 20 trending hashtags for {niche} content
            
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


class CloudinaryUploader:
    @staticmethod
    def upload_image(base64_image: str, cloud_name: str, upload_preset: str, api_key: str, api_secret: str) -> str:
        import hashlib
        
        timestamp = int(time.time())
        params = {
            'api_key': api_key,
            'timestamp': timestamp,
            'upload_preset': upload_preset,
        }
        
        excluded_params = {'file', 'cloud_name', 'resource_type', 'api_key'}
        filtered_params = {k: v for k, v in params.items() if k not in excluded_params}
        sorted_params = sorted(filtered_params.items())
        string_to_sign = '&'.join(f"{k}={v}" for k, v in sorted_params)
        string_to_sign += api_secret
        signature = hashlib.sha1(string_to_sign.encode('utf-8')).hexdigest()
        
        url = f"https://api.cloudinary.com/v1_1/{cloud_name}/image/upload"
        
        files = {'file': f"data:image/jpeg;base64,{base64_image}"}
        data = {
            'api_key': api_key,
            'signature': signature,
            'timestamp': timestamp,
            'upload_preset': upload_preset,
        }
        
        try:
            response = requests.post(url, files=files, data=data, timeout=30)
            response_data = response.json()
            
            if response.status_code != 200 or 'secure_url' not in response_data:
                error_msg = response_data.get('error', {}).get('message', f'Upload failed: {response.status_code}')
                raise Exception(error_msg)
            
            return response_data['secure_url']
            
        except Exception as error:
            print(f"Error uploading to Cloudinary: {error}")
            raise Exception(f"Cloudinary upload failed: {str(error)}")


class InstagramPublisher:
    def __init__(self):
        self.api_version = 'v20.0'
        self.base_url = f'https://graph.facebook.com/{self.api_version}'
    
    def publish_post(self, account_id: str, access_token: str, image_url: str, caption: str) -> str:
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
    
    google_api_key = os.getenv('GOOGLE_API_KEY')
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
    cloudinary_cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    cloudinary_upload_preset = os.getenv('CLOUDINARY_UPLOAD_PRESET')
    cloudinary_api_key = os.getenv('CLOUDINARY_API_KEY')
    cloudinary_api_secret = os.getenv('CLOUDINARY_API_SECRET')
    instagram_account_id = os.getenv('INSTAGRAM_ACCOUNT_ID')
    instagram_access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')
    
    required_vars = {
        'GOOGLE_API_KEY': google_api_key,
        'HUGGINGFACE_TOKEN': huggingface_token,
        'CLOUDINARY_CLOUD_NAME': cloudinary_cloud_name,
        'CLOUDINARY_UPLOAD_PRESET': cloudinary_upload_preset,
        'CLOUDINARY_API_KEY': cloudinary_api_key,
        'CLOUDINARY_API_SECRET': cloudinary_api_secret,
        'INSTAGRAM_ACCOUNT_ID': instagram_account_id,
        'INSTAGRAM_ACCESS_TOKEN': instagram_access_token
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return
    
    try:
        history_manager = PostHistoryManager(cloudinary_cloud_name, cloudinary_upload_preset, cloudinary_api_key, cloudinary_api_secret)
        generator = InstagramPostGenerator(google_api_key)
        
        print("📥 Loading post history...")
        post_history = history_manager.download_history()
        print(f"Found {len(post_history)} previous posts")
        
        next_niche = history_manager.get_next_niche(post_history)
        print(f"🎯 Target niche: {next_niche}")
        
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
        
        print("☁️ Uploading to Cloudinary...")
        uploader = CloudinaryUploader()
        image_url = uploader.upload_image(
            base64_image=generated_post.base64Image,
            cloud_name=cloudinary_cloud_name,
            upload_preset=cloudinary_upload_preset,
            api_key=cloudinary_api_key,
            api_secret=cloudinary_api_secret
        )
        print(f"✅ Image uploaded: {image_url}")
        
        print("📱 Publishing to Instagram...")
        publisher = InstagramPublisher()
        full_caption = f"{generated_post.caption}\n\n{' '.join(generated_post.hashtags)}"
        
        post_id = publisher.publish_post(
            account_id=instagram_account_id,
            access_token=instagram_access_token,
            image_url=image_url,
            caption=full_caption
        )
        
        print("💾 Updating post history...")
        new_metadata = PostMetadata(
            id=post_id,
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
