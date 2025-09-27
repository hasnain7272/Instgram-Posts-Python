import json
import time
import base64
import requests
import os
from typing import List
from dataclasses import dataclass
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


class InstagramPostGenerator:
    def __init__(self, google_api_key: str):
        genai.configure(api_key=google_api_key)
        self.google_api_key = google_api_key
    
    def fetch_inspiration_posts(self, location: str) -> List[InspirationPost]:
        prompt = f"""
        Using Google Search, find 5 recent and visually interesting Instagram posts that are trending in {location}.
        For each post, provide a detailed, vivid description of the image content (imageDescription), the original username, and the original caption.
        The imageDescription is the most important part; it should be descriptive enough for an AI image generator to create a new image.
        Do not include any URLs.
        Return the result as a valid JSON array of objects. Each object must have a unique "id" field (can be a random string), a "username" field, a "caption" field, and an "imageDescription" field.
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
            raise Exception("Failed to find inspiration. The AI may be busy or the API key may be invalid. Please try again.")
    
    def generate_ready_post(self, inspiration: InspirationPost) -> GeneratedPost:
        try:
            image_prompt = f'A high-quality, realistic photograph inspired by: "{inspiration.imageDescription}". Professional, clean, and suitable for Instagram.'
            
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token:
                raise Exception("HUGGINGFACE_TOKEN environment variable is required for image generation")
            
            hf_headers = {"Authorization": f"Bearer {hf_token}"}
            hf_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            
            print("🎨 Generating image with HuggingFace FLUX...")
            response = requests.post(hf_url, headers=hf_headers, json={"inputs": image_prompt}, timeout=60)
            if response.status_code == 200:
                base64_image = base64.b64encode(response.content).decode('utf-8')
                print("✅ Image generated successfully")
            else:
                raise Exception(f"HuggingFace image generation failed: {response.status_code} - {response.text}")
            
            text_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            text_prompt = f"""
            You are an expert Instagram content creator.
            Based on the inspiration from this caption: "{inspiration.caption}", and this image description: "{inspiration.imageDescription}", create a brand new, engaging post.
            Generate a short, catchy caption (max 100 characters).
            Generate a list of 7 relevant and effective hashtags.
            
            Return as JSON with this exact structure:
            {{
                "caption": "your caption here",
                "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5", "#tag6", "#tag7"]
            }}
            """
            
            text_response = text_model.generate_content(
                text_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500
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
                caption=text_content.get('caption', 'Check out this amazing view!'),
                hashtags=text_content.get('hashtags', ['#instagram', '#photography', '#amazing', '#beautiful', '#instagood', '#photooftheday', '#nature'])
            )
            
        except Exception as error:
            print(f"Error generating ready post: {error}")
            raise Exception("Failed to generate the post. The AI may be experiencing high traffic or the API key may be invalid. Please try again.")


class CloudinaryUploader:
    @staticmethod
    def upload_image(base64_image: str, cloud_name: str, upload_preset: str) -> str:
        url = f"https://api.cloudinary.com/v1_1/{cloud_name}/image/upload"
        
        file_data = f"data:image/png;base64,{base64_image}"
        
        form_data = {
            'file': file_data,
            'upload_preset': upload_preset
        }
        
        try:
            response = requests.post(url, data=form_data, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'secure_url' not in data:
                error_msg = data.get('error', {}).get('message', 'Cloudinary upload failed. Check your Cloud Name and Upload Preset.')
                raise Exception(error_msg)
            
            return data['secure_url']
            
        except requests.exceptions.RequestException as error:
            print(f"Error uploading to Cloudinary: {error}")
            raise Exception(f"Cloudinary Error: {str(error)}")
        except Exception as error:
            print(f"Error uploading to Cloudinary: {error}")
            raise Exception("An unknown error occurred during the Cloudinary upload.")


class InstagramPublisher:
    def __init__(self):
        self.api_version = 'v20.0'
        self.base_url = f'https://graph.facebook.com/{self.api_version}'
    
    def publish_post(self, account_id: str, access_token: str, image_url: str, caption: str) -> None:
        try:
            create_url = f"{self.base_url}/{account_id}/media"
            create_params = {
                'image_url': image_url,
                'caption': caption,
                'access_token': access_token
            }
            
            print(f"Creating media container for account {account_id}")
            container_response = requests.post(create_url, data=create_params, timeout=30)
            container_response.raise_for_status()
            container_data = container_response.json()
            
            if 'id' not in container_data:
                error_msg = container_data.get('error', {}).get('message', 'Failed to create media container.')
                raise Exception(error_msg)
            
            creation_id = container_data['id']
            print(f"Media container created: {creation_id}")
            
            max_retries = 10
            retry_delay = 3
            
            print("Waiting for media to process...")
            for i in range(max_retries):
                status_url = f"https://graph.facebook.com/{creation_id}"
                status_params = {
                    'fields': 'status_code',
                    'access_token': access_token
                }
                
                status_response = requests.get(status_url, params=status_params, timeout=10)
                status_response.raise_for_status()
                status_data = status_response.json()
                
                status_code = status_data.get('status_code')
                print(f"Media status: {status_code}")
                
                if status_code == 'FINISHED':
                    print("Media processing complete!")
                    break
                elif status_code in ['ERROR', 'EXPIRED']:
                    raise Exception(f"Media processing failed on Instagram with status: {status_code}.")
                
                if i == max_retries - 1:
                    raise Exception("Media is taking too long to process on Instagram's servers. Please try again in a moment.")
                
                time.sleep(retry_delay)
            
            print("Publishing post...")
            publish_url = f"{self.base_url}/{account_id}/media_publish"
            publish_params = {
                'creation_id': creation_id,
                'access_token': access_token
            }
            
            publish_response = requests.post(publish_url, data=publish_params, timeout=30)
            publish_response.raise_for_status()
            publish_data = publish_response.json()
            
            if 'id' not in publish_data:
                error_msg = publish_data.get('error', {}).get('message', 'Failed to publish the media.')
                raise Exception(error_msg)
            
            print(f"Post published successfully! Post ID: {publish_data['id']}")
                
        except requests.exceptions.RequestException as error:
            print(f"Error publishing to Instagram: {error}")
            raise Exception(f"Publishing failed: {str(error)}")
        except Exception as error:
            print(f"Error publishing to Instagram: {error}")
            raise Exception(f"Publishing failed: {str(error)}")


def main():
    print("🤖 Instagram Post Generator Starting...")
    
    google_api_key = os.getenv('GOOGLE_API_KEY')
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
    cloudinary_cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    cloudinary_upload_preset = os.getenv('CLOUDINARY_UPLOAD_PRESET')
    instagram_account_id = os.getenv('INSTAGRAM_ACCOUNT_ID')
    instagram_access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')
    location = os.getenv('LOCATION', 'New York')
    
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
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return
    
    try:
        print(f"📍 Searching for inspiration in {location}...")
        generator = InstagramPostGenerator(google_api_key)
        
        inspiration_posts = generator.fetch_inspiration_posts(location)
        print(f"✅ Found {len(inspiration_posts)} inspiration posts")
        
        if not inspiration_posts:
            print("❌ No inspiration posts found")
            return
        
        print("🎨 Generating new post content...")
        generated_post = generator.generate_ready_post(inspiration_posts[0])
        print(f"✅ Generated caption: {generated_post.caption}")
        print(f"✅ Generated hashtags: {' '.join(generated_post.hashtags)}")
        
        if not generated_post.base64Image:
            raise Exception("Failed to generate image content")
        
        print("☁️ Uploading image to Cloudinary...")
        uploader = CloudinaryUploader()
        image_url = uploader.upload_image(
            base64_image=generated_post.base64Image,
            cloud_name=cloudinary_cloud_name,
            upload_preset=cloudinary_upload_preset
        )
        print(f"✅ Image uploaded: {image_url}")
        
        print("📱 Publishing to Instagram...")
        publisher = InstagramPublisher()
        
        full_caption = f"{generated_post.caption}\n\n{' '.join(generated_post.hashtags)}"
        
        publisher.publish_post(
            account_id=instagram_account_id,
            access_token=instagram_access_token,
            image_url=image_url,
            caption=full_caption
        )
        
        print("🎉 Post published successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
