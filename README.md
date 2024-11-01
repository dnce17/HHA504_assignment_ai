# Exploring AI and Analytics with Pre-trained Models in Azure and GCP

## 1. Work with Pre-trained Speech Models
### GCP Speech-to-Text
#### Create a Bucket in Cloud Storage
1. Click "Buckets" in the GCP search bar
![Buckets link in search bar](img/gcp/buckets_link.png)
2. Create a bucket with the following configuration
    * Location type: Region
3. Click the bucket after it's made
4. Upload an mp3 file
5. Click "Permission," then click "Grant Access" and apply these configurations:
    * New principals: allUsers
    * Role: Storage Object Viewer

NOTE: The permissions above makes the bucket public, which will allow the Vertex AI notebook to access it later on

#### Speech-to-Text in Vertex AI Notebook
NOTE: If you run into errors that mention needing to enable certain APIs, make sure to enable them to resolve the issue.
1. Click "Vertex AI" in the GCP search bar
2. Hover to the left pane and click "Colab Enterprise"
![Colab Enterprise link](img/gcp/colab_enterprise_link.png)
3. Under "Sample notebooks," click "Getting started with Gemini 1.5 Flash"
    * NOTE: These sample notebooks are pre-configured
4. Run the default codes under "Install Vertex AI SDK for Python" to "Load the Gemini 1.5 Flash model"
5. Under "Audio understanding," the code was adjusted as followed:
* DEFAULT CODE
```python
audio_file_path = "cloud-samples-data/generative-ai/audio/pixel.mp3"
audio_file_uri = f"gs://{audio_file_path}"
audio_file_url = f"https://storage.googleapis.com/{audio_file_path}"

IPython.display.Audio(audio_file_url)
```
* ADJUSTED CODE TO INCLUDE BUCKETS
    * Reason: Google Cloud Storage (GCS) files have paths that include the bucket name and the file path within that bucket. 
```python
# Define bucket name and file path
bucket_name = "transcription_use"  # Replace with your actual bucket name
file_name = "harvard.mp3"

# GCS URI and Public URL
audio_file_uri = f"gs://{bucket_name}/{file_name}"
audio_file_url = f"https://storage.googleapis.com/{bucket_name}/{file_name}"

IPython.display.Audio(audio_file_url)
```
6. Under "Example 2: Transcription," adjust the prompt to simply transcribe the audio content. Leave the remaining default code as is.
```python
prompt = """
    Can you transcribe the audio content?
"""
```
7. Results
![Result of speech to text](img/gcp/speech_to_text_output.png)

## Credits
1. [How to Set Buckets and Files Public In Google Cloud Storage](https://www.youtube.com/watch?v=3V8aDWRreFU)