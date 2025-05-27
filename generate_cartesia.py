import os
import asyncio
from cartesia import AsyncCartesia
import time

# Set up API credentials
API_KEY = "sk_car_y4i5pnWSWP7HfgpyD7sfoM"  # Replace with your actual Cartesia API key

# List of voice IDs
voice_ids = [
    "a0e99841-438c-4a64-b679-ae501e7d6091",
    "694f9389-aac1-45b6-b726-9d9369183238",
    "5c29d7e3-a133-4c7e-804a-1d9c6dea83f6",
    "79743797-2087-422f-8dc7-86f9efca85f1",
    "c99d36f3-5ffd-4253-803a-535c1bc9c306",
    "c8605446-247c-4d39-acd4-8f4c28aa363c"
]

# List of monosyllables to synthesize
monosyllables = ["sí", "no", "ok", "va", "ya", "lo", "me", "te", "di", "ve", "sol", "mar", "luz"]

# Output directory
output_dir = "cartesia_synthesized_voices"
os.makedirs(output_dir, exist_ok=True)

# Initialize async Cartesia client
client = AsyncCartesia(api_key=API_KEY)

async def generate_and_save_audio(voice_id: str, text: str, output_dir: str) -> None:
    """Generate speech for a specific voice and save it as an MP3 file."""
    try:
        # Create unique filename using voice ID, text, and timestamp
        output_file = os.path.join(output_dir, f"{text}_cartesia_{voice_id}_{int(time.time())}.mp3")
        
        # Open file in binary write mode
        with open(output_file, "wb") as f:
            # Stream audio chunks asynchronously
            async for chunk in client.tts.bytes(
                model_id="sonic-2",
                transcript=text,
                voice={"id": voice_id},
                language="en",
                output_format={
                    "container": "mp3",
                    "sample_rate": 44100,
                    "encoding": "mp3"
                },
            ):
                if chunk:  # Ensure chunk is not empty
                    f.write(chunk)
        
        print(f"Successfully saved {output_file}")
    
    except Exception as e:
        print(f"Error generating audio for {voice_id} and text '{text}': {str(e)}")

async def main() -> None:
    """Main function to process all voice IDs and monosyllables."""
    # Generate audio for each monosyllable and voice ID
    for text in monosyllables:
        for voice_id in voice_ids:
            await generate_and_save_audio(voice_id, text, output_dir)
    
    print("Audio generation for monosyllables complete!")
    await client.close()  # Clean up the client connection

# Run the application
if __name__ == "__main__":
    asyncio.run(main())