import os
from mutagen.easyid3 import EasyID3

def rename_mp3_files_in_directory(directory_path):
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    # Loop over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.mp3'):
            file_path = os.path.join(directory_path, filename)
            try:
                # Load MP3 metadata
                audio = EasyID3(file_path)
                
                # Extract metadata
                title = audio.get('title', ['Unknown Title'])[0]
                artist = audio.get('artist', ['Unknown Artist'])[0]
                album = audio.get('album', ['Unknown Album'])[0]
                
                # Construct new file name
                new_name = f"{title} - {artist} ({album}).mp3"
                
                # Remove invalid characters from file name
                invalid_chars = r'<>:"/\|?*'
                new_name = ''.join(c for c in new_name if c not in invalid_chars)
                
                # Full path for the new file name
                new_path = os.path.join(directory_path, new_name)
                
                # Rename the file
                os.rename(file_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

# Set the correct path to the directory
directory_path = "Dataset/DLMusicTest_30"

# Process the directory
rename_mp3_files_in_directory(directory_path)
