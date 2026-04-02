import os
import json
import time

INPUT_DIR = r"C:\Users\helpf\Desktop\DD2477 Project\podcasts-no-audio-13GB" 
OUTPUT_FILE = r"C:\Users\helpf\Desktop\DD2477 Project\cleaned_output\cleaned_data.jsonl"

def clean_podcast_directory(input_folder, output_filepath):
    print(f"🚀 Starting cleanup of '{input_folder}'...")
    print(f"📁 Saving pure RAG data to '{output_filepath}'...\n")
    
    files_processed = 0
    chunks_saved = 0
    start_time = time.time()

    # Open the output file in 'w' (write) mode to start fresh
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        # os.walk automatically goes through every subdirectory for you!
        for root, dirs, files in os.walk(input_folder):
            for filename in files:
                # Ignore anything that isn't a JSON file
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(root, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        data = json.load(infile)
                        
                    # Loop through the results to find the transcript
                    for result in data.get("results", []):
                        try:
                            alt = result["alternatives"][0]
                            transcript = alt.get("transcript", "").strip()
                            words_array = alt.get("words", [])
                            
                            # Skip if the text or words are missing
                            if not transcript or not words_array:
                                continue
                                
                            # Grab start/end times and clean off the 's'
                            start_sec = float(words_array[0]["startTime"].replace('s', ''))
                            end_sec = float(words_array[-1]["endTime"].replace('s', ''))
                            
                            # Create our clean, lightweight dictionary
                            clean_chunk = {
                                "file_id": filename,
                                "text": transcript,
                                "start_time": start_sec,
                                "end_time": end_sec
                            }
                            
                            # Write this single chunk to the file immediately (keeps RAM empty!)
                            outfile.write(json.dumps(clean_chunk) + '\n')
                            chunks_saved += 1
                            
                        except (KeyError, IndexError, ValueError):
                            # Skip malformed blocks
                            continue
                            
                except Exception as e:
                    # If a file is completely broken, print the error but don't crash the script
                    print(f"⚠️ Error reading {filename}: {e}")
                    
                files_processed += 1
                
                # Print an update every 5,000 files so you know it's working
                if files_processed % 5000 == 0:
                    print(f"✅ Processed {files_processed} files... Saved {chunks_saved} chunks.")

    # Calculate how long it took
    elapsed_time = round((time.time() - start_time) / 60, 2)
    print("\n========================================")
    print(f"🎉 DONE! Processed {files_processed} files in {elapsed_time} minutes.")
    print(f"🔥 Total clean chunks saved: {chunks_saved}")
    print("========================================")

if __name__ == "__main__":
    # This automatically creates the 'cleaned_output' folder if it doesn't exist yet
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Run the cleaner
    clean_podcast_directory(INPUT_DIR, OUTPUT_FILE)