# prepare_cloud_zip.py
import zipfile
import os

def zip_project(output_filename="colab_upload.zip"):
    # 1. CORE CODE DIRECTORIES
    code_items = [
        'app_api', 'configs', 'datasets', 'engine', 'losses', 
        'metrics', 'models', 'utils', 'main.py', 'create_gallery.py', 
        'plot_losses.py', 'predict.py', 'requirements.txt'
    ]
    
    # 2. DATASET DIRECTORY
    # Looking for 'dataset' in the parent directory as per your structure
    dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "dataset"))
    
    exclude_ext = ['.pyc', '.pth', '.json', '.sh']
    exclude_dir = ['__pycache__', 'outputs', 'logs', '.git', '.venv']

    print(f"📦 Starting ZIP creation: {output_filename}")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # --- Add Code ---
        for item in code_items:
            if os.path.isfile(item):
                zipf.write(item, arcname=os.path.join("forgery-detection", item))
            elif os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    dirs[:] = [d for d in dirs if d not in exclude_dir]
                    for file in files:
                        if not any(file.endswith(ext) for ext in exclude_ext):
                            file_path = os.path.join(root, file)
                            # Store in a consistent 'forgery-detection' subfolder
                            archive_name = os.path.join("forgery-detection", file_path)
                            zipf.write(file_path, archive_name)
        print("✅ Code added.")

        # --- Add Dataset ---
        if os.path.exists(dataset_dir):
            print(f"📂 Found dataset at: {dataset_dir}. Adding to ZIP...")
            count = 0
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.png', '.tif')):
                        file_path = os.path.join(root, file)
                        # Relative path from the parent of 'dataset'
                        rel_path = os.path.relpath(file_path, os.path.join(dataset_dir, ".."))
                        zipf.write(file_path, arcname=rel_path)
                        count += 1
            print(f"✅ Dataset added ({count} images).")
        else:
            print("⚠️ Dataset folder not found in parent directory. Only code was zipped.")

    print(f"\n✨ SUCCESS! Upload '{output_filename}' to your Google Drive.")

if __name__ == "__main__":
    zip_project()
