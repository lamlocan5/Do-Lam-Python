import os
import json
from androguard.misc import AnalyzeAPK
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Specify the folder containing APK files
folder_path = "/home/iec/HaiLam/DoLam/Adware"

# Get all files in the folder (assuming they are APKs even if they don't have .apk extension)
apk_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Use androguard to extract permissions from APK
def extract_permissions_from_apk(apk_path):
    try:
        a, _, _ = AnalyzeAPK(apk_path)
        return a.get_permissions()
    except Exception as e:
        print(f"Error extracting permissions from {apk_path}: {e}")
        return []

# Process permissions into sentences (assuming process_permissions is defined elsewhere)
def process_permissions(permissions):
    # Dummy implementation of process_permissions
    return " ".join(permissions)

# Multiprocess extraction of permissions
def extract_permissions_multiprocess(apk_paths):
    max_workers = min(len(apk_paths), multiprocessing.cpu_count())
    if max_workers == 0:
        print("No workers available for processing.")
        return []

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_apk = {executor.submit(extract_permissions_from_apk, apk_path): apk_path for apk_path in apk_paths}
        for future in as_completed(future_to_apk):
            apk_path = future_to_apk[future]
            try:
                permissions = future.result()
                if permissions:
                    sentence = process_permissions(permissions)
                    results.append({
                        "apk_path": apk_path,
                        "permissions": permissions,
                        "normalized_sentence": sentence
                    })
            except Exception as e:
                print(f"Error processing {apk_path}: {e}")
    return results

# Extract permissions from all APKs using multiprocessing and save to JSON file
if __name__ == "__main__":
    results = extract_permissions_multiprocess(apk_paths)

    # Save results to a JSON file if there are any results
    if results:
        output_file = "/home/iec/HaiLam/DoLam/Adware/output.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        # Print confirmation
        print(f"Extraction completed. Results saved to {output_file}.")
    else:
        print("No permissions were extracted.")
