import requests
import json

#public
PROJECT_ID = "nmxfh"
VIEW_ONLY = "b214f40525fe44b5bea3103478d08d0d"

#private
# PROJECT_ID = "c9zab" 
# VIEW_ONLY = "d3c6b547365d4e1f8095e928101ee4a5"

BASE_URL = f"https://files.us.osf.io/v1/resources/{PROJECT_ID}/providers/osfstorage/"

def get_json(url):
    params = {"view_only": VIEW_ONLY}
    resp = requests.get(url, params=params)
    # resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def traverse(url, path_prefix=""):
    data = get_json(url)
    for item in data.get("data", []):
        attrs = item["attributes"]
        name = attrs["name"]
        kind = attrs["kind"]
        path = attrs["path"]
        full_path = f"{path_prefix}/{name}" if path_prefix else name

        if kind == "folder":
            next_url = item["links"]["move"]
            yield from traverse(next_url, full_path)
        elif kind == "file":
            file_id = path.strip("/")
            # download_url = f"https://files.us.osf.io/v1/resources/{PROJECT_ID}/providers/osfstorage/{file_id}?view_only={VIEW_ONLY}"
            download_url = f"https://files.us.osf.io/v1/resources/{PROJECT_ID}/providers/osfstorage/{file_id}"
            yield (full_path, download_url)

if __name__ == "__main__":
    results = list(traverse(BASE_URL))
    print(f"Found {len(results)} files\n")
    for path, link in results:
        print(f"{path}\n  -> {link}\n")

    with open("osf_file_links.json", "w") as f:
        json.dump({p: l for p, l in results}, f, indent=2)