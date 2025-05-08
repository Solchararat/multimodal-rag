import asyncio
import os

import aiohttp
import pandas as pd

df = pd.read_csv("dataset/data.csv")

os.makedirs("dataset/images", exist_ok=True)

CONCURRENT_DOWNLOADS = 10


async def download_image(session, idx, row):
    image_url: str = row["image_url"]
    scientific_name= row["scientific_name"].lower().replace(" ", "-")
    image_name: str = os.path.join("dataset", "images", f"{idx}-{scientific_name}.jpg")
    if os.path.exists(image_name):
        print(f"{image_name} already exists.")
        return
    try:
        async with session.get(image_url) as response:
            if response.status == 200:
                with open(image_name, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                print(f"Downloaded {image_name}")
            else:
                print(f"Failed to download {image_url}: {response.status}")
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")


async def main():
    connector = aiohttp.TCPConnector(limit=CONCURRENT_DOWNLOADS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_image(session, idx, row) for idx, row in df.iterrows()]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
