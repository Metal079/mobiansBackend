import os
import asyncio

from dotenv import load_dotenv
import pyodbc
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob.aio import BlobClient


load_dotenv()

DBHOST = os.environ.get('DBHOST')
DBNAME = os.environ.get('DBNAME')
DBUSER = os.environ.get('DBUSER')
DBPASS = os.environ.get('DBPASS')
driver= '{ODBC Driver 17 for SQL Server}'
# cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
dsn = f'DRIVER={driver};SERVER={DBHOST};DATABASE={DBNAME};UID={DBUSER};PWD={DBPASS}'

container_name = "mobiansratings"

filtered_tags = [
    "masterpiece",
    "high quality",
    "intricate details"
    "soft lighting",
    "vibrant lighting",
    "perfect details",
    "best quality",
    "furry female",
    "animale ears",
    "vibrant lightning",
]

async def download_blob(blob_url, download_image_path, download_text_path):
    try:
        blob_client = BlobClient.from_blob_url(blob_url)
        stream = await blob_client.download_blob()
        data = await stream.readall()
        with open(download_image_path, "wb") as download_file:
            download_file.write(data)
    except Exception as e:
        print(f"An error occurred while downloading blob from {blob_url}: {str(e)}")
        print(f"Deleting corresponding text file: {download_text_path}")
        try:
            os.remove(download_text_path)
        except:
            print(f"An error occurred while deleting the text file: {download_text_path}")


async def download_blobs_from_database():
    # SQL connection parameters
    driver = '{ODBC Driver 17 for SQL Server}'
    dsn = f'DRIVER={driver};SERVER={DBHOST};DATABASE={DBNAME};UID={DBUSER};PWD={DBPASS}'

    # Connect to the SQL database
    conn = pyodbc.connect(dsn)
    cursor = conn.cursor()

    # Execute the query to fetch AzureBlobURLs, Prompt, FileName, and UserRating
    query = """SELECT [Prompt], [FileName], [UserRating], [AzureBlobURL]
               FROM [Mobians].[dbo].[UserRatings]
               WHERE AzureBlobURL IS NOT NULL AND CFG > 3"""
    cursor.execute(query)

    # Directory to save the downloaded files
    download_folder_path = os.path.join(os.getcwd(), "images")
    os.makedirs(download_folder_path, exist_ok=True)

    # Create a list to hold the download tasks
    tasks = []

    # Iterate through the result and create download tasks
    for row in cursor.fetchall():
        prompt = row.Prompt
        file_name = row.FileName
        user_rating = row.UserRating
        blob_url = row.AzureBlobURL

        # Prepend "bad quality, " to the prompt if UserRating is 0
        if user_rating == 0:
            prompt = "bad quality, " + prompt

        # Remove parentheses from the prompt
        prompt = prompt.replace('(', '').replace(')', '')

        # Split the prompt into components and filter out the specified tags
        components = prompt.split(', ')
        filtered_prompt = ', '.join([comp for comp in components if comp.lower() not in [tag.lower() for tag in filtered_tags]])


        # Define the paths for the downloaded image and text file
        download_image_path = os.path.join(download_folder_path, file_name + '.png')
        download_text_path = os.path.join(download_folder_path, file_name + '.txt')

        print(f"Downloading blob to {download_image_path}")
        print(f"Saving prompt to {download_text_path}")

        # Add the download task to the tasks list
        tasks.append(download_blob(blob_url, download_image_path, download_text_path))

        # Save the prompt to a text file
        with open(download_text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(filtered_prompt)

    # Wait for all the download tasks to complete
    await asyncio.gather(*tasks)

    print("All blobs and prompts downloaded successfully.")
    conn.close()

# Create an event loop
loop = asyncio.get_event_loop()

# Run the function to download all blobs
loop.run_until_complete(download_blobs_from_database())

# Close the loop when done
loop.close()