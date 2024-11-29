

def get_gemini_api_key():
    return ""

# import os
# from yt_dlp import YoutubeDL
#
# def download_playlist(playlist_url, save_path):
#     os.makedirs(save_path, exist_ok=True)
#     ydl_opts = {
#         'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Highest quality video + audio
#         'outtmpl': os.path.join(save_path, '%(playlist_index)s - %(title)s.%(ext)s'),  # Output format
#         'ffmpeg_location': r'C:\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe',
#         'merge_output_format': 'mp4',  # Merge video and audio into MP4
#         'postprocessors': [{
#             'key': 'FFmpegVideoConvertor',
#             'preferedformat': 'mp4',  # Ensure MP4 format
#         }],
#     }
#     with YoutubeDL(ydl_opts) as ydl:
#         try:
#             print(f"Downloading playlist: {playlist_url}")
#             ydl.download([playlist_url])
#             print("Download complete!")
#         except Exception as e:
#             print(f"Error downloading playlist: {e}")
#
# if __name__ == "__main__":
#     playlist_url = "https://www.youtube.com/watch?v=CqOfi41LfDw&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1"
#     save_path = "./vedio"
#     download_playlist(playlist_url, save_path)

