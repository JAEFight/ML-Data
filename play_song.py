import random
import yt_dlp
import webbrowser
from selenium import webdriver
import time

driver = webdriver.Chrome()
happy_url = "https://www.youtube.com/playlist?list=RDCLAK5uy_kvmdYWgmu7MBsrWUzv53AyF02ytmE18bo"
sad_url = "https://www.youtube.com/playlist?list=RDCLAK5uy_n8Hg9csbDQCPDH-PKmWZmKVG_oMgvrozY"
def play(url):
    # Playlist-Videos abrufen
    ydl_opts = {'quiet': True, 'extract_flat': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_info = ydl.extract_info(url, download=False)
    #print(playlist_info['entries'])
    # Alle Video-IDs extrahieren
    video_urls = [entry['url'] for entry in playlist_info['entries']]
    titles = [entry['title'] for entry in playlist_info['entries']]
    index = random.randint(0, len(video_urls))

    # Zufälliges Video wählen und abspielen
    random_video = video_urls[index]
    print(titles[index])
    #webbrowser.open(random_video)
    driver.get(random_video)
    driver.implicitly_wait(0.5)
    #button = driver.find_element_by_class('yt-spec-button-shape-next yt-spec-button-shape-next--filled yt-spec-button-shape-next--mono yt-spec-button-shape-next--size-m')
    #button.click()
    time.sleep(10)
    driver.quit()
def happy_song():
    play(happy_url)
def sad_song():
    play(sad_url)
sad_song()