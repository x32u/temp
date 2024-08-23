import os
import hashlib
import xml.etree.ElementTree as ET
import re
import uwuify
import string
import random
import secrets
import asyncio
import aiohttp
import asyncpg
import logging
import uvicorn
import pyppeteer
import time

from typing import Optional, List, Literal
from models import *
from nudenet import NudeDetector
from collections import defaultdict
from aiofiles import open as aio_open
from starlette.requests import Request
from socials import Socials, DiscordOauth
from PIL import Image
from io import BytesIO
from rembg import remove 
from dotenv import load_dotenv


from fastapi import Header
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

templates = Jinja2Templates(directory="templates")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

load_dotenv(verbose=True)


class Worker:
    def check_token(self, token: str) -> bool:
        # Add logic to validate bot tokens
        if token == os.getenv("DISCORD_BOT_TOKEN"):
            return True  # Validate against the bot token from .env
        # Add your existing token validation logic here
        return self.validate_user_token(token)

    def validate_user_token(self, token: str) -> bool:
        # Your existing user token validation logic
        pass

class LastFM():
    def __init__(self):
        self.api_key = "43693facbb24d1ac893a7d33846b15cc"
        self.baseurl = "https://ws.audioscrobbler.com/2.0/"
        self.cache = Cache()    
    
    async def read_image(self, url: str):
        async with aiohttp.ClientSession() as cs: 
            async with cs.get(url) as r: 
                return Image.open(
                   BytesIO(await r.read())
                ).convert("RGBA")
    
    async def lastfm_user_exists(self, username: str) -> bool: 
        return 'error' not in await self.get_user_info(username)
    
    async def do_request(self, data: dict) -> dict:  
        async with aiohttp.ClientSession() as cs: 
            async with cs.get(self.baseurl, params=data) as r: 
                return await r.json()

    async def get_user_info(self, username: str) -> dict: 
        data = {
            'method': 'user.getinfo',
            'user': username, 
            'api_key': self.api_key, 
            'format': 'json'
        }
        return await self.do_request(data) 
    
    async def get_album_info(self, username: str, album: str, artist: str):
        data = {
            'method': 'album.getInfo',
            'user': username, 
            'api_key': self.api_key, 
            'format': 'json',
            'album': album, 
            'artist': artist
        }
        return await self.do_request(data)
    
    async def get_recent_tracks(self, username: str, tracks: int):
        data = {
            'method': 'user.getrecenttracks',
            'api_key': self.api_key, 
            'format': 'json',
            'user': username, 
            'limit': tracks
        }
        return await self.do_request(data)
    
    async def get_recents(self, username: str, period: str, limit: int, method: str):
        data = {
            'method': method, 
            'api_key': self.api_key, 
            'user': username, 
            'format': 'json',
            'limit': limit, 
            'period': period
        }
        return await self.do_request(data)

class CustomApp(FastAPI):
    def __init__(self):
        super().__init__(
            redoc_url=None,
            docs_url=None, 
            title="Kure API",
            on_shutdown=[self.shutdown],
            on_startup=[self.startup],
            openapi_tags=[
                {
                    'name': 'Uwuify', 
                    'description': 'convert a message to the uwu format'
                },
                {
                    'name': 'TikTok',
                    'description': 'TikTok related endpoints'
                },
                {
                    'name': 'Roblox',
                    'description': 'Roblox related endpoints'
                },
                {
                    'name': 'Snapchat',
                    'description': 'Snapchat related endpoints'
                },
                {
                    'name': 'Instagram', 
                    'description': 'Instagram related endpoints' 
                },
                {
                    'name': 'Spotify', 
                    'description': 'Spotify related endpoints'   
                },
                {
                    'name': 'LastFM', 
                    'description': 'Last FM service related endpoints'
                },
                {
                    'name': 'Discord',
                    'description': 'Discord related endpoints'
                },
                {
                    'name': 'OpenAI', 
                    'description': 'Interact with OpenAI services'
                },
                {
                    'name': 'Media',
                    'description': 'Image related endpoints'
                },
                {
                    'name': 'Roleplay',
                    'description': 'Express yourself using gifs'
                },
                {
                    'name': 'Master',
                    'description': 'API endpoints that can be used by masters only (evict bot owners)'
                }
            ]
        )
        
        self.socials = Socials()
        self.cache = Cache()
        self.threshold = Threshold()
        self.worker = Worker()
        self.characters = list(string.ascii_letters + string.digits)
        self.openapi = self.custom_openapi
        self.nude_detector = NudeDetector()
        self.last_change_time = None
        self.lastfm = LastFM()
        self.locks = defaultdict(asyncio.Lock)
        self.gpt_key = os.environ.get("gpt_key")
    
    def run(self, *args, **kwargs):
        return uvicorn.run(*args, **kwargs)

    def shutdown(self):
        for file in os.listdir("./screenshots"):
            os.remove(os.path.join("./screenshots/", file))

        for file in os.listdir("./spotify"): 
            os.remove(os.path.join("./spotify/", file))

        for file in os.listdir("./temporary/"):
            os.remove(os.path.join("./temporary/", file))

    async def startup(self):
        app.state.db = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='postgres',
            password=os.environ.get("postgresql_password"),
            database=os.environ.get("postgresql_database"),
            record_class=Record,
            max_inactive_connection_lifetime=0
        )
        
        api_keys = await app.state.db.fetch("SELECT * FROM api_key")
        logging.info(f"Caching {len(api_keys)} API keys")
        for api_key in api_keys: 
            api = APIKey(
                key=api_key.key, 
                user_id=api_key.user_id, 
                role=api_key.role
            )
            await self.cache.set(f"api-key-{api.key}", api)

    def log_change(self):
        current_time = time.time()
        if self.last_change_time is None or current_time - self.last_change_time > 60:
            logging.info("1 change detected")
            self.last_change_time = current_time

    async def read_url(self, url: str, content_type: Optional[List[str]] = None): 
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))" 
        if not re.search(regex, url):
            raise HTTPException(
                status_code=400, 
                detail="The parameter provided is not an url"
            )
        
        async with aiohttp.ClientSession() as cs: 
            async with cs.get(url) as r: 
                if content_type: 
                    if r.content_type not in content_type:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"I expected to receive {', '.join(content_type)} not {r.content_type}"
                        )
                    
                if int(r.headers.get("Content-Length")) > 5000000:
                    raise HTTPException(
                        status_code=400, 
                        detail="Image size is bigger than 5 MB"
                    )
                
                return await r.read()
    
    def contains_nude(self, path: str) -> bool: 
        bad_filters = [
            "BUTTOCKS_EXPOSED", 
            "FEMALE_BREAST_EXPOSED", 
            "ANUS_EXPOSED", 
            "FEMALE_GENITALIA_EXPOSED", 
            "MALE_GENITALIA_EXPOSED"
        ]
        detections = self.nude_detector.detect(path)
        return any([prediction['class'] in bad_filters for prediction in detections])

    async def screenshot(self, url: str, timeout: int):
        path = f"./screenshots/{url.replace('https://', '').replace('http://', '').replace('/', '')}"

        if os.path.exists(path):
            return path
        
        viewport = {
            'width': 1980,
            'height': 1080
        }
        
        browser = await pyppeteer.launch(
            headless=True,
            defaultViewport=viewport, 
            args = ["--no-sandbox"]
        )

        page = await browser.newPage()
        r = await page.goto(url)
        if content_type := r.headers.get('content-type'):
            if not any([m in content_type for m in ['text/html', 'application/json']]):
                raise HTTPException(
                    status_code=400, 
                    detail="Not allowed to screenshot this page"
                )
            
            keywords = ['pussy', 'tits', 'porn']
            page_content = await page.content()
            
            if any(re.search(r'\b{}\b'.format(keyword), page_content, re.IGNORECASE) for keyword in keywords):
                raise HTTPException(
                    status_code=403, 
                    detail="This website contains explicit content"
                )

            await asyncio.sleep(timeout)
            await page.screenshot(path=path + ".png") 
            if app.contains_nude(path + ".png"):
                os.remove(path + ".png")
                raise HTTPException(
                    status_code=403,
                    detail="This website contains explicit content"  
                )

            await browser.close()
            return path
        
        return None

    def custom_openapi(self):
        if self.openapi_schema:
            return self.openapi_schema

        openapi_schema = get_openapi(
            title="Kure API",
            version="1.0.0",
            routes=self.routes,
            summary="A private API for Evict."
        )
        openapi_schema["info"]["x-logo"] = {
            "url": "https://cdn.discordapp.com/banners/1177424668328726548/a_107bac482c5a64d493a87f989b84f202.gif?size=1024"
        }
        self.openapi_schema = openapi_schema
        return openapi_schema

app = CustomApp()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class AuthScheme(APIKeyHeader):

    async def get_ratelimit(self, api: APIKey):
        if api.role == "basic": 
            threshold_limit = 100
        elif api.role == "pro": 
            threshold_limit = 250
        else: 
            threshold_limit = 500
        
        threshold = app.threshold.get(api.key)
        if threshold > threshold_limit:
            logging.info(f"Key {api.key} ({api.role}) is ratelimited")
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )
        
        await app.threshold.set(api.key)

    async def __call__(self, request: Request) -> str:
        key = await super().__call__(request)
        api: Optional[APIKey] = app.cache.get(f"api-key-{key}")

        if not api:
            raise HTTPException(status_code=403, detail="Unauthorized")

        logging.info(f"{api.user_id} ({api.role}) made a request using the key: {api.key} on {request.url}")
        
        if api.role == "master" or (
            api.role == "bot_developer"
            and request.method == "POST" 
            and str(request.url) == f"{request.base_url}/avatars"
        ):
            return api 
        
        await self.get_ratelimit(api)
        return api

auth_scheme = AuthScheme(name="api-key")

@app.get("/", include_in_schema=False)
async def docs():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title,
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
        swagger_favicon_url="https://cdn.resent.dev/resent.png"
    )

@app.get("/shards/get", include_in_schema=False)
async def shards():
    return app.cache.get('shards') or []

@app.post("/shards/post", include_in_schema=False)
async def create_shards(body: Shards, token: APIKey = Depends(auth_scheme)):
    if token.role != 'master':
        raise HTTPException(
            status_code=403,
            detail="You are not allowed to access this endpoint."
        )
    await app.cache.set('shards', {"bot": body.bot, "shards": body.shards})

@app.get("/docs", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/")


LastFM_API_KEY = "166f77a5ce11d4860de9acf5d384e426"
LastFM_API_SECRET = "8242c399917a50594d38e0371d50ed53"

def generate_api_sig(params: dict, secret: str) -> str:
    """Generate API signature for Last.fm"""
    sorted_params = ''.join(f"{key}{value}" for key, value in sorted(params.items()))
    signature_base_string = sorted_params + secret
    logging.info(f"Signature base string: {signature_base_string}")  # Log the base string used for signature
    return hashlib.md5(signature_base_string.encode('utf-8')).hexdigest()

@app.get("/lastfm/authorize")
async def authorize(discord_user_id: str):
    redirect_uri = f"https://kure.pl/lastfm/callback?discord_user_id={discord_user_id}"
    return RedirectResponse(f"http://www.last.fm/api/auth/?api_key={LastFM_API_KEY}&cb={redirect_uri}")

@app.get("/lastfm/callback", tags=['auth'], include_in_schema=False)
async def lastfm_callback(request: Request, token: str, discord_user_id: str):
    """Last.fm OAuth callback"""
    params = {
        'method': 'auth.getSession',
        'api_key': LastFM_API_KEY,
        'token': token,
    }
    api_sig = generate_api_sig(params, LastFM_API_SECRET)
    params['api_sig'] = api_sig

    logging.info(f"Request params with api_sig: {params}")  # Log the request parameters with api_sig
    logging.info(f"Generated API signature: {api_sig}")  # Log the generated API signature

    async with aiohttp.ClientSession() as session:
        async with session.get("http://ws.audioscrobbler.com/2.0/", params=params) as resp:
            text = await resp.text()
            logging.info(f"Last.fm API response: {text}")  # Log the response data

            try:
                root = ET.fromstring(text)
                error_element = root.find('error')
                if error_element is not None:
                    error_message = error_element.find('message')
                    error_message_text = error_message.text if error_message is not None else "Unknown error"
                    logging.error(f"Invalid response from Last.fm API: {error_message_text}")  # Log the error response
                    raise HTTPException(status_code=400, detail=f"Invalid response from Last.fm API: {error_message_text}")

                session_element = root.find('session')
                username = session_element.find('name').text
                session_key = session_element.find('key').text

                # Convert discord_user_id to integer
                discord_user_id_int = int(discord_user_id)

                # Store the Discord user ID, Last.fm username, and session key in the database
                async with app.state.db.acquire() as connection:
                    await connection.execute(
                        "INSERT INTO lastfm_users (discord_user_id, username, session_key) VALUES ($1, $2, $3)",
                        discord_user_id_int, username, session_key
                    )

                return {"message": "Authorization successful!", "username": username}
            except ET.ParseError as e:
                logging.error(f"Failed to parse XML response: {e}")
                raise HTTPException(status_code=500, detail="Failed to parse XML response from Last.fm API")
            except AttributeError as e:
                logging.error(f"Missing expected elements in XML response: {e}")
                raise HTTPException(status_code=500, detail="Missing expected elements in XML response from Last.fm API")
            except ValueError as e:
                logging.error(f"Invalid discord_user_id: {e}")
                raise HTTPException(status_code=400, detail="Invalid discord_user_id")

@app.get("/callback", tags=['auth'], include_in_schema=True)
async def callback(request: Request, code: str, state: str):
    """Oauth callback"""

    oauth = DiscordOauth(
        "1203514684326805524", 
        os.environ.get("DISCORD_CLIENT_SECRET"), 
        os.environ.get("DISCORD_BOT_TOKEN")
    )

    if access_token := await oauth.exchange_code(code): 
        if user := await oauth.get_user(access_token):
            if guilds := await oauth.get_user_guilds(access_token): 
                if current_guild := next(
                    (g for g in guilds if g['id'] == state), 
                    None
                ):
                    await oauth.add_roles_for(
                        user['id'], 
                        current_guild['id']
                    )
                    return templates.TemplateResponse(
                        "verify.html",
                        {
                            'request': request, 
                            'guild': current_guild['name'],
                            'user': user['username']
                        }
                    )
                return templates.TemplateResponse(
                    "error.html",
                    {
                        'request': request,
                        'message': "You are not in this guild"
                    }
                )
    
    return templates.TemplateResponse(
        "error.html",
        {
            'request': request,
            'message': "Unable to verify you"
        }
    )

@app.get("/uwu", response_model=UwuModel, tags=['Uwuify'])
async def uwu(message: str) -> JSONResponse:
    """Uwuify your messages"""
    
    flags = uwuify.YU | uwuify.STUTTER
    return {"message": uwuify.uwu(message, flags=flags)}

@app.get("/snapchat/user", tags=['Socials'], response_model=SnapChatUserModel)
async def snapchat(username: str, token=Depends(auth_scheme)) -> JSONResponse:
    """Get a snapchat user information"""
    return await app.socials.snap.get_user(username)

@app.get("/snapchat/story", tags=['Socials'], response_model=SnapChatStoryModel)
async def snap_story(username: str, token=Depends(auth_scheme)) -> JSONResponse:
    """Fetch an user's snapchat story"""
    return await app.socials.snap.get_story(username)

@app.get("/tiktok", tags=['Socials'], response_model=TikTokModel)
async def tiktok(username: str, token=Depends(auth_scheme)) -> JSONResponse:
    """Get a tiktok user's information"""
    return await app.socials.tiktok.scrape(username)

@app.get("/roblox", tags=['Socials'], response_model=RobloxModel)
async def roblox(username: str, token=Depends(auth_scheme)) -> JSONResponse:
    """Get a roblox profile's information"""
    return await app.socials.roblox.scrape(username)

@app.get("/instagram/user", tags=['Socials'], response_model=InstagramUser)
async def instagram_user(username: str, token = Depends(auth_scheme)) -> JSONResponse:
    """Get information about someone's instagram account"""
    payload = await app.socials.instagram.get_user(username)
    if payload == 404: 
        raise HTTPException(
            status_code=404, 
            detail="Instagram user not found"
        )
    return payload

@app.get("/instagram/story", tags=['Socials'], response_model=InstagramStories)
async def instagram_story(username: str, token = Depends(auth_scheme)):
    """Get an user's instagram story (if any)"""
    payload = await app.socials.instagram.get_story(username)
    if payload == 404: 
        raise HTTPException(
            status_code=404, 
            detail="Instagram story not found"
        )
    return payload

@app.get("/spotify/downloads/{id}", tags=['Audio'])
async def spotify_downloads(id: str): 
    """Get the downloaded spotify song"""
    path = os.path.join("./spotify", f"{id}.mp3")
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg")
    return "Content Not Available"

@app.get("/spotify/song", tags=['Audio'], response_model=SpotifySong)
async def spotify_song(request: Request, url: str, token=Depends(auth_scheme)) -> JSONResponse:
    """Download a spotify song from url"""
    async with app.locks[token.key]:
        if cache := app.cache.get(f"spotify-{url}"):
            return cache 
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36",
            "Content-Type": "application/json"
        } 

        async with cs.get("https://api.fabdl.com/spotify/get", params={'url': url}) as r: 
            x = await r.json()
        
        if result := x.get('result'):
            id = result['id']
            gid = result['gid']
            artist = result['artists']
            title = result['name']
            image = result['image']

            async with aiohttp.ClientSession(headers=headers) as cs: 
                async with cs.get(f"https://api.fabdl.com/spotify/mp3-convert-task/{gid}/{id}") as r:
                    y: dict = await r.json()

            if re := y.get('result'):
                download_url = re.get('download_url')

                async def recursive_download_url(tid: str) -> str: 
                    async with aiohttp.ClientSession(headers=headers) as session:
                        async with session.get(f"https://api.fabdl.com/spotify/mp3-convert-progress/{tid}") as res:
                            d = await res.json()
    
                            if p := d.get('result'):
                                if url := p.get('download_url'):
                                    return url 
    
                                return await recursive_download_url(p['tid']) 
                        
                if not download_url: 
                    download_url = await recursive_download_url(re['tid'])
                  
                async with aiohttp.ClientSession(headers=headers) as cs: 
                    async with cs.get(f"https://api.fabdl.com{download_url}") as r:
                        download_byte = await r.read()
                
                download_path = os.path.join("./spotify", f"{id}.mp3")
                async with aio_open(download_path, "wb") as f: 
                    await f.write(download_byte)
                
                payload = {
                    "artist": artist, 
                    "title": title, 
                    "image": image, 
                    "download_url": str(request.url_for("spotify_downloads", id=id))
                } 

                await app.cache.set(f"spotify-{url}", payload)
                return payload 
        
        raise HTTPException(
            status_code=400, 
            detail="Unable to download song"
        )

@app.get("/lastfm/albumplays", tags=['Audio'], response_model=LastFmAlbumPlays)
async def lastfm_album_plays(username: str, album: str, artist: str, token = Depends(auth_scheme)):
    """Get the amount of plays a Last FM user has on a specific album"""
  
    results = await app.lastfm.get_album_info(
        username,
        album,
        artist
    )
    
    if message := results.get('message'): 
        raise HTTPException(
            status_code=404,
            detail=message
        )

    return {
        "album_name": results['album']['name'],
        "artist_name": results['album']['artist'],
        "plays": results['album']['userplaycount'],
        "tracks": len(results['album']['tracks']['track']),
        "url": results['album']['url'],
        "listeners": results['album']['listeners']
    }

@app.get("/lastfm/recenttracks", tags=['Audio'], response_model=LastFmRecent)
async def lastfm_recent_tracks(username: str, tracks: int = 1, token = Depends(auth_scheme)):
    """Get the most recent scrobbled tracks of a Last FM user"""
    
    if recent_tracks := app.cache.get(f"recenttracks-{username}-{tracks}"):
        return recent_tracks

    results = await app.lastfm.get_recent_tracks(username, tracks)
    fetched_tracks = results['recenttracks']['track']
    tracks = {
        "tracks": [
            {
                'name': fetched_track['name'],
                'artist': fetched_track['artist']['#text'],
                'image': next((e['#text'] for e in fetched_track['image'] if e['size'] == 'large'), None),
                'album': fetched_track['album']['#text'],
                'url': fetched_track['url']
            }
            for fetched_track in fetched_tracks
        ]
    }
    await app.cache.set(f"recenttracks-{username}-{tracks}", tracks, 1800)
    return tracks

@app.get("/lastfm/chart", tags=['Audio'], response_model=TransparentModel)
async def lastfm_chart(
    request: Request,
    username: str,
    size: str = "3x3",
    period: Literal['overall', '7day', '1month', '3month', '6month', '12month'] = 'overall',
    token = Depends(auth_scheme)
):
    """Create a chart of top album covers (beta)"""
    
    key = next(
        (k for k in app.cache.payload.keys() if k.startswith(f"lastfm_{size}_{period}_{username}")),
        None
    )

    if key: 
        return {
            "image_url": str(request.url_for('cdn', key=key, format='png'))
        }
    
    async with app.locks[f"lastfm-{username}"]:
        mode = "album" 
        try:   
            a, b = map(int, size.split("x"))
        except ValueError: 
            raise HTTPException(
                status_code=400, 
                detail="Wrong size format. Size example (3x3)"
            )
        
        album_count = a * b

        if album_count > 50: 
            raise HTTPException(
                status_code=400, 
                detail="Your chart cannot contain more than 50 albums"
            )

        results = await app.lastfm.get_recents(
            username, 
            period, 
            album_count, 
            "user.gettopalbums"
        )
            
        fetched_results = results[next(iter(results.keys()))][mode]
        tasks = []

        for r in fetched_results: 
            if r['image'][2]['#text'] != '': 
                tasks.append(app.lastfm.read_image(r['image'][2]['#text']))

        imgs = [image for image in await asyncio.gather(*tasks)]

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(a * w, b * h))
        
        for i, image in enumerate(imgs):
            grid.paste(
                image, 
                (
                    (i % a) * w,
                    (i // a) * h 
                )
            )

        buffer = BytesIO()
        grid.save(buffer, format="png")
        buffer.seek(0)
        key = f"lastfm_{size}_{period}_{username}_" + ''.join(secrets.choice(app.characters) for _ in range(8))
        await app.cache.set(
            key, 
            buffer.read(), 
            3600
        )
        
        return {
            "image_url": str(request.url_for('cdn', key=key, format='png'))
        }

@app.post("/openai/chatgpt", tags=['OpenAI'], response_model=ChatgptResponse)
async def chatgpt(prompt: str, version: float = 3.5, token: APIKey = Depends(auth_scheme)): 
    """Get a response from chatgpt"""
    
    if token.role not in ['premium', 'bot_developer', 'master']: 
        raise HTTPException(
            status_code=403, 
            detail="Please upgrade to premium in order to use this endpoint"
        )

    if version not in [3.5, 4]:
        raise HTTPException(
            status_code=400, 
            detail="This is not a valid version of chatgpt. Please choose between 3.5 and 4"
        )
    
    if version == 4 and token.role != 'master': 
        raise HTTPException(
            status_code=403, 
            detail="You do not have access to this chatgpt version at the moment"
        )
    
    async with app.locks[str(token)]:
        if res := app.cache.get(f"chatgpt-{prompt.lower().strip()}"):
            return {
                "response": res
            }
        
        headers = {
            "Authorization": f"Bearer {app.gpt_key}", 
            "Content-Type": "application/json"
        }
    
        payload = {
            "model": "gpt-3.5-turbo" if version == 3.5 else "gpt-4", 
            "messages": [{"role": "user", "content": prompt}], 
            "temperature": 0.7
        }
    
        async with aiohttp.ClientSession(headers=headers) as cs: 
            async with cs.post("https://api.openai.com/v1/chat/completions", json=payload) as r: 
                data = await r.json()
                try:
                    response = data['choices'][0]['message']['content']
                    await app.cache.set(
    f"chatgpt-{prompt.lower().strip()}", 
                        response
                    )
                    return {
                        "response": response
                    }
                except: 
                    raise HTTPException(
                        status_code=429, 
                        detail="Too many requests to chatgpt"
                    )

@app.post("/avatars", tags=['Socials'])
async def discord_user_avatar_post(body: DiscordAvatarPost, token = Depends(auth_scheme)):
    """Post an user avatar to our network (bot developers only endpoint)"""

    if token.role not in ['master', 'bot_developer']:
        raise HTTPException(
            status_code=403, 
            detail="You do not have access to this endpoint"
        )
    
    async with app.locks[1]:
        if app.cache.get(f"avatars-{body.url}"):
            return "Duplicated avatar detected in our network"
        
        await app.cache.set(
            f"avatars-{body.url}", 
            True
        )
    
        headers = {
            "Authorization": os.environ.get("evict_key"),
            "Content-Type": "application/json"
        }
    
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post("https://images.evict.cc/upload", json=body.dict()) as r: 
                return await r.text()

@app.get("/avatars/{user_id}", tags=['Socials'])
async def discord_user_avatars(request: Request, user_id: int, token = Depends(auth_scheme)):
    """
    Get a list of discord user avatars
    """
    
    if user_avatars := app.cache.get(f"user-avatars-{user_id}"):
        return user_avatars

    async with aiohttp.ClientSession() as cs: 
        async with cs.get(f"https://evict.cc/avatarhistory/{user_id}") as r: 
            data = await r.text()
            if data == "No avatars found":
                return data
            
            data = json.loads(data)
            user_avatars = [f"https://images.evict.cc/images/{av}" for av in data]
            
            await app.cache.set(
                f"user-avatars-{user_id}", 
                user_avatars, 
                600
            )
            
            return user_avatars 

@app.get("/discord/user/profile", tags=['Socials'], response_model=DiscordUserProfileModel)
async def discord_user_profile(user_id: int, guild_id: Optional[int] = None, api_key: str = Header(None)):
    """Get a discord user's information"""

    token = os.getenv("DISCORD_BOT_TOKEN")
    logging.info(f"Fetching profile for user_id: {user_id}, guild_id: {guild_id}, token: {token}")

    # Check if the token is valid
    if not app.worker.check_token(token): 
        logging.warning(f"Invalid token for user_id: {user_id}")
        raise HTTPException(
            status_code=401, 
            detail="Given token is invalid or not a discord user token"
        )

    # Check if the user information is cached
    payload = app.cache.get(f"discorduser-{user_id}-{guild_id}")
    
    if payload == token: 
        logging.warning(f"Couldn't fetch details about user_id: {user_id}, guild_id: {guild_id}")
        raise HTTPException(
            status_code=404, 
            detail="Couldn't fetch details about this user"
        )
    else: 
        payload = None

    # Fetch user information from Discord
    if not payload: 
        try:
            user_data = await app.socials.discord.get_user(token, str(user_id), str(guild_id))
            logging.info(f"Fetched user data: {user_data}")
            if not user_data:
                raise HTTPException(
                    status_code=404, 
                    detail="Couldn't fetch details about this user"
                )
            payload = DiscordUserProfileModel.from_discord_response(user_data)
        except Exception as e:
            logging.error(f"Error fetching user data: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Error fetching user data"
            )

        await app.cache.set(
            f"discorduser-{user_id}-{guild_id}", 
            payload.dict(), 
            3600
        )

    return payload


@app.get("/roleplay/bite", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_bite(request: Request, token = Depends(auth_scheme)):
    """Gen a bite anime gif"""
    
    p = "./roleplay/bite/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    }  

@app.get("/roleplay/cuddle", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_cuddle(request: Request, token = Depends(auth_scheme)):
    """Gen a cuddle anime gif"""
    
    p = "./roleplay/cuddle/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 

@app.get("/roleplay/hug", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_hug(request: Request, token = Depends(auth_scheme)):
    """Gen a hug anime gif"""
    
    p = "./roleplay/hug/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 
    
@app.get("/roleplay/fuck", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_fuck(request: Request, token = Depends(auth_scheme)):
    """Gen a fuck anime gif"""
    
    p = "./roleplay/fuck/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 
    
@app.get("/roleplay/threesome", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_threesome(request: Request, token = Depends(auth_scheme)):
    """Gen a threesome anime gif"""
    
    p = "./roleplay/fuck/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 
    
@app.get("/roleplay/kill", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_kill(request: Request, token = Depends(auth_scheme)):
    """Gen a kill anime gif"""
    
    p = "./roleplay/kill/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 

@app.get("/roleplay/hump", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_hump(request: Request, token = Depends(auth_scheme)):
    """Gen a hump anime gif"""
    
    p = "./roleplay/hump/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 
    
@app.get("/roleplay/kiss", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_kiss(request: Request, token = Depends(auth_scheme)):
    """Gen a kiss anime gif"""
    
    p = "./roleplay/kiss/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 

@app.get("/roleplay/lick", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_lick(request: Request, token = Depends(auth_scheme)):
    """Gen a lick anime lick"""
    
    p = "./roleplay/lick/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 

@app.get("/roleplay/pat", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_pat(request: Request, token = Depends(auth_scheme)):
    """Gen a pat anime gif"""
    
    p = "./roleplay/pat/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 

@app.get("/roleplay/punch", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_punch(request: Request, token = Depends(auth_scheme)):
    """Gen a punch anime gif"""
    
    p = "./roleplay/punch/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 
    
@app.get("/roleplay/poke", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_poke(request: Request, token = Depends(auth_scheme)):
    """Gen a poke anime gif"""
    
    p = "./roleplay/poke/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 
    
@app.get("/roleplay/slap", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_slap(request: Request, token = Depends(auth_scheme)):
    """Gen a slap anime gif"""
    
    p = "./roleplay/slap/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 
    
@app.get("/roleplay/spank", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_spank(request: Request, token = Depends(auth_scheme)):
    """Gen a spank anime gif"""
    
    p = "./roleplay/spank/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 
    
@app.get("/roleplay/feed", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_feed(request: Request, token = Depends(auth_scheme)):
    """Gen a feed anime gif"""
    
    p = "./roleplay/feed/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 
    
@app.get("/roleplay/highfive", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_highfive(request: Request, token = Depends(auth_scheme)):
    """Gen a highfive anime gif"""
    
    p = "./roleplay/highfive/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 
    
@app.get("/roleplay/nutkick", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_nutkick(request: Request, token = Depends(auth_scheme)):
    """Gen a nutkick anime gif"""
    
    p = "./roleplay/nutkick/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 

@app.get("/roleplay/tickle", tags=['Roleplay'], response_model=TransparentModel)
async def roleplay_tickle(request: Request, token = Depends(auth_scheme)):
    """Gen a tickle anime gif"""
    
    p = "./roleplay/tickle/"
    path = os.path.join(p, random.choice(os.listdir(p)))
    key = ''.join(secrets.choice(app.characters) for _ in range(13))
    await app.cache.set(
        key, 
        path[:-4]
    )

    return {
        "image_url": str(request.url_for('cdn', key=key, format="gif"))
    } 

@app.get("/cdn/{key}.{format}", include_in_schema=False)
async def cdn(key: str, format: str):
    """Get an image"""

    if cache := app.cache.get(key):
        if isinstance(cache, str):
            return FileResponse(
                f"{cache}.{format}"
            )  
        else: 
            return Response(
                content=cache, media_type=f"image/{format}"
            )
    else: 
        return "No Content Available"

@app.get("/nsfw", tags=['Media'], response_model=IsNsfw)
async def nsfw(url: str, token = Depends(auth_scheme)):
    """Check if the provided image is nsfw"""
    
    data = await app.read_url(url, content_type=['image/png', 'image/jpeg'])
    rand = ''.join([secrets.choice(app.characters) for _ in range(5)])
    path = os.path.join("./temporary/", f"{rand}.png")

    async with aio_open(path, "wb") as f: 
        await f.write(data)

    contains_nude = app.contains_nude(path)
    os.remove(path)
    return {"is_nsfw": contains_nude}

@app.get("/videotogif", tags=['Media'], response_model=TransparentModel)
async def video_to_gif(request: Request, url: str, token = Depends(auth_scheme)):
    """Convert an .mp4 or .mov file to .gif"""
    
    async with app.locks[token.user_id]:
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))" 
        if not re.search(regex, url):
            raise HTTPException(
                status_code=400, 
                detail="This is not an url"
            )
    
        async with aiohttp.ClientSession() as cs: 
            async with cs.get(url) as r: 
                if r.headers.get("Content-Type") not in ['video/mp4', 'video/mov', 'video/quicktime']:
                    raise HTTPException(
                        status_code=400, 
                        detail="URL must be an mp4 or mov file"
                    )
    
                if int(r.headers.get("Content-Length")) > 5000000:
                    raise HTTPException(
                        status_code=400, 
                        detail="Image size is bigger than 5 MB"
                    )
                
                temp_path = f"./temporary/attachment_{token.user_id}.{r.headers['Content-Type'].split('/')[1]}"
                async with aio_open(temp_path, "wb") as f:
                    await f.write(await r.read())
                
                gif_path = temp_path[:-3] + 'gif'
                os.system(f"ffmpeg -i {temp_path} {gif_path}")
                key = ''.join(secrets.choice(app.characters) for _ in range(10))
                await app.cache.set(
                    key, 
                    gif_path[:-4]
                )

                return {
                    "image_url": str(request.url_for('cdn', key=key, format='gif'))
                }

@app.get("/pictures/{type}/{category}", tags=['Media'], response_model=PfpModel)
async def pictures(
    request: Request, 
    type: str, 
    category: str, 
    format: Optional[Literal['png', 'gif', None]] = None, 
    token=Depends(auth_scheme)
) -> JSONResponse: 
    """Get a random picture from a category"""

    path = os.path.join(
        "./PretendImages", 
        type.capitalize(), 
        category.capitalize()
    )

    if not os.path.exists(path): 
        raise HTTPException(status_code=404, detail="Category does not exist.")
    
    dir_pictures = os.listdir(path)

    if format: 
        dir_pictures = [d for d in dir_pictures if d.endswith('gif')] if format == "gif" else [d for d in dir_pictures if not d.endswith('gif')]

    img = random.choice(dir_pictures)
    
    key = next(
        (k for k, v in app.cache.payload.items() if v == f"{type.capitalize()}/{category.capitalize()}/{img[:-4]}"),
        None
    )

    if not key:
        key = ''.join(
            random.choice(app.characters) for _ in range(12)
        )
    
        await app.cache.set(
            key,
            f"./PretendImages/{type.capitalize()}/{category.capitalize()}/{img[:-4]}"
        )

    return {
        "type": type, 
        "category": category, 
        "url": request.url_for('cdn', key=key, format=img.split(".")[1]).__str__()
    }

@app.get("/screenshot", response_model=ScreenshotModel, tags=['Media'])
async def website_screenshot(request: Request, url: str, timeout: int = 1, token = Depends(auth_scheme)) -> JSONResponse: 
    """
    Screenshot a website
    """
    
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))" 
    if re.search(regex, url):
        key = next(
            (k for k, v in app.cache.payload.items() if v == f"./screenshots/{url.replace('https://', '').replace('http://', '').replace('/', '')}"),
            None
        )

        if key: 
            return {
                "website_url": url, 
                "screenshot_url": request.url_for('cdn', key=key, format="png").__str__()
            }
        
        if website_path := await app.screenshot(url, timeout):
            key = ''.join(
                random.choice(app.characters) for _ in range(12)
            )

            await app.cache.set(
                key, 
                website_path
            )

            return {
                "website_url": url, 
                "screenshot_url": request.url_for('cdn', key=key, format="png").__str__()
            }

    raise HTTPException(
        status_code=400, 
        detail=f"Unable to screenshot {url}"
    )        

@app.get("/transparent", response_model=TransparentModel, tags=['Media'])
async def transparent(request: Request, url: str, token: APIKey = Depends(auth_scheme)) -> JSONResponse: 
    """Remove the background from an image"""
    
    async with app.locks[str(token)]:
        data = await app.read_url(url, ["image/png", "image/jpeg"])
        random_char = ''.join(secrets.choice(app.characters) for _ in range(6)) 
        temp_file = os.path.join("./transparent/", f"transparent_{random_char}.png") 
        temp_file_output = os.path.join("./transparent/", f"transparent_output_{random_char}.png")      

        async with aio_open(temp_file, "wb") as f: 
            await f.write(data)

        def remove_bg():
            original = Image.open(temp_file)
            nobg = remove(original)
            nobg.save(temp_file_output)

        try: 
            await asyncio.to_thread(remove_bg)
        except Exception as e:
            print(f"exception: {e}") 
            raise HTTPException(
                status_code=400, 
                detail="Unable to make this image transparent"
            )
        
        if os.path.exists(temp_file):
            os.remove(temp_file)

        key = ''.join(random.choice(app.characters) for _ in range(12))
        await app.cache.set(
            key, 
            temp_file_output[:-4]
        )

        return {
            "image_url": request.url_for("cdn", key=key, format="png").__str__()
        }

@app.get("/apikey", tags=['Master'], response_model=APIKey, response_class=JSONResponse)
async def get_key(token: APIKey = Depends(auth_scheme)) -> JSONResponse: 
    """Get info about an API key"""

    if cache := app.cache.get(f"api-key-{token}"): 
        return cache.dict()

@app.post("/apikey/create", tags=['Master'], response_model=APIKey)
async def api_create(form: APIForm, token = Depends(auth_scheme)) -> JSONResponse:
    """Create an api key for someone"""

    if token.role != "master":
        raise HTTPException(
            status_code=403, 
            detail="You are not allowed to use this endpoint"
        )
    
    key = await app.state.db.fetchval("SELECT key FROM api_key WHERE user_id = $1", form.user_id)
    
    if not key:   
        key = ''.join(secrets.choice(app.characters) for _ in range(64))

    await app.state.db.execute(
        "INSERT INTO api_key VALUES ($1,$2,$3) ON CONFLICT (user_id) DO UPDATE SET role = $3",
        form.user_id, key, form.role
    )
    
    api = APIKey(
        key=key, 
        user_id=form.user_id, 
        role=form.role
    )

    await app.cache.set(f"api-key-{key}", api)
    return api.dict()

@app.delete("/apikey/revoke", tags=["Master"])
async def api_revoke(token = Depends(auth_scheme)) -> JSONResponse:
    """Revoke an API key."""

    await app.state.db.execute(
        "DELETE FROM api_key WHERE key = $1", str(token)
    )

    return Response(status_code=204)

if __name__ == "__main__":
    app.run(
        "main:app", 
        host="0.0.0.0", 
        port=5500, 
        reload=True
    )