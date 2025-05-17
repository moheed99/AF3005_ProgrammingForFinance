import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import yfinance as yf
import seaborn as sns
import time
import random

# --- Constants for Publicly Available Assets (Examples) ---
HOGWARTS_CREST_URL = "https://i.imgur.com/Bt5Uyvw.png"
GRYFFINDOR_LOGO_URL = "https://i.imgur.com/nCU2QKo.png"
SLYTHERIN_LOGO_URL = "https://i.imgur.com/DZ9tEb2.png"
HUFFLEPUFF_LOGO_URL = "https://i.imgur.com/vQT68mS.png"
RAVENCLAW_LOGO_URL = "https://i.imgur.com/RoTJCGM.png"

DEFAULT_HOGWARTS_VIDEO_URL = "https://i.imgur.com/xnTzwxu.mp4"
GRYFFINDOR_VIDEO_URL = "https://i.imgur.com/nXEZb5S.mp4"
SLYTHERIN_VIDEO_URL = "https://i.imgur.com/tWEp9VH.mp4"
HUFFLEPUFF_VIDEO_URL = "https://i.imgur.com/Xqzc2OQ.mp4"
RAVENCLAW_VIDEO_URL = "https://i.imgur.com/C5UmTTK.mp4"

DEFAULT_STATIC_BACKGROUND_URL = "https://i.imgur.com/6hZ0Q1M.jpg"
GRYFFINDOR_STATIC_BACKGROUND_URL = "https://i.imgur.com/9gD0x0g.jpg"
SLYTHERIN_STATIC_BACKGROUND_URL = "https://i.imgur.com/pwxpNrN.jpg"
HUFFLEPUFF_STATIC_BACKGROUND_URL = "https://i.imgur.com/B9Y4S9A.jpg"
RAVENCLAW_STATIC_BACKGROUND_URL = "https://i.imgur.com/s6p0NKy.jpg"

ARTIFACT_GIF_1_URL = "https://i.imgur.com/uQKGWJZ.gif"
ARTIFACT_GIF_2_URL = "https://i.imgur.com/xAQbA1N.gif"
ARTIFACT_GIF_3_URL = "https://i.imgur.com/N7PmfTI.gif"
CONSTRUCTION_GIF_URL = "https://i.imgur.com/9jY2L4K.gif"
OBSERVATORY_GIF_URL = "https://i.imgur.com/lqzJCQW.gif"
NO_DATA_IMAGE_URL = "https://i.imgur.com/gbsU7V1.gif"

st.set_page_config(
    page_title="Harry Potter Financial Mystics - Futuristic Wizarding World",
    page_icon="🧙‍♂️", layout="wide", initial_sidebar_state="expanded"
)

house_themes = {
    "None": {"primary":"#fedd00","secondary":"#662d91","text":"#eee7db","button_bg":"linear-gradient(45deg,#662d91,#fedd00)","button_hover_bg":"linear-gradient(45deg,#fedd00,#662d91)","house_logo":HOGWARTS_CREST_URL,"background_video":DEFAULT_HOGWARTS_VIDEO_URL,"background_image_url":DEFAULT_STATIC_BACKGROUND_URL},
    "Gryffindor": {"primary":"#AE0001","secondary":"#EEBA30","text":"#fff2cc","button_bg":"linear-gradient(45deg,#AE0001,#EEBA30)","button_hover_bg":"linear-gradient(45deg,#EEBA30,#AE0001)","house_logo":GRYFFINDOR_LOGO_URL,"background_video":GRYFFINDOR_VIDEO_URL,"background_image_url":GRYFFINDOR_STATIC_BACKGROUND_URL},
    "Slytherin": {"primary":"#1A472A","secondary":"#AAAAAA","text":"#d0f0c0","button_bg":"linear-gradient(45deg,#1A472A,#AAAAAA)","button_hover_bg":"linear-gradient(45deg,#AAAAAA,#1A472A)","house_logo":SLYTHERIN_LOGO_URL,"background_video":SLYTHERIN_VIDEO_URL,"background_image_url":SLYTHERIN_STATIC_BACKGROUND_URL},
    "Hufflepuff": {"primary":"#FFDB00","secondary":"#372E29","text":"#fff8e1","button_bg":"linear-gradient(45deg,#372E29,#FFDB00)","button_hover_bg":"linear-gradient(45deg,#FFDB00,#372E29)","house_logo":HUFFLEPUFF_LOGO_URL,"background_video":HUFFLEPUFF_VIDEO_URL,"background_image_url":HUFFLEPUFF_STATIC_BACKGROUND_URL},
    "Ravenclaw": {"primary":"#0E1A40","secondary":"#946B2D","text":"#E2F1FF","button_bg":"linear-gradient(45deg,#0E1A40,#946B2D)","button_hover_bg":"linear-gradient(45deg,#946B2D,#0E1A40)","house_logo":RAVENCLAW_LOGO_URL,"background_video":RAVENCLAW_VIDEO_URL,"background_image_url":RAVENCLAW_STATIC_BACKGROUND_URL}
}

st.sidebar.title("Hogwarts Financial Divination")
selected_house = st.sidebar.selectbox(
    "Choose Your House", options=list(house_themes.keys()),
    format_func=lambda x: f"{x} House" if x != "None" else "Hogwarts (Default)"
)
theme_config = house_themes[selected_house]
current_background_image = theme_config.get("background_image_url", DEFAULT_STATIC_BACKGROUND_URL)

house_welcome_messages = {
    "None":"Welcome to mystical financial divination!","Gryffindor":"Brave Gryffindors! Financial mastery awaits!","Slytherin":"Ambitious Slytherins! Cunning guides strategy!","Hufflepuff":"Loyal Hufflepuffs! Patience yields rewards!","Ravenclaw":"Wise Ravenclaws! Intellect solves mysteries!"
}
logo_url_from_theme = theme_config.get("house_logo", HOGWARTS_CREST_URL)
caption_text = f"The Crest of {selected_house}" if selected_house != "None" else "Hogwarts School"
st.sidebar.image(logo_url_from_theme, use_container_width=True, caption=caption_text)
st.sidebar.markdown(f"<h3 style='text-align:center;color:{theme_config['primary']};text-shadow:0 0 5px {theme_config['secondary']};'><i>{house_welcome_messages[selected_house]}</i></h3>", unsafe_allow_html=True)

def remote_css(url): st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)
remote_css("https://fonts.googleapis.com/css2?family=Creepster&display=swap")
remote_css("https://fonts.googleapis.com/css2?family=MedievalSharp&display=swap")
remote_css("https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap")
remote_css("https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap")

def hex_to_rgb(hex_color): hex_color = hex_color.lstrip('#'); return tuple(int(hex_color[i:i+2], 16) for i in (0,2,4))
primary_rgb_tuple = hex_to_rgb(theme_config["primary"]); secondary_rgb_tuple = hex_to_rgb(theme_config["secondary"])
primary_rgb_css = f"{primary_rgb_tuple[0]},{primary_rgb_tuple[1]},{primary_rgb_tuple[2]}"
secondary_rgb_css = f"{secondary_rgb_tuple[0]},{secondary_rgb_tuple[1]},{secondary_rgb_tuple[2]}"

dynamic_app_background_css = f"""
<style>
.stApp {{
    background-image: url('{current_background_image}'); background-color:#030108; background-size:cover;
    background-attachment:fixed; background-position:center center; color:{theme_config["text"]};
    font-family:'Cinzel',serif; margin:0; padding:0; overflow-x:hidden; position:relative; 
}}
.stApp::before {{
    content:''; position:fixed; top:0; left:0; width:100vw; height:100vh;
    background-image:url('https://i.imgur.com/WJIc0JL.gif'); background-size:300% 300%;
    animation:magicSparkle 20s infinite linear; pointer-events:none; z-index:-1; opacity:0.08; 
}}
.css-1d391kg .sidebar .sidebar-content {{ 
  background-color:rgba(5,2,15,0.75); backdrop-filter:blur(4px); position:relative; overflow:hidden; 
  border-right:3px solid {theme_config["primary"]}; box-shadow:7px 0 20px rgba({primary_rgb_css},0.4);
}}
.css-1d391kg .sidebar .sidebar-content::after {{
  content:''; position:absolute; top:0; left:0; right:0; bottom:0;
  background-image:url('https://i.imgur.com/aE3BnKy.gif'); opacity:0.15; z-index:-1;
  pointer-events:none; animation:magicSparkle 10s infinite linear reverse;
}}
.main .block-container {{
  position:relative; background-color:rgba(5,2,15,0.93); padding:30px !important; border-radius:30px;
  border:3px solid transparent; animation:neonBorderMove 5s ease-in-out infinite alternate, floatingElement 8s ease-in-out infinite alternate;
  background-image:linear-gradient(rgba(5,2,15,0.93),rgba(5,2,15,0.93)),linear-gradient(45deg,{theme_config["primary"]},{theme_config["secondary"]},{theme_config["primary"]});
  background-origin:border-box; background-clip:padding-box,border-box; backdrop-filter:blur(4px);
  box-shadow:0 10px 40px 0 rgba(0,0,0,0.6); margin-top:2.5rem; margin-bottom:2.5rem;
}}
</style>
"""
st.markdown(dynamic_app_background_css, unsafe_allow_html=True)

base_css_rules = f"""
<style>
@keyframes floatingElement {{0%{{transform:translate(0,0) rotate(0deg)}}25%{{transform:translate(3px,-3px) rotate(.5deg)}}50%{{transform:translate(0,-5px) rotate(0deg)}}75%{{transform:translate(-3px,-3px) rotate(-.5deg)}}100%{{transform:translate(0,0) rotate(0deg)}}}}
@keyframes neonBorderMove {{0%{{border-image-source:linear-gradient(45deg,{theme_config["primary"]},{theme_config["secondary"]},{theme_config["primary"]})}}50%{{border-image-source:linear-gradient(45deg,{theme_config["secondary"]},{theme_config["primary"]},{theme_config["secondary"]})}}100%{{border-image-source:linear-gradient(45deg,{theme_config["primary"]},{theme_config["secondary"]},{theme_config["primary"]})}}}}
@keyframes pulseGlow {{0%{{box-shadow:0 0 8px rgba({primary_rgb_css},.7),inset 0 0 8px rgba({primary_rgb_css},.5),0 0 15px {theme_config["primary"]}}}50%{{box-shadow:0 0 12px rgba({primary_rgb_css},.9),inset 0 0 12px rgba({primary_rgb_css},.7),0 0 25px {theme_config["primary"]},0 0 10px {theme_config["secondary"]}}}100%{{box-shadow:0 0 8px rgba({primary_rgb_css},.7),inset 0 0 8px rgba({primary_rgb_css},.5),0 0 15px {theme_config["primary"]}}}}}
@keyframes magicSparkle {{0%{{background-position:0% 0%}}50%{{background-position:100% 100%}}100%{{background-position:0% 0%}}}}
#video-background {{position:fixed;top:0;left:0;width:100vw;height:100vh;object-fit:cover;z-index:-2;opacity:.5}}
h1,h2 {{font-family:'Orbitron','Cinzel',sans-serif;color:{theme_config["primary"]};text-shadow:0 0 10px {theme_config["primary"]},0 0 20px {theme_config["primary"]},0 0 25px rgba(255,255,255,.4);letter-spacing:2px;position:relative}}
h1 {{font-size:2.8rem}} h2 {{font-size:2.2rem}}
h1:hover,h2:hover {{animation:titleGlow 1.5s infinite}}
@keyframes titleGlow {{0%{{text-shadow:0 0 10px {theme_config["primary"]},0 0 20px {theme_config["primary"]},0 0 25px rgba(255,255,255,.4)}}50%{{text-shadow:0 0 15px {theme_config["primary"]},0 0 30px {theme_config["primary"]},0 0 45px {theme_config["secondary"]},0 0 50px rgba(255,255,255,.6)}}100%{{text-shadow:0 0 10px {theme_config["primary"]},0 0 20px {theme_config["primary"]},0 0 25px rgba(255,255,255,.4)}}}}
h3 {{font-family:'MedievalSharp','Orbitron',cursive;color:{theme_config["secondary"]};text-shadow:0 0 8px {theme_config["secondary"]},0 0 12px rgba({secondary_rgb_css},.8);font-size:1.8rem}}
.stButton>button {{background:{theme_config["button_bg"]};border:2px solid transparent;border-radius:15px;color:#fff;font-family:'Orbitron','Cinzel',serif;font-weight:700;padding:15px 35px;font-size:18px;cursor:pointer;transition:all .4s ease;position:relative;z-index:1;text-shadow:0 0 5px #000;letter-spacing:1px;animation:neonBorderMove 4s ease-in-out infinite alternate,pulseGlow 2.5s infinite ease-in-out;background-origin:border-box;background-clip:padding-box,border-box}}
.stButton>button:hover {{background:{theme_config["button_hover_bg"]};transform:scale(1.12) rotate(-1.5deg);color:#fff;animation:none;box-shadow:0 0 35px 10px {theme_config["primary"]},0 0 20px {theme_config["secondary"]}}}
.stButton>button:hover:before {{content:"✦";position:absolute;top:-20px;left:-20px;font-size:22px;color:{theme_config["secondary"]};animation:floatingSymbol 2.5s infinite ease-in-out;text-shadow:0 0 6px {theme_config["secondary"]}}}
.stButton>button:hover:after {{content:"✧";position:absolute;bottom:-20px;right:-20px;font-size:22px;color:{theme_config["primary"]};animation:floatingSymbol 2.5s infinite ease-in-out reverse;text-shadow:0 0 6px {theme_config["primary"]}}}
@keyframes floatingSymbol {{0%{{transform:translate(0,0) rotate(0deg);opacity:.7}}50%{{transform:translate(5px,-5px) rotate(180deg);opacity:1}}100%{{transform:translate(0,0) rotate(360deg);opacity:.7}}}}
.stTextInput>div>input,.stSelectbox>div>div,.stMultiSelect>div>div,.stDateInput>div>div>input {{background-color:rgba(10,5,25,.8);color:{theme_config["text"]};border-radius:12px;border:2px solid {theme_config["primary"]};padding:12px;font-size:16px;font-family:'Orbitron','Cinzel',serif;transition:all .3s ease;box-shadow:0 0 7px rgba({primary_rgb_css},.6)}}
.stTextInput>div>input::placeholder {{color:rgba({primary_rgb_css},.7);font-family:'Cinzel',serif}}
.stTextInput>div>input:focus,.stSelectbox>div>div:focus-within,.stMultiSelect>div>div:focus-within,.stDateInput>div>div>input:focus {{border:2px solid {theme_config["secondary"]};box-shadow:0 0 15px {theme_config["primary"]},0 0 10px {theme_config["secondary"]};background-color:rgba(15,10,35,.9)}}
.st-emotion-cache-10oheav {{background-color:rgba(5,2,15,.97)!important;border:1px solid {theme_config["primary"]}!important;color:{theme_config["text"]}!important;font-family:'Orbitron','Cinzel',serif!important}}
.st-emotion-cache-trf2nb:hover {{background-color:rgba({primary_rgb_css},.35)!important;color:{theme_config["primary"]}!important}}
.css-1d391kg .sidebar .sidebar-content .css-1aumxhk {{color:{theme_config["primary"]}!important;text-shadow:0 0 7px {theme_config["primary"]};font-family:'Orbitron','Cinzel',sans-serif}}
.css-1d391kg .sidebar .sidebar-content .stRadio label span {{color:{theme_config["text"]}!important;font-size:1.1em;font-family:'Orbitron','Cinzel',sans-serif}}
.css-1d391kg .sidebar .sidebar-content .stRadio label span:hover {{color:{theme_config["primary"]}!important;text-shadow:0 0 5px {theme_config["primary"]}}}
@keyframes wand-wiggle{{0%{{transform:rotate(0deg)}}25%{{transform:rotate(18deg)}}50%{{transform:rotate(-18deg)}}75%{{transform:rotate(18deg)}}100%{{transform:rotate(0deg)}}}}
.wand-icon{{display:inline-block;animation:wand-wiggle 2.2s infinite ease-in-out;font-size:1.7rem;margin-left:10px;position:relative}}
.wand-icon:after{{content:'✦';position:absolute;color:{theme_config["primary"]};text-shadow:0 0 10px {theme_config["primary"]};border-radius:50%;top:-7px;right:-18px;animation:glow 1.3s infinite alternate;font-size:.9em}}
@keyframes glow{{0%{{opacity:.6;transform:scale(.9)}}100%{{opacity:1;transform:scale(1.3)}}}}
::-webkit-scrollbar{{width:14px}} ::-webkit-scrollbar-track{{background:rgba(5,2,15,.8);border-radius:10px;box-shadow:inset 0 0 6px rgba(0,0,0,.4)}} ::-webkit-scrollbar-thumb{{background:linear-gradient(60deg,{theme_config["primary"]},{theme_config["secondary"]});border-radius:10px;border:2px solid rgba(5,2,15,.8)}} ::-webkit-scrollbar-thumb:hover{{background:linear-gradient(60deg,{theme_config["secondary"]},{theme_config["primary"]});box-shadow:0 0 12px {theme_config["primary"]}}}
.tooltip{{position:relative;display:inline-block;cursor:help}}
.tooltip .tooltiptext{{visibility:hidden;width:240px;background-color:rgba(5,2,15,.97);color:{theme_config["text"]};text-align:center;border-radius:10px;padding:14px;position:absolute;z-index:100;bottom:135%;left:50%;margin-left:-120px;opacity:0;transition:opacity .4s,transform .4s;transform:translateY(12px);border:1px solid {theme_config["primary"]};box-shadow:0 0 15px rgba({primary_rgb_css},.8);font-size:.95em;font-family:'Cinzel',serif}}
.tooltip:hover .tooltiptext{{visibility:visible;opacity:1;transform:translateY(0)}}
.js-plotly-plot{{animation:floatingElement 10s infinite ease-in-out alternate;border-radius:18px;overflow:hidden}}
.dataframe{{font-family:'Cinzel',serif;border-collapse:separate;border-spacing:0;border-radius:15px;overflow:hidden;border:2px solid {theme_config["primary"]};background-color:rgba(10,5,25,.85);box-shadow:0 0 12px rgba({primary_rgb_css},.4);margin:1.2em 0}}
.dataframe th{{background:linear-gradient(45deg,{theme_config["primary"]},{theme_config["secondary"]});color:#fff;padding:16px;text-shadow:0 0 6px #000;font-size:1.15em;text-align:left;font-family:'Orbitron','Cinzel',serif}}
.dataframe td{{padding:14px;border-bottom:1px solid rgba({secondary_rgb_css},.25);color:{theme_config["text"]};font-size:1em}}
.dataframe tbody tr:hover{{background-color:rgba({primary_rgb_css},.2)}}
.dataframe tbody tr:nth-child(even){{background-color:rgba({primary_rgb_css},.08)}}
.dataframe tbody tr:nth-child(even):hover{{background-color:rgba({primary_rgb_css},.25)}}
@keyframes magicLoading{{0%{{transform:rotate(0deg);border-top-color:{theme_config["primary"]}}}25%{{border-top-color:{theme_config["secondary"]}}}50%{{transform:rotate(180deg);border-top-color:{theme_config["primary"]}}}75%{{border-top-color:{theme_config["secondary"]}}}100%{{transform:rotate(360deg);border-top-color:{theme_config["primary"]}}}}}
.loading-magic-container{{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:25px}}
.loading-magic{{display:inline-block;width:75px;height:75px;border:6px solid rgba({primary_rgb_css},.25);border-radius:50%;border-top-color:{theme_config["primary"]};animation:magicLoading 1.1s infinite linear;position:relative;margin-bottom:18px}}
.loading-magic:before{{content:'';position:absolute;top:6px;left:6px;right:6px;bottom:6px;border-radius:50%;border:5px solid transparent;border-top-color:{theme_config["secondary"]};animation:magicLoading 1.6s infinite linear reverse}}
.loading-magic-text{{color:{theme_config["text"]};font-family:'MedievalSharp','Orbitron',cursive;font-size:1.3em;text-shadow:0 0 6px {theme_config["primary"]}}}
@keyframes graphPulse{{0%{{box-shadow:0 0 5px rgba({primary_rgb_css},.5),0 0 10px rgba({secondary_rgb_css},.3)}}50%{{box-shadow:0 0 15px rgba({primary_rgb_css},.8),0 0 22px rgba({secondary_rgb_css},.6)}}100%{{box-shadow:0 0 5px rgba({primary_rgb_css},.5),0 0 10px rgba({secondary_rgb_css},.3)}}}}
.js-plotly-plot .plotly{{border-radius:18px;animation:graphPulse 3s infinite ease-in-out}}
.floating-avatar{{animation:floatingElement 6s infinite ease-in-out;border-radius:50%;box-shadow:0 0 18px {theme_config["primary"]},0 0 30px rgba({primary_rgb_css},.6);padding:6px;background-color:rgba({primary_rgb_css},.15)}}
@keyframes sparkle{{0%{{background-position:200% center}}100%{{background-position:-200% center}}}}
.sparkling-text{{background:linear-gradient(90deg,{theme_config["text"]},{theme_config["primary"]},{theme_config["secondary"]},{theme_config["primary"]},{theme_config["text"]});background-size:350% auto;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:sparkle 4s linear infinite;font-weight:700}}
@keyframes wizardHat{{0%{{transform:translateY(0) rotate(0deg)}}25%{{transform:translateY(-9px) rotate(-5deg)}}50%{{transform:translateY(0) rotate(0deg)}}75%{{transform:translateY(-9px) rotate(5deg)}}100%{{transform:translateY(0) rotate(0deg)}}}}
.wizard-hat{{display:inline-block;font-size:2rem;animation:wizardHat 3s infinite ease-in-out;margin-right:7px}}
hr{{border:0;height:3px;background-image:linear-gradient(to right,transparent,{theme_config["primary"]},{theme_config["secondary"]},{theme_config["primary"]},transparent);margin:2.2em 0;position:relative;box-shadow:0 0 7px {theme_config["primary"]}}}
hr:before{{content:'✨';position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);background:rgba(5,2,15,.95);padding:0 18px;color:{theme_config["secondary"]};font-size:1.6em;text-shadow:0 0 7px {theme_config["secondary"]}}}
.stProgress>div>div>div>div{{background:linear-gradient(90deg,{theme_config["primary"]},{theme_config["secondary"]});border-radius:10px}}
.stProgress>div>div>div{{background-color:rgba({secondary_rgb_css},.25);border-radius:10px}}
.plotly-graph-div .hovertext{{background-color:rgba(5,2,15,.97)!important;border:1px solid {theme_config["primary"]}!important;border-radius:10px!important;box-shadow:0 0 12px rgba({primary_rgb_css},.8)!important;color:{theme_config["text"]}!important;font-family:'Orbitron','Cinzel',serif!important}}
.plotly-graph-div .hovertext .nums,.plotly-graph-div .hovertext .name{{color:{theme_config["text"]}!important}}
@keyframes fadeInUp{{from{{opacity:0;transform:translate3d(0,30px,0)}}to{{opacity:1;transform:translate3d(0,0,0)}}}}
.animated-text{{opacity:0;animation:fadeInUp .9s forwards ease-out}}
.animated-text-delay-1{{animation-delay:.25s}} .animated-text-delay-2{{animation-delay:.5s}} .animated-text-delay-3{{animation-delay:.75s}}
.modal-content{{background-color:rgba(5,2,15,.98);border:2px solid {theme_config["primary"]};border-radius:18px;box-shadow:0 0 30px rgba({primary_rgb_css},.7);padding:30px;position:relative}}
.close-button{{position:absolute;top:18px;right:18px;cursor:pointer;font-size:30px;color:{theme_config["primary"]};transition:color .3s,transform .3s}}
.close-button:hover{{color:{theme_config["secondary"]};transform:rotate(180deg)}}
@keyframes glitch{{0%{{clip-path:inset(30% 0 71% 0);transform:translate(-2px,1px)}}10%{{clip-path:inset(10% 0 31% 0);transform:translate(1px,-1px);opacity:.9}}20%{{clip-path:inset(82% 0 1% 0);transform:translate(2px,2px)}}30%{{clip-path:inset(23% 0 41% 0);transform:translate(-1px,-2px);opacity:.85}}40%{{clip-path:inset(13% 0 1% 0);transform:translate(1px,1px);opacity:.9}}50%{{clip-path:inset(55% 0 28% 0);transform:translate(-2px,-1px)}}60%{{clip-path:inset(5% 0 57% 0);transform:translate(2px,1px);opacity:.85}}70%{{clip-path:inset(64% 0 7% 0);transform:translate(-1px,2px)}}80%{{clip-path:inset(38% 0 23% 0);transform:translate(1px,-2px);opacity:.9}}90%{{clip-path:inset(28% 0 43% 0);transform:translate(-2px,1px)}}100%{{clip-path:inset(50% 0 51% 0);transform:translate(0,0);opacity:1}}}}
.holodeck-container{{position:relative;overflow:hidden;border-radius:18px;padding:25px;background-color:rgba(5,2,15,.75);border:1px solid rgba({primary_rgb_css},.35);box-shadow:0 0 12px rgba({primary_rgb_css},.25);margin-bottom:1.8rem}}
.holodeck-container:after{{content:'';position:absolute;top:0;left:0;width:100%;height:100%;background:linear-gradient(135deg,rgba({primary_rgb_css},.06) 25%,transparent 25%,transparent 50%,rgba({primary_rgb_css},.06) 50%,rgba({primary_rgb_css},.06) 75%,transparent 75%,transparent 100%);background-size:7px 7px;opacity:.35;animation:glitch 7s infinite linear alternate-reverse;pointer-events:none;z-index:1}}
.holodeck-container > * {{position:relative;z-index:2;}}
.hologram{{position:relative;border-radius:12px;overflow:hidden;padding:12px;display:inline-block;background:radial-gradient(ellipse at center,rgba({primary_rgb_css},.18) 0%,rgba({primary_rgb_css},.07) 70%,transparent 100%)}}
.hologram:before{{content:'';position:absolute;top:-100%;left:0;width:100%;height:300%;background:repeating-linear-gradient(0deg,transparent,transparent 2.5px,rgba({primary_rgb_css},.25) 2.5px,rgba({primary_rgb_css},.25) 3.5px);animation:hologramLines 2.2s infinite linear;pointer-events:none;opacity:.75}}
@keyframes hologramLines{{0%{{transform:translateY(0%)}}100%{{transform:translateY(33.33%)}}}}
.hologram img{{opacity:.9;filter:drop-shadow(0 0 12px {theme_config["primary"]})}}
@keyframes appearsWithSparkles{{0%{{opacity:0;filter:blur(5px);transform:translateY(18px) scale(.93)}}70%{{opacity:.75;filter:blur(1.5px);transform:translateY(0) scale(1.02)}}100%{{opacity:1;filter:blur(0);transform:translateY(0) scale(1)}}}}
.spell-note{{animation:appearsWithSparkles .9s forwards ease-out;position:relative;padding:1.5rem;border-radius:15px;background-color:rgba(10,5,25,.9);border:1px solid {theme_config["primary"]};margin:1.2rem 0;box-shadow:0 0 12px rgba({primary_rgb_css},.35)}}
.spell-note:before{{content:"🔮";position:absolute;top:-14px;left:18px;font-size:24px;color:{theme_config["primary"]};background-color:rgba(10,5,25,.98);padding:0 6px;border-radius:50%;text-shadow:0 0 6px {theme_config["primary"]}}}
.stExpander{{border:1.5px solid {theme_config["primary"]};border-radius:12px;background-color:rgba(10,5,25,.75);margin-bottom:1.2rem}}
.stExpander header{{font-family:'MedievalSharp','Orbitron',cursive;font-size:1.3em;color:{theme_config["text"]}}}
.stExpander header:hover{{color:{theme_config["primary"]};text-shadow:0 0 5px {theme_config["primary"]}}}
.stExpander svg{{fill:{theme_config["primary"]};transform:scale(1.1)}}
.stTabs [data-baseweb=tab-list]{{gap:28px;border-bottom:2.5px solid {theme_config["primary"]}}}
.stTabs [data-baseweb=tab]{{height:55px;white-space:pre-wrap;background-color:transparent;font-family:'MedievalSharp','Orbitron',cursive;font-size:1.2em;color:{theme_config["text"]};padding-bottom:12px;transition:color .3s,background-color .3s}}
.stTabs [data-baseweb=tab]:hover{{background-color:rgba({primary_rgb_css},.15);color:{theme_config["primary"]}}}
.stTabs [aria-selected=true]{{color:{theme_config["primary"]};font-weight:700;text-shadow:0 0 7px {theme_config["primary"]};border-bottom:4px solid {theme_config["primary"]}}}
.stApp>header{{z-index:-3!important;background-color:transparent!important}}
.futuristic-text{{font-family:'Orbitron',sans-serif;letter-spacing:1px}}
.tc{{text-align:center}} .fs11{{font-size:1.1em}} .fs12{{font-size:1.2em}} .fs105{{font-size:1.05em}} .i{{font-style:italic}}
.ad{{animation:fadeInUp .9s forwards ease-out;opacity:0}} .mt15{{margin-top:15px}} .mt20{{margin-top:20px}}
.mt25{{margin-top:25px}} .gap10{{gap:10px}} .flex-around-wrap{{display:flex;justify-content:space-around;flex-wrap:wrap}}
.sum-item{{text-align:center;padding:10px;background:rgba({primary_rgb_css},.1);border-radius:8px}}
.sum-title{{color:{theme_config["text"]};font-size:.9em;margin-bottom:5px}}
.sum-value{{font-size:1.4rem;color:{theme_config["primary"]}}}
.snerr{{border-color:#ff4444!important}}
.curriculum-card{{height:320px;display:flex;flex-direction:column;justify-content:space-between}}
.fs15{{font-size:1.5em}} .fs09{{font-size:.9em}} .fs08{{font-size:.8em}}
.artifact-card{{height:380px;display:flex;flex-direction:column;justify-content:space-between}}
.artifact-img{{border-radius:10px;margin-bottom:10px;border:1px solid {theme_config["secondary"]};box-shadow:0 0 8px {theme_config["secondary"]}}}
.no-data-img{{width:150px;margin-top:15px;border-radius:10px;opacity:.7}}
.placeholder-gif{{width:200px;margin-top:20px;border-radius:10px;opacity:.8}}
</style>
"""
st.markdown(base_css_rules, unsafe_allow_html=True)

video_url_from_theme = theme_config.get("background_video", DEFAULT_HOGWARTS_VIDEO_URL)
is_video_placeholder = "YOUR_" in video_url_from_theme or video_url_from_theme == DEFAULT_HOGWARTS_VIDEO_URL

if video_url_from_theme and not is_video_placeholder:
    video_html_content = f'''<video autoplay muted loop id="video-background"><source src="{video_url_from_theme}" type="video/mp4"></video>'''
    st.markdown(video_html_content, unsafe_allow_html=True)

WAND = "🪄"; WIZARD_HAT_ICON = "🧙"; CRYSTAL_BALL = "🔮"; OWL = "🦉"; BROOM = "🧹"
POTION = "⚗️"; SPELL_BOOK = "📖"; STARS = "✨"; LIGHTNING = "⚡"; ROCKET = "🚀"
GEAR = "⚙️"; ATOM = "⚛️"

st.sidebar.markdown(f"### {SPELL_BOOK} Celestial Navigation")
page_options = ["Hogwarts Holo-Welcome", "Data Transmutation Chamber",
                "Predictive Enchantment Matrix", "Quantum Market Observatory"]
page_session_key = "selected_page_fm_v7" # New unique key

if page_session_key not in st.session_state:
    st.session_state[page_session_key] = page_options[0]

current_page_selection = st.sidebar.radio(
    "Select Your Destination:", options=page_options,
    index=page_options.index(st.session_state[page_session_key]),
    key="sidebar_page_selector_v7", # New unique key
    help="Navigate through the different chronomantic sections of the application."
)
if st.session_state[page_session_key] != current_page_selection:
    st.session_state[page_session_key] = current_page_selection
    st.experimental_rerun()

page = st.session_state[page_session_key]

if "df" not in st.session_state: st.session_state.df = None
if "ticker_data" not in st.session_state: st.session_state.ticker_data = None
if "spell_cast" not in st.session_state: st.session_state.spell_cast = False
if "user_name" not in st.session_state: st.session_state.user_name = ""
if "sorting_complete" not in st.session_state: st.session_state.sorting_complete = False

def magical_loading(message="Engaging Warp Drive..."):
    loading_html = f"""<div class="loading-magic-container"><div class="loading-magic"></div><div class="loading-magic-text">{message}</div></div>"""
    spinner_placeholder = st.empty(); spinner_placeholder.markdown(loading_html, unsafe_allow_html=True)
    time.sleep(2.5); spinner_placeholder.empty()

def load_data_from_upload():
    st.markdown(f"### {SPELL_BOOK} Upload Ancient Data Scrolls (CSV)")
    uploaded_file = st.file_uploader("Upload financial dataset (CSV)", type=["csv"], key="csv_up_expl_p5", help="...") # Unique key
    if uploaded_file:
        try:
            magical_loading("Decrypting data streams..."); data = pd.read_csv(uploaded_file); st.session_state.df = data
            st.success(f"{STARS} Scrolls decrypted! {STARS}")
            st.markdown("""<div class="spell-note ad"><p class="futuristic-text">Data stream decoded. Matrix ready.</p></div>""", unsafe_allow_html=True)
        except Exception as e: st.error(f"Decryption failed: {e}"); st.markdown(f"""<div class="spell-note ad snerr"><p class="futuristic-text">Error: Stream corrupted. Verify format.</p></div>""", unsafe_allow_html=True)

def load_data_from_stock():
    st.markdown(f"### {CRYSTAL_BALL} Summon Quantum Market Signatures (Stocks)")
    tc,sc,ec = st.columns(3)
    with tc: ticker = st.text_input("Quantum Signature (e.g., MSFT)", value="GOOG", key="tk_stk_p5", help="...") # Unique key
    with sc: start_date = st.date_input("Initial Chrono-Marker", pd.to_datetime("2023-01-01"), key="sd_stk_p5", help="...") # Unique key
    with ec: end_date = st.date_input("Final Chrono-Marker", pd.to_datetime("today"), key="ed_stk_p5", help="...") # Unique key
    if st.button(f"{ATOM} Summon Signatures", key="sum_stk_p5"): # Unique key
        if not ticker: st.warning("Enter Quantum Signature."); return
        try:
            magical_loading(f"Calibrating for {ticker}..."); data = yf.download(ticker,start_date,end_date)
            if data.empty: st.error(f"No sigs for '{ticker}'."); st.session_state.ticker_data=None; st.markdown(f"""<div class="spell-note ad snerr"><p class="futuristic-text">No signal. Try diff sig/params.</p></div>""", unsafe_allow_html=True)
            else:
                data.reset_index(inplace=True); st.session_state.ticker_data=data; st.success(f"{ROCKET} Sigs for {ticker} acquired! {ROCKET}")
                op,cp,hp,lp = data['Open'].iloc[0],data['Close'].iloc[-1],data['High'].max(),data['Low'].min()
                pc = ((cp-op)/op)*100 if op!=0 else 0; cc = theme_config["primary"] if pc > 0 else "#F44336"
                summary_html = f"""<div class="holodeck-container ad mt20"><h3 class="futuristic-text tc mb15" style="color:{theme_config['secondary']};">{ticker} Projection Summary {OWL}</h3><div class="flex-around-wrap gap10"><div class="sum-item"><h4 class="futuristic-text sum-title">Initiation</h4><p class="sum-value">${op:.2f}</p></div><div class="sum-item"><h4 class="futuristic-text sum-title">Termination</h4><p class="sum-value">${cp:.2f}</p></div><div class="sum-item"><h4 class="futuristic-text sum-title">Zenith</h4><p class="sum-value">${hp:.2f}</p></div><div class="sum-item"><h4 class="futuristic-text sum-title">Nadir</h4><p class="sum-value">${lp:.2f}</p></div><div class="sum-item"><h4 class="futuristic-text sum-title">Delta</h4><p class="sum-value" style="color:{cc};font-weight:bold;">{pc:.2f}%</p></div></div></div>"""
                st.markdown(summary_html, unsafe_allow_html=True)
        except Exception as e: st.error(f"Entanglement fail: {e}"); st.markdown(f"""<div class="spell-note ad snerr"><p class="futuristic-text">Analysis spell fail. Verify sig/net.</p></div>""", unsafe_allow_html=True)

def reset_data():
    if st.button(f"{WAND} Evanesco (Purge Data)", key="reset_dat_p5"): # Unique key
        magical_loading("Purging data streams..."); st.session_state.df=None; st.session_state.ticker_data=None
        st.success("Streams purged!"); time.sleep(1); st.experimental_rerun()

def welcome_page():
    cm, ca = st.columns([2,1])
    with cm:
        st.markdown(f"""<div class="ad"><h1 style="font-size:3rem;margin-bottom:.3rem;"><span class="wizard-hat">{WIZARD_HAT_ICON}</span>Welcome <br><span class="sparkling-text futuristic-text">Hogwarts Financial Mystics</span><span class="wand-icon">{ROCKET}</span></h1><p class="fs12 futuristic-text" style="color:{theme_config['secondary']};font-family:'Orbitron','MedievalSharp',cursive;">Initializing Holo-Interface... Ancient Wizardry Meets Quantum Dynamics!</p></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        cols_btn_center = st.columns([.5,2,.5]); 
        with cols_btn_center[1]:
            if st.button(f"{ROCKET} Dive into Financial Cosmos! {ATOM}", key="dive_welcome_p5", use_container_width=True): # Unique key
                if not st.session_state.get("sorting_complete",False) and not st.session_state.get("user_name",""): st.toast("Oracle awaits calibration!",icon="🎩")
                elif not st.session_state.get("sorting_complete",False): st.toast("Oracle assessment pending!",icon="🎩")
                else: st.session_state[page_session_key] = "Data Transmutation Chamber"; st.experimental_rerun()
        if not st.session_state.get("sorting_complete", False):
            st.markdown("""<div class="spell-note ad ad-delay-1 mt25"><p class="fs11 futuristic-text">Holo-Scan... New Chronomancer! Oracle must assess aptitude.</p></div>""", unsafe_allow_html=True)
            un_val = st.session_state.get("user_name","")
            un_input_val = st.text_input("Chronomancer designation:",key="wiz_name_welcome_p5",value=un_val,help="...") # Unique key
            if un_input_val: st.session_state.user_name = un_input_val
            if st.session_state.user_name:
                with st.expander(f"🌌 Oracle Calibration for {st.session_state.user_name}...", expanded=True):
                    st.markdown(f"""<p class="i futuristic-text" style="color:{theme_config['text']};">"Analyzing: {st.session_state.user_name}. Determining core frequency..."</p>""", unsafe_allow_html=True)
                    fq_val = st.selectbox("Chrono-crystal protocol:",["Boldly invest (Gryffindor)","Strategically multiply (Slytherin)","Sustainably grow (Hufflepuff)","Research application (Ravenclaw)"],index=None,key="sort_q_welcome_p5",placeholder="Select prime directive...",help="...") # Unique key
                    if st.button("Activate Oracle!",key="act_oracle_welcome_p5"): # Unique key
                        if fq_val:
                            magical_loading("Oracle Calibrating..."); sh_val = "Gryffindor" if "(G)" in fq_val else "Slytherin" if "(S)" in fq_val else "Hufflepuff" if "(H)" in fq_val else "Ravenclaw"
                            st.balloons();st.success(f"**Oracle Confirmed: {sh_val.upper()} ALIGNMENT!**")
                            st.markdown(f"""<div class="spell-note ad futuristic-text"><p>Calibrated, {st.session_state.user_name}! Matrix resonates with {sh_val}. Attune Holo-Interface or use universal field.</p></div>""", unsafe_allow_html=True)
                            st.session_state.sorting_complete=True
                        else:st.warning("Oracle needs input!")
        else: st.markdown(f"""<div class="spell-note ad ad-delay-1 mt25"><p class="fs12 futuristic-text">Welcome back, Chronomancer <strong style="color:{theme_config['primary']};">{st.session_state.user_name}</strong>! Ready for more temporal streams? {ROCKET}</p></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="ad ad-delay-2 futuristic-text mt20"><p class="fs105">Interface with enchanted algorithms, quantum models, data-driven chronomancy. Arcane echoes harmonize with tech symphonies for temporal keys to prosperity.</p></div>""", unsafe_allow_html=True)
    with ca:
        st.markdown("<br><br><br>",unsafe_allow_html=True);logo_d=theme_config.get("house_logo",HOGWARTS_CREST_URL);cap_t=f"Holo-Projection: {selected_house} Matrix" if selected_house!="None" else "Central Hogwarts Quantum Core"
        st.markdown(f"""<div class="hologram ad ad-delay-2 tc"><img src="{logo_d}" width="80%" class="floating-avatar" alt="{cap_t}"/></div><p class="tc i mt10 futuristic-text" style="color:{theme_config['secondary']};font-family:'Orbitron','MedievalSharp',cursive;">{cap_t}</p>""",unsafe_allow_html=True)
    st.markdown("<hr class='ad ad-delay-3'>",unsafe_allow_html=True)
    cfeat,cfort = st.columns(2)
    with cfeat:
        with st.expander(f"{ATOM} Advanced Holo-Features {ATOM}",expanded=False):st.markdown(f"""<div class="ad futuristic-text"><ul style="list-style-type:none;padding-left:0;"><li><span class="fs13">{SPELL_BOOK}</span> Data Scroll Decryption</li><li><span class="fs13">{GEAR}</span> Predictive AI</li><li><span class="fs13">{CRYSTAL_BALL}</span> Holo-Visualizations</li><li><span class="fs13">{ROCKET}</span> Personalized Field</li></ul></div>""",unsafe_allow_html=True)
        with st.expander(f"{OWL} Hogwarts Frequencies & Protocols {OWL}",expanded=False):st.markdown(f"""<div class="futuristic-text"><h4>Gryffindor🦁</h4><p>Bold investors.</p><h4>Slytherin🐍</h4><p>Ambitious.</p><h4>Hufflepuff🦡</h4><p>Sustainable.</p><h4>Ravenclaw🦅</h4><p>Analytical.</p></div>""",unsafe_allow_html=True)
    with cfort:
        st.markdown(f"""<div class="ad ad-delay-3"><h3 class="tc futuristic-text">Oracle Matrix Interface {CRYSTAL_BALL}</h3></div>""",unsafe_allow_html=True)
        with st.form("fortune_form_welcome_p5"):fi_val=st.text_input("Binary query to Oracle:",key="fortune_q_welcome_p5",placeholder="Investments achieve singularity?",help="...");csb_val=st.form_submit_button(f"{ATOM} Query Oracle") # Unique keys
        if csb_val:
            if fi_val:st.session_state.spell_cast=True;magical_loading("Oracle Processing...");fl=[{"r":"Affirmative.","c":"#4CAF50"},{"r":"Negative.","c":"#F44336"},{"r":"Indeterminate.","c":"#FF9800"},{"r":"Probable.","c":theme_config['primary']},{"r":"Uncertain.","c":theme_config['secondary']},{"r":"Deferred.","c":"#FF5722"}];cf_val=random.choice(fl);st.markdown(f"""<div class="spell-note ad futuristic-text tc" style="border-color:{cf_val['c']};"><h4 style="color:{cf_val['c']};">Oracle Response:</h4><p class="fs115" style="color:{theme_config['text']};">{cf_val['r']}</p><p class="i fs09" style="color:{theme_config['text']}b3;">*Disclaimer: Probabilistic.*</p></div>""",unsafe_allow_html=True)
            else:st.warning("Oracle needs query!")
    st.markdown("<hr class='ad'>",unsafe_allow_html=True)
    st.markdown("""<div class="ad"><h2 class="tc futuristic-text">Chronomancer Training Protocol {SPELL_BOOK}</h2><p class="tc fs105 futuristic-text">Master temporal financial dynamics:</p></div>""",unsafe_allow_html=True)
    cols_c=st.columns(4)
    cs_l=[
        {"t":"Data Influx","i":POTION,"d":"Interface CSVs/tickers."},
        {"t":"Holo-Charting","i":CRYSTAL_BALL,"d":"Weave visual algorithms."},
        {"t":"Predictive AI","i":GEAR,"d":"Deploy ML."},
        {"t":"Quantum Observatory","i":ATOM,"d":"Interpret real-time."}
    ]
    for i,s_val in enumerate(cs_l):
        with cols_c[i]: # Corrected
            st.markdown(f"""<div class="spell-note ad ad-delay-{i+1} futuristic-text curriculum-card"><div><h4 class="tc" style="color:{theme_config['primary']};"><span class="fs15">{s_val['i']}</span> {s_val['t']}</h4><p class="fs09">{s_val['d']}</p></div><p class="tc i fs08" style="color:{theme_config['secondary']};">Integrate protocol...</p></div>""",unsafe_allow_html=True)
    st.markdown("<hr class='ad'>",unsafe_allow_html=True)
    st.markdown("""<div class="ad"><h2 class="tc futuristic-text">Arsenal of Chronomantic Artifacts {LIGHTNING}</h2></div>""",unsafe_allow_html=True)
    cols_g=st.columns(3)
    art_l=[
        {"n":"Temporal Ledger","img":ARTIFACT_GIF_1_URL,"d":"Reveals historical waves."},
        {"n":"Quantum Abacus","img":ARTIFACT_GIF_2_URL,"d":"Transmutes data streams."},
        {"n":"Oracle Holo-Projector","img":ARTIFACT_GIF_3_URL,"d":"Simulates market futures."}
    ]
    for i,art_val in enumerate(art_l):
        with cols_g[i]: # Corrected
            st.markdown(f"""<div class="holodeck-container ad ad-delay-{i+1} futuristic-text artifact-card tc"><div><img src="{art_val['img']}" width="90%" class="artifact-img" alt="{art_val['n']}"/><h4 style="color:{theme_config['primary']};">{art_val['n']}</h4><p class="fs09">{art_val['d']}</p></div></div>""",unsafe_allow_html=True)

def data_exploration():
    st.markdown(f"""<div class="ad"><h1 class="futuristic-text tc">Data Transmutation Chamber {POTION}</h1><p class="futuristic-text tc fs11">Transmute raw data. Choose influx method.</p></div><hr class="ad">""", unsafe_allow_html=True)
    tab1,tab2=st.tabs([f"{SPELL_BOOK} CSV Scrolls",f"{ATOM} Stock Signatures"]) # Corrected
    with tab1: # Corrected
        load_data_from_upload()
    with tab2: # Corrected
        load_data_from_stock()
    st.markdown("<hr class='ad'>",unsafe_allow_html=True);reset_data()
    df,td=st.session_state.get('df'),st.session_state.get('ticker_data')
    if df is not None:
        st.markdown(f"""<h2 class="futuristic-text tc ad">Decrypted Glyphs</h2>""",unsafe_allow_html=True);st.dataframe(df.head(5))
        st.markdown(f"""<h3 class="futuristic-text tc ad">Quantum Insights</h3>""",unsafe_allow_html=True);cs,cst=st.columns(2)
        with cs:st.markdown(f"""<div class="spell-note ad futuristic-text"><h4>Stats Resonance</h4></div>""",unsafe_allow_html=True);st.dataframe(df.describe().T.head(3))
        with cst:st.markdown(f"""<div class="spell-note ad futuristic-text"><h4>Data Weave</h4></div>""",unsafe_allow_html=True);st.dataframe(pd.DataFrame({'F':df.columns,'T':df.dtypes.astype(str)}).head(3))
        st.markdown(f"""<h3 class="futuristic-text tc ad" id="chart-df">Holo-Viz</h3>""",unsafe_allow_html=True);num_c,cat_c=df.select_dtypes(np.number).columns.tolist(),df.select_dtypes('object').columns.tolist()
        if num_c:
            viz_s=st.selectbox("Viz Algo:",["Density","Correlation","Harmonics"],index=None,key="viz_df_p6",placeholder="Select..."); # Unique key
            if viz_s == "Density" and num_c: 
                sel_cols = st.multiselect("Select streams for Density Map:", num_c, default=num_c[:min(1,len(num_c))], key="hist_ms_p6") # Unique key
                if sel_cols:
                    for col_name in sel_cols:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(df[col_name].dropna(), kde=True, color=theme_config["primary"], ax=ax, bins=30, line_kws={'linewidth': 2.5, 'color': theme_config["secondary"]})
                        ax.set_title(f"Density Map of {col_name}", color=theme_config["text"], fontsize=16, fontfamily='Orbitron, Cinzel')
                        ax.set_xlabel(col_name, color=theme_config["text"], fontfamily='Orbitron, Cinzel')
                        ax.set_ylabel("Datapoint Frequency", color=theme_config["text"], fontfamily='Orbitron, Cinzel')
                        ax.set_facecolor('rgba(5,2,15,0.85)'); fig.patch.set_facecolor('rgba(5,2,15,0.85)')
                        ax.tick_params(axis='x', colors=theme_config["text"]); ax.tick_params(axis='y', colors=theme_config["text"])
                        ax.spines['bottom'].set_color(theme_config["primary"]); ax.spines['left'].set_color(theme_config["primary"])
                        ax.spines['top'].set_color('rgba(5,2,15,0.85)'); ax.spines['right'].set_color('rgba(5,2,15,0.85)')
                        plt.grid(axis='y', linestyle=':', alpha=0.4, color=theme_config["secondary"]); st.pyplot(fig)
                        st.markdown(f"""<div class="spell-note ad futuristic-text"><p><strong>Quantum Measures for {col_name}:</strong></p><ul><li>Mean: {df[col_name].mean():.2f}</li><li>Median: {df[col_name].median():.2f}</li><li>StdDev: {df[col_name].std():.2f}</li><li>Range: {df[col_name].min():.2f} to {df[col_name].max():.2f}</li></ul></div>""", unsafe_allow_html=True)
            elif viz_s == "Correlation" and len(num_c)>1:
                fig, ax = plt.subplots(figsize=(12, 10)); corr_matrix_df = df[num_c].corr(); mask = np.triu(np.ones_like(corr_matrix_df, dtype=bool))
                cmap = sns.diverging_palette(260,20,s=80,l=45,n=10,center="dark",as_cmap=True)
                sns.heatmap(corr_matrix_df,mask=mask,cmap=cmap,vmax=1,vmin=-1,center=0,annot=True,fmt=".2f",square=True,linewidths=.7,cbar_kws={"shrink":.85},ax=ax,annot_kws={"color":theme_config['text'],"fontfamily":"Orbitron, Cinzel"})
                ax.set_title("Correlation Nebula",color=theme_config["text"],fontsize=18,fontfamily='Orbitron, Cinzel')
                ax.set_facecolor('rgba(5,2,15,0.85)'); fig.patch.set_facecolor('rgba(5,2,15,0.85)')
                ax.tick_params(axis='x',colors=theme_config["text"],rotation=45); ax.tick_params(axis='y',colors=theme_config["text"])
                plt.xticks(fontfamily='Orbitron, Cinzel'); plt.yticks(fontfamily='Orbitron, Cinzel'); st.pyplot(fig)
            elif viz_s == "Harmonics" and cat_c and num_c: 
                sel_cat = st.selectbox("Select Category Stream:", cat_c, key="bar_cat_p6") # Unique key
                sel_num = st.selectbox("Select Value Stream:", num_c, key="bar_num_p6") # Unique key
                if sel_cat and sel_num:
                    fig, ax = plt.subplots(figsize=(12, 7))
                    sns.barplot(x=sel_cat,y=sel_num,data=df,ax=ax,palette=[theme_config["primary"],theme_config["secondary"],'#A076F9','#FDA7DF','#76D7C4'],estimator=np.mean,errorbar=None)
                    ax.set_title(f"Mean '{sel_num}' by '{sel_cat}' Harmonics",color=theme_config["text"],fontsize=16,fontfamily='Orbitron, Cinzel')
                    ax.set_xlabel(sel_cat,color=theme_config["text"],fontfamily='Orbitron, Cinzel'); ax.set_ylabel(f"Mean {sel_num}",color=theme_config["text"],fontfamily='Orbitron, Cinzel')
                    ax.set_facecolor('rgba(5,2,15,0.85)'); fig.patch.set_facecolor('rgba(5,2,15,0.85)')
                    ax.tick_params(axis='x',colors=theme_config["text"],rotation=45,ha="right"); ax.tick_params(axis='y',colors=theme_config["text"])
                    plt.grid(axis='y',linestyle=':',alpha=0.4,color=theme_config["secondary"]); st.pyplot(fig)
        else:st.info("No numeric data for Holo-Viz.")
    elif td is not None:
        st.markdown(f"""<h2 class="futuristic-text tc ad">Market Readings</h2>""",unsafe_allow_html=True);st.dataframe(td.head(5))
        st.markdown(f"""<h3 class="futuristic-text tc ad">Quantum Matrix Holo-Visions {ATOM}</h3>""",unsafe_allow_html=True)
        tabs_stock = st.tabs(["📈Price","📊Volume","🕯️Candles"]) # Use a different variable name for tabs here
        td['MA20']=td['Close'].rolling(20).mean();td['MA50']=td['Close'].rolling(50).mean()
        with tabs_stock[0]: # Price Journey Corrected
            fig_price=go.Figure();fig_price.add_trace(go.Scatter(x=td['Date'],y=td['Close'],mode='lines',name='Closing Price Vector',line=dict(color=theme_config["primary"],width=2.5,shape='spline'),hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'))
            if len(td)>=20:fig_price.add_trace(go.Scatter(x=td['Date'],y=td['MA20'],mode='lines',name='20-Cycle Trend',line=dict(color=theme_config["secondary"],width=1.5,dash='dash'),hovertemplate='<b>MA20</b>: $%{y:.2f}<extra></extra>'))
            if len(td)>=50:fig_price.add_trace(go.Scatter(x=td['Date'],y=td['MA50'],mode='lines',name='50-Cycle Resonance',line=dict(color='rgba(150,107,45,0.7)',width=1.5,dash='dot'),hovertemplate='<b>MA50</b>: $%{y:.2f}<extra></extra>'))
            fig_price.update_layout(title=dict(text=f"{td.columns[1] if len(td.columns)>1 else 'Stock'} Price Vector",x=0.5,font=dict(family="Orbitron,Cinzel",size=20,color=theme_config['primary'])),xaxis_title_text="Chrono-Marker",yaxis_title_text="Value (Credits)",plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font_color=theme_config["text"],font_family="Orbitron,Cinzel",hovermode="x unified",legend=dict(bgcolor='rgba(5,2,15,0.75)',bordercolor=theme_config["primary"],font=dict(family="Orbitron,Cinzel")),xaxis=dict(gridcolor=f'rgba({secondary_rgb_css},0.25)',showgrid=True,type='date',rangeselector=dict(buttons=list([dict(count=1,label="1M",step="month",stepmode="backward"),dict(count=6,label="6M",step="month",stepmode="backward"),dict(count=1,label="YTD",step="year",stepmode="todate"),dict(count=1,label="1Y",step="year",stepmode="backward"),dict(step="all",label="ALL")])),rangeslider=dict(visible=True,bgcolor=f'rgba({primary_rgb_css},0.15)')),yaxis=dict(gridcolor=f'rgba({secondary_rgb_css},0.25)',showgrid=True));st.plotly_chart(fig_price,use_container_width=True)
        with tabs_stock[1]: # Volume Corrected
            fig_volume=go.Figure();colors_v=[theme_config["primary"] if td['Close'][i]>=td['Open'][i] else '#E74C3C' for i in range(len(td))]
            fig_volume.add_trace(go.Bar(x=td['Date'],y=td['Volume'],name='Transaction Volume',marker_color=colors_v,opacity=0.75,hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Volume</b>: %{y:,}<extra></extra>'))
            fig_volume.update_layout(title=dict(text="Volume Runes - Market Energy Flux",x=0.5,font=dict(family="Orbitron,Cinzel",size=20,color=theme_config['primary'])),xaxis_title_text="Chrono-Marker",yaxis_title_text="Volume Units",plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font_color=theme_config["text"],font_family="Orbitron,Cinzel",hovermode="x unified",legend=dict(bgcolor='rgba(5,2,15,0.75)',bordercolor=theme_config["primary"]),xaxis=dict(gridcolor=f'rgba({secondary_rgb_css},0.25)',type='date'),yaxis=dict(gridcolor=f'rgba({secondary_rgb_css},0.25)'),bargap=0.25);st.plotly_chart(fig_volume,use_container_width=True)
        with tabs_stock[2]: # Candlestick Corrected
            fig_candle=go.Figure(data=[go.Candlestick(x=td['Date'],open=td['Open'],high=td['High'],low=td['Low'],close=td['Close'],increasing_line_color=theme_config['primary'],decreasing_line_color='#E74C3C',name="Price Vectors",hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>Open: $%{open:.2f}<br>High: $%{high:.2f}<br>Low: $%{low:.2f}<br>Close: $%{close:.2f}<extra></extra>')])
            if len(td)>=20:fig_candle.add_trace(go.Scatter(x=td['Date'],y=td['MA20'],mode='lines',name='20-Cycle Trend',line=dict(color=theme_config["secondary"],width=1.5,dash='dash'),opacity=0.75))
            if len(td)>=50:fig_candle.add_trace(go.Scatter(x=td['Date'],y=td['MA50'],mode='lines',name='50-Cycle Resonance',line=dict(color='rgba(150,107,45,0.75)',width=1.5,dash='dot'),opacity=0.75))
            fig_candle.update_layout(title=dict(text="Candlestick Chronomancy - Daily Signatures",x=0.5,font=dict(family="Orbitron,Cinzel",size=20,color=theme_config['primary'])),xaxis_title_text="Chrono-Marker",yaxis_title_text="Value (Credits)",plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font_color=theme_config["text"],font_family="Orbitron,Cinzel",hovermode="x unified",legend=dict(bgcolor='rgba(5,2,15,0.75)',bordercolor=theme_config["primary"]),xaxis_rangeslider_visible=False,xaxis=dict(gridcolor=f'rgba({secondary_rgb_css},0.25)',type='date'),yaxis=dict(gridcolor=f'rgba({secondary_rgb_css},0.25)'));st.plotly_chart(fig_candle,use_container_width=True)
    elif not df and not td:st.markdown(f"""<div class="spell-note ad futuristic-text tc"><p class="fs12">{SPELL_BOOK} Chamber awaits! {ATOM}</p><img src="{NO_DATA_IMAGE_URL}" alt="Awaiting Data" class="no-data-img"></div>""",unsafe_allow_html=True)

def machine_learning_spells():
    st.markdown(f"""<div class="ad"><h1 class="futuristic-text tc">Predictive Matrix {GEAR}</h1></div><hr class="ad">""", unsafe_allow_html=True)
    st.markdown(f"""<div class="spell-note ad futuristic-text tc"><h3 style="color:{theme_config['primary']};">Matrix Calibration {BROOM}</h3><p>Engineers calibrating algorithms.</p><img src="{CONSTRUCTION_GIF_URL}" alt="WIP" class="placeholder-gif"><p class="i mt15">Return when flux stable!</p></div>""",unsafe_allow_html=True)

def market_divination_observatory():
    st.markdown(f"""<div class="ad"><h1 class="futuristic-text tc">Quantum Observatory {OWL}</h1></div><hr class="ad">""", unsafe_allow_html=True)
    st.markdown(f"""<div class="spell-note ad futuristic-text tc"><h3 style="color:{theme_config['primary']};">Telescope Attunement {LIGHTNING}</h3><p>Astro-Quantomancer aligning lenses.</p><img src="{OBSERVATORY_GIF_URL}" alt="WIP" class="placeholder-gif"><p class="i mt15">Return when resonances optimal!</p></div>""",unsafe_allow_html=True)

if page == "Hogwarts Holo-Welcome": welcome_page()
elif page == "Data Transmutation Chamber": data_exploration()
elif page == "Predictive Enchantment Matrix": machine_learning_spells()
elif page == "Quantum Market Observatory": market_divination_observatory()

st.markdown("<hr class='ad'>",unsafe_allow_html=True)
st.markdown(f"""<p class="futuristic-text tc" style="font-family:'Orbitron','MedievalSharp',cursive;color:{theme_config['secondary']};font-size:.9em;">Engineered by Humble Chronomancer <span class="wand-icon">{ATOM}</span><br>May investments entangle with prosperity!</p>""",unsafe_allow_html=True)
