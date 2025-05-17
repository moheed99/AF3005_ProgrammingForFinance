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
import base64
from PIL import Image
import io
import time
import random

# Set page config
st.set_page_config(
    page_title="Harry Potter Financial Mystics - Futuristic Wizarding World",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hogwarts houses with enhanced futuristic magical themes
house_themes = {
    "None": {
        "wallpaper_url": "https://i.imgur.com/6hZ0Q1M.jpg",  # Hogwarts Castle - Hogwarts Legacy Style
        "primary": "#fedd00", # Gold
        "secondary": "#662d91", # Purple
        "text": "#eee7db", # Parchment
        "button_bg": "linear-gradient(45deg, #662d91, #fedd00)",
        "button_hover_bg": "linear-gradient(45deg, #fedd00, #662d91)",
        "house_logo": "https://i.imgur.com/Bt5Uyvw.png",  # Hogwarts crest
        "background_video": "https://i.imgur.com/xnTzwxu.mp4"
    },
    "Gryffindor": {
        "wallpaper_url": "https://i.imgur.com/9gD0x0g.jpg",  # Gryffindor Common Room - Hogwarts Legacy
        "primary": "#AE0001",
        "secondary": "#EEBA30",
        "text": "#fff2cc",
        "button_bg": "linear-gradient(45deg, #AE0001, #EEBA30)",
        "button_hover_bg": "linear-gradient(45deg, #EEBA30, #AE0001)",
        "house_logo": "https://i.imgur.com/nCU2QKo.png",
        "background_video": "https://i.imgur.com/nXEZb5S.mp4"
    },
    "Slytherin": {
        "wallpaper_url": "https://i.imgur.com/pwxpNrN.jpg",  # Slytherin Common Room - Hogwarts Legacy Underwater
        "primary": "#1A472A",
        "secondary": "#AAAAAA",
        "text": "#d0f0c0",
        "button_bg": "linear-gradient(45deg, #1A472A, #AAAAAA)",
        "button_hover_bg": "linear-gradient(45deg, #AAAAAA, #1A472A)",
        "house_logo": "https://i.imgur.com/DZ9tEb2.png",
        "background_video": "https://i.imgur.com/tWEp9VH.mp4"
    },
    "Hufflepuff": {
        "wallpaper_url": "https://i.imgur.com/B9Y4S9A.jpg",  # Hufflepuff Common Room - Hogwarts Legacy Cozy
        "primary": "#FFDB00",
        "secondary": "#372E29",
        "text": "#fff8e1",
        "button_bg": "linear-gradient(45deg, #372E29, #FFDB00)",
        "button_hover_bg": "linear-gradient(45deg, #FFDB00, #372E29)",
        "house_logo": "https://i.imgur.com/vQT68mS.png",
        "background_video": "https://i.imgur.com/Xqzc2OQ.mp4"
    },
    "Ravenclaw": {
        "wallpaper_url": "https://i.imgur.com/s6p0NKy.jpg",  # Ravenclaw Common Room - Hogwarts Legacy Starry
        "primary": "#0E1A40",
        "secondary": "#946B2D",
        "text": "#E2F1FF",
        "button_bg": "linear-gradient(45deg, #0E1A40, #946B2D)",
        "button_hover_bg": "linear-gradient(45deg, #946B2D, #0E1A40)",
        "house_logo": "https://i.imgur.com/RoTJCGM.png",
        "background_video": "https://i.imgur.com/C5UmTTK.mp4"
    }
}

# Select house theme from sidebar with enhanced UI
st.sidebar.title("Hogwarts Financial Divination")
selected_house = st.sidebar.selectbox(
    "Choose Your House",
    options=list(house_themes.keys()),
    format_func=lambda x: f"{x} House" if x != "None" else "Hogwarts (Default)"
)

theme = house_themes[selected_house]

# House-specific welcome messages
house_welcome_messages = {
    "None": "Welcome to the mystical realm of financial divination at Hogwarts!",
    "Gryffindor": "Brave Gryffindors! Channel your courage into financial mastery!",
    "Slytherin": "Ambitious Slytherins! Let your cunning guide your financial strategy!",
    "Hufflepuff": "Loyal Hufflepuffs! Your patience will yield financial rewards!",
    "Ravenclaw": "Wise Ravenclaws! Apply your intellect to financial mysteries!"
}

# Display house crest in sidebar
if selected_house != "None":
    st.sidebar.image(theme["house_logo"], use_column_width=True, caption=f"The Crest of {selected_house}")
    st.sidebar.markdown(f"<h3 style='text-align:center; color:{theme['primary']}; text-shadow: 0 0 5px {theme['secondary']};'><i>{house_welcome_messages[selected_house]}</i></h3>", unsafe_allow_html=True)
else:
    st.sidebar.image(theme["house_logo"], use_column_width=True, caption="Hogwarts School of Witchcraft and Wizardry")
    st.sidebar.markdown(f"<h3 style='text-align:center; color:{theme['primary']}; text-shadow: 0 0 5px {theme['secondary']};'><i>{house_welcome_messages[selected_house]}</i></h3>", unsafe_allow_html=True)


# Inject fonts & CSS with dynamic colors and magical animations
def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

remote_css("https://fonts.googleapis.com/css2?family=Creepster&display=swap")
remote_css("https://fonts.googleapis.com/css2?family=MedievalSharp&display=swap")
remote_css("https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap")
remote_css("https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap") # Futuristic font

# Function to convert hex to RGB for CSS
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

primary_rgb_tuple = hex_to_rgb(theme["primary"])
secondary_rgb_tuple = hex_to_rgb(theme["secondary"])

primary_rgb_css = f"{primary_rgb_tuple[0]}, {primary_rgb_tuple[1]}, {primary_rgb_tuple[2]}"
secondary_rgb_css = f"{secondary_rgb_tuple[0]}, {secondary_rgb_tuple[1]}, {secondary_rgb_tuple[2]}"


# Enhanced magical UI with floating elements and particle effects
background_css = f"""
<style>
@keyframes floatingElement {{
  0% {{ transform: translate(0, 0) rotate(0deg); }}
  25% {{ transform: translate(3px, -3px) rotate(0.5deg); }}
  50% {{ transform: translate(0, -5px) rotate(0deg); }}
  75% {{ transform: translate(-3px, -3px) rotate(-0.5deg); }}
  100% {{ transform: translate(0, 0) rotate(0deg); }}
}}

@keyframes neonBorderMove {{
  0% {{
    border-image-source: linear-gradient(45deg, {theme["primary"]}, {theme["secondary"]}, {theme["primary"]});
  }}
  50% {{
    border-image-source: linear-gradient(45deg, {theme["secondary"]}, {theme["primary"]}, {theme["secondary"]});
  }}
  100% {{
    border-image-source: linear-gradient(45deg, {theme["primary"]}, {theme["secondary"]}, {theme["primary"]});
  }}
}}

@keyframes pulseGlow {{
  0% {{
    box-shadow: 0 0 8px rgba({primary_rgb_css}, 0.7),
                inset 0 0 8px rgba({primary_rgb_css}, 0.5),
                0 0 15px {theme["primary"]};
  }}
  50% {{
    box-shadow: 0 0 12px rgba({primary_rgb_css}, 0.9),
                inset 0 0 12px rgba({primary_rgb_css}, 0.7),
                0 0 25px {theme["primary"]},
                0 0 10px {theme["secondary"]};
  }}
  100% {{
    box-shadow: 0 0 8px rgba({primary_rgb_css}, 0.7),
                inset 0 0 8px rgba({primary_rgb_css}, 0.5),
                0 0 15px {theme["primary"]};
  }}
}}


@keyframes magicSparkle {{
  0% {{ background-position: 0% 0%; opacity: 0.2; }}
  50% {{ background-position: 100% 100%; opacity: 0.4; }}
  100% {{ background-position: 0% 0%; opacity: 0.2; }}
}}

/* Main app styling */
.stApp {{
    background-image: url('{theme["wallpaper_url"]}'); /* USE STATIC WALLPAPER URL */
    background-size: cover;
    background-attachment: fixed;
    color: {theme["text"]};
    font-family: 'Cinzel', serif;
    margin: 0;
    padding: 0;
    overflow-x: hidden; /* Prevent horizontal scroll */
}}

#video-background {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw; /* Use viewport width */
    height: 100vh; /* Use viewport height */
    object-fit: cover;
    z-index: -2; /* Behind main background image */
    opacity: 0.5; /* Adjust opacity as needed, slightly more visible */
}}

/* Floating magical containers for main content blocks */
.main .block-container {{
  position: relative;
  background-color: rgba(5, 2, 15, 0.93); /* Darker, more mystical base */
  padding: 30px !important; /* Increased padding */
  border-radius: 30px; /* More rounded */
  border: 3px solid transparent;
  animation: neonBorderMove 5s ease-in-out infinite alternate, floatingElement 8s ease-in-out infinite alternate;
  background-image:
    linear-gradient(rgba(5, 2, 15, 0.93), rgba(5, 2, 15, 0.93)),
    linear-gradient(45deg, {theme["primary"]}, {theme["secondary"]}, {theme["primary"]});
  background-origin: border-box;
  background-clip: padding-box, border-box;
  backdrop-filter: blur(4px); /* Slightly more blur */
  box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.6); /* Deeper shadow */
  margin-top: 2.5rem; 
  margin-bottom: 2.5rem;
}}

.main .block-container:before {{ /* Sparkle overlay for main content */
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('https://i.imgur.com/WJIc0JL.gif'); /* Subtle sparkle gif */
  background-size: 250% 250%; 
  opacity: 0.1; 
  border-radius: 27px; /* Inside the border */
  z-index: -1;
  animation: magicSparkle 12s infinite linear;
  pointer-events: none;
}}


/* Futuristic titles with magical glow */
h1, h2 {{
    font-family: 'Orbitron', 'Cinzel', sans-serif; /* Added Orbitron for futuristic touch */
    color: {theme["primary"]};
    text-shadow:
      0 0 10px {theme["primary"]},
      0 0 20px {theme["primary"]},
      0 0 25px rgba(255,255,255,0.4);
    letter-spacing: 2px;
    position: relative;
}}

h1 {{ font-size: 2.8rem; }} /* Slightly larger */
h2 {{ font-size: 2.2rem; }}

/* Magical animations on hover for titles */
h1:hover, h2:hover {{
    animation: titleGlow 1.5s infinite;
}}

@keyframes titleGlow {{
  0% {{ text-shadow: 0 0 10px {theme["primary"]}, 0 0 20px {theme["primary"]}, 0 0 25px rgba(255,255,255,0.4); }}
  50% {{ text-shadow: 0 0 15px {theme["primary"]}, 0 0 30px {theme["primary"]}, 0 0 45px {theme["secondary"]}, 0 0 50px rgba(255,255,255,0.6); }}
  100% {{ text-shadow: 0 0 10px {theme["primary"]}, 0 0 20px {theme["primary"]}, 0 0 25px rgba(255,255,255,0.4); }}
}}

h3 {{
    font-family: 'MedievalSharp', 'Orbitron', cursive;
    color: {theme["secondary"]};
    text-shadow: 0 0 8px {theme["secondary"]}, 0 0 12px rgba({secondary_rgb_css}, 0.8);
    font-size: 1.8rem; /* Larger h3 */
}}

/* Enchanted buttons with magical hover effects */
.stButton>button {{
  background: {theme["button_bg"]};
  border: 2px solid transparent; 
  border-radius: 15px; /* Slightly more rounded */
  color: white;
  font-family: 'Orbitron', 'Cinzel', serif; /* Futuristic font for buttons */
  font-weight: bold;
  padding: 15px 35px; /* Larger buttons */
  font-size: 18px; /* Larger font */
  cursor: pointer;
  transition: all 0.4s ease;
  position: relative;
  z-index: 1;
  text-shadow: 0 0 5px black;
  letter-spacing: 1px;
  animation: neonBorderMove 4s ease-in-out infinite alternate, pulseGlow 2.5s infinite ease-in-out;
  background-origin: border-box;
  background-clip: padding-box, border-box;
}}

.stButton>button:hover {{
  background: {theme["button_hover_bg"]};
  box-shadow: 0 0 35px 10px {theme["primary"]}, 0 0 20px {theme["secondary"]}; /* More intense hover glow */
  transform: scale(1.12) rotate(-1.5deg); /* More pronounced hover effect */
  color: white;
  animation: none; /* Stop pulsing on hover to make hover effect more prominent */
}}

/* Add magical runes/symbols around the button on hover */
.stButton>button:hover:before {{
  content: "✦"; 
  position: absolute;
  top: -20px;
  left: -20px;
  font-size: 22px;
  color: {theme["secondary"]};
  animation: floatingSymbol 2.5s infinite ease-in-out;
  text-shadow: 0 0 6px {theme["secondary"]};
}}

.stButton>button:hover:after {{
  content: "✧"; 
  position: absolute;
  bottom: -20px;
  right: -20px;
  font-size: 22px;
  color: {theme["primary"]};
  animation: floatingSymbol 2.5s infinite ease-in-out reverse;
  text-shadow: 0 0 6px {theme["primary"]};
}}

@keyframes floatingSymbol {{
  0% {{ transform: translate(0, 0) rotate(0deg); opacity: 0.7; }}
  50% {{ transform: translate(5px, -5px) rotate(180deg); opacity: 1;}}
  100% {{ transform: translate(0, 0) rotate(360deg); opacity: 0.7; }}
}}

/* Magic glowing inputs */
.stTextInput>div>input, .stSelectbox>div>div, .stMultiSelect>div>div, .stDateInput>div>div>input {{
  background-color: rgba(10, 5, 25, 0.8); 
  color: {theme["text"]}; 
  border-radius: 12px; /* More rounded */
  border: 2px solid {theme["primary"]};
  padding: 12px; /* More padding */
  font-size: 16px; /* Larger font */
  font-family: 'Orbitron', 'Cinzel', serif;
  transition: all 0.3s ease;
  box-shadow: 0 0 7px rgba({primary_rgb_css}, 0.6);
}}
.stTextInput>div>input::placeholder {{
  color: rgba({primary_rgb_css}, 0.7);
  font-family: 'Cinzel', serif; /* Ensure placeholder has right font */
}}
.stTextInput>div>input:focus, .stSelectbox>div>div:focus-within, .stMultiSelect>div>div:focus-within, .stDateInput>div>div>input:focus {{
  border: 2px solid {theme["secondary"]};
  box-shadow: 0 0 15px {theme["primary"]}, 0 0 10px {theme["secondary"]};
  background-color: rgba(15, 10, 35, 0.9);
}}
.st-emotion-cache-10oheav {{ 
    background-color: rgba(5, 2, 15, 0.97) !important;
    border: 1px solid {theme["primary"]} !important;
    color: {theme["text"]} !important;
    font-family: 'Orbitron', 'Cinzel', serif !important;
}}
.st-emotion-cache-10oheav:hover {{
    background-color: rgba({primary_rgb_css}, 0.35) !important;
}}


/* Magical sidebar with floating particles */
.css-1d391kg .sidebar .sidebar-content {{ 
  background-image: url('{theme["wallpaper_url"]}'); /* USE STATIC WALLPAPER URL */
  background-size: cover;
  background-repeat: no-repeat;
  position: relative;
  overflow: hidden; 
  border-right: 3px solid {theme["primary"]}; /* Thicker border */
  box-shadow: 7px 0 20px rgba({primary_rgb_css}, 0.4); /* Stronger shadow */
}}

.css-1d391kg .sidebar .sidebar-content:before {{
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('https://i.imgur.com/aE3BnKy.gif'); 
  opacity: 0.12; /* Slightly more visible particles */
  z-index: 0; 
  pointer-events: none;
  animation: magicSparkle 10s infinite linear reverse; /* Varied animation */
}}
.css-1d391kg .sidebar .sidebar-content > div {{ 
    position: relative;
    z-index: 1;
}}

.css-1d391kg .sidebar .sidebar-content .css-1aumxhk {{ 
  color: {theme["primary"]} !important;
  text-shadow: 0 0 7px {theme["primary"]};
  font-family: 'Orbitron', 'Cinzel', sans-serif;
}}
.css-1d391kg .sidebar .sidebar-content .stRadio label span{{ 
    color: {theme["text"]} !important;
    font-size: 1.1em; /* Larger radio labels */
    font-family: 'Orbitron', 'Cinzel', sans-serif;
}}
.css-1d391kg .sidebar .sidebar-content .stRadio label span:hover{{
    color: {theme["primary"]} !important;
    text-shadow: 0 0 5px {theme["primary"]};
}}


/* Animated magic wand icon with glowing particles */
@keyframes wand-wiggle {{
  0% {{ transform: rotate(0deg); }}
  25% {{ transform: rotate(18deg); }}
  50% {{ transform: rotate(-18deg); }}
  75% {{ transform: rotate(18deg); }}
  100% {{ transform: rotate(0deg); }}
}}

.wand-icon {{
  display: inline-block;
  animation: wand-wiggle 2.2s infinite ease-in-out;
  font-size: 1.7rem;
  margin-left: 10px;
  position: relative;
}}

.wand-icon:after {{
  content: '✦';
  position: absolute;
  color: {theme["primary"]};
  text-shadow: 0 0 10px {theme["primary"]};
  border-radius: 50%;
  top: -7px;
  right: -18px;
  animation: glow 1.3s infinite alternate;
  font-size: 0.9em;
}}
@keyframes glow {{
    0% {{ opacity: 0.6; transform: scale(0.9); }}
    100% {{ opacity: 1; transform: scale(1.3); }}
}}

/* Magical scrollbar styles */
::-webkit-scrollbar {{
  width: 14px; /* Wider scrollbar */
}}
::-webkit-scrollbar-track {{
  background: rgba(5, 2, 15, 0.8);
  border-radius: 10px;
  box-shadow: inset 0 0 6px rgba(0,0,0,0.4);
}}
::-webkit-scrollbar-thumb {{
  background: linear-gradient(60deg, {theme["primary"]}, {theme["secondary"]});
  border-radius: 10px;
  border: 2px solid rgba(5, 2, 15, 0.8);
}}
::-webkit-scrollbar-thumb:hover {{
  background: linear-gradient(60deg, {theme["secondary"]}, {theme["primary"]});
  box-shadow: 0 0 12px {theme["primary"]};
}}

/* Magical tooltip style */
.tooltip {{
  position: relative;
  display: inline-block;
  cursor: help; 
}}

.tooltip .tooltiptext {{
  visibility: hidden;
  width: 240px; /* Wider tooltip */
  background-color: rgba(5, 2, 15, 0.97);
  color: {theme["text"]};
  text-align: center;
  border-radius: 10px;
  padding: 14px; /* More padding */
  position: absolute;
  z-index: 100; 
  bottom: 135%; 
  left: 50%;
  margin-left: -120px; 
  opacity: 0;
  transition: opacity 0.4s, transform 0.4s;
  transform: translateY(12px);
  border: 1px solid {theme["primary"]};
  box-shadow: 0 0 15px rgba({primary_rgb_css}, 0.8);
  font-size: 0.95em;
  font-family: 'Cinzel', serif; /* Ensure tooltip font */
}}

.tooltip:hover .tooltiptext {{
  visibility: visible;
  opacity: 1;
  transform: translateY(0);
}}

/* Dynamic floating graphs and charts */
.js-plotly-plot {{ 
  animation: floatingElement 10s infinite ease-in-out alternate; /* Slower, smoother float */
  border-radius: 18px; 
  overflow: hidden; 
}}

/* Magical table styles */
.dataframe {{
  font-family: 'Cinzel', serif;
  border-collapse: separate;
  border-spacing: 0;
  border-radius: 15px; /* More rounded */
  overflow: hidden;
  border: 2px solid {theme["primary"]};
  background-color: rgba(10, 5, 25, 0.85);
  box-shadow: 0 0 12px rgba({primary_rgb_css}, 0.4);
  margin: 1.2em 0; 
}}

.dataframe th {{
  background: linear-gradient(45deg, {theme["primary"]}, {theme["secondary"]});
  color: white;
  padding: 16px; /* More padding */
  text-shadow: 0 0 6px black;
  font-size: 1.15em;
  text-align: left;
  font-family: 'Orbitron', 'Cinzel', serif; /* Futuristic font for headers */
}}

.dataframe td {{
  padding: 14px; /* More padding */
  border-bottom: 1px solid rgba({secondary_rgb_css}, 0.25);
  color: {theme["text"]};
  font-size: 1em;
}}

.dataframe tbody tr:hover {{
  background-color: rgba({primary_rgb_css}, 0.2);
}}
.dataframe tbody tr:nth-child(even) {{
    background-color: rgba({primary_rgb_css}, 0.08);
}}
.dataframe tbody tr:nth-child(even):hover {{
    background-color: rgba({primary_rgb_css}, 0.25);
}}


/* Magical loading animation */
@keyframes magicLoading {{
  0% {{ transform: rotate(0deg); border-top-color: {theme["primary"]}; }}
  25% {{ border-top-color: {theme["secondary"]}; }}
  50% {{ transform: rotate(180deg); border-top-color: {theme["primary"]}; }}
  75% {{ border-top-color: {theme["secondary"]}; }}
  100% {{ transform: rotate(360deg); border-top-color: {theme["primary"]}; }}
}}

.loading-magic-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 25px;
}}

.loading-magic {{
  display: inline-block;
  width: 75px; /* Larger */
  height: 75px;
  border: 6px solid rgba({primary_rgb_css}, 0.25); 
  border-radius: 50%;
  border-top-color: {theme["primary"]};
  animation: magicLoading 1.1s infinite linear;
  position: relative;
  margin-bottom: 18px; 
}}

.loading-magic:before {{ 
  content: '';
  position: absolute;
  top: 6px; left: 6px; right: 6px; bottom: 6px; 
  border-radius: 50%;
  border: 5px solid transparent;
  border-top-color: {theme["secondary"]};
  animation: magicLoading 1.6s infinite linear reverse; 
}}
.loading-magic-text {{
    color: {theme["text"]};
    font-family: 'MedievalSharp', 'Orbitron', cursive;
    font-size: 1.3em; /* Larger text */
    text-shadow: 0 0 6px {theme["primary"]};
}}


/* Pulsing effect for plotly graphs */
@keyframes graphPulse {{
  0% {{ box-shadow: 0 0 5px rgba({primary_rgb_css},0.5), 0 0 10px rgba({secondary_rgb_css},0.3); }}
  50% {{ box-shadow: 0 0 15px rgba({primary_rgb_css},0.8), 0 0 22px rgba({secondary_rgb_css},0.6); }}
  100% {{ box-shadow: 0 0 5px rgba({primary_rgb_css},0.5), 0 0 10px rgba({secondary_rgb_css},0.3); }}
}}

.js-plotly-plot .plotly {{ 
  border-radius: 18px; 
  animation: graphPulse 3s infinite ease-in-out; /* Faster pulse */
}}

/* Floating avatar for the welcome page */
.floating-avatar {{
  animation: floatingElement 6s infinite ease-in-out; /* Slightly faster float */
  border-radius: 50%; 
  box-shadow: 0 0 18px {theme["primary"]}, 0 0 30px rgba({primary_rgb_css}, 0.6);
  padding: 6px; 
  background-color: rgba({primary_rgb_css}, 0.15); 
}}

/* Sparkling text effect */
@keyframes sparkle {{
  0% {{ background-position: 200% center; }} 
  100% {{ background-position: -200% center; }} 
}}

.sparkling-text {{
  background: linear-gradient(90deg, {theme["text"]}, {theme["primary"]}, {theme["secondary"]}, {theme["primary"]}, {theme["text"]});
  background-size: 350% auto; /* Wider gradient */
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: sparkle 4s linear infinite; /* Faster sparkle */
  font-weight: bold; 
}}

/* Animated wizard hat icon */
@keyframes wizardHat {{
  0% {{ transform: translateY(0) rotate(0deg); }}
  25% {{ transform: translateY(-9px) rotate(-5deg); }}
  50% {{ transform: translateY(0) rotate(0deg); }}
  75% {{ transform: translateY(-9px) rotate(5deg); }}
  100% {{ transform: translateY(0) rotate(0deg); }}
}}

.wizard-hat {{
  display: inline-block;
  font-size: 2rem; /* Larger hat */
  animation: wizardHat 3s infinite ease-in-out;
  margin-right: 7px; 
}}

/* Magical dividers */
hr {{
  border: 0;
  height: 3px; 
  background-image: linear-gradient(to right, rgba(0,0,0,0), {theme["primary"]}, {theme["secondary"]}, {theme["primary"]}, rgba(0,0,0,0));
  margin: 2.2em 0; 
  position: relative;
  box-shadow: 0 0 7px {theme["primary"]};
}}

hr:before {{
  content: '✨'; 
  position: absolute;
  left: 50%;
  top: 50%; 
  transform: translate(-50%, -50%);
  background: rgba(5, 2, 15, 0.95); 
  padding: 0 18px;
  color: {theme["secondary"]};
  font-size: 1.6em;
  text-shadow: 0 0 7px {theme["secondary"]};
}}

/* Magical progress bars */
.stProgress > div > div > div > div {{
  background: linear-gradient(90deg, {theme["primary"]}, {theme["secondary"]}); 
  border-radius: 10px; 
}}

.stProgress > div > div > div {{
  background-color: rgba({secondary_rgb_css}, 0.25); 
  border-radius: 10px; /* Rounded track */
}}

/* Chart hover tooltip effect for Plotly */
.plotly-graph-div .hovertext {{ 
  background-color: rgba(5, 2, 15, 0.97) !important;
  border: 1px solid {theme["primary"]} !important;
  border-radius: 10px !important; /* More rounded */
  box-shadow: 0 0 12px rgba({primary_rgb_css}, 0.8) !important;
  color: {theme["text"]} !important; 
  font-family: 'Orbitron', 'Cinzel', serif !important; /* Futuristic font */
}}
.plotly-graph-div .hovertext .nums, .plotly-graph-div .hovertext .name {{
    color: {theme["text"]} !important;
}}


/* Animated text effect */
@keyframes fadeInUp {{
  from {{
    opacity: 0;
    transform: translate3d(0, 30px, 0); /* More pronounced entry */
  }}
  to {{
    opacity: 1;
    transform: translate3d(0, 0, 0);
  }}
}}

.animated-text {{
  opacity: 0; 
  animation: fadeInUp 0.9s forwards ease-out; /* Slightly slower animation */
}}
.animated-text-delay-1 {{ animation-delay: 0.25s; }}
.animated-text-delay-2 {{ animation-delay: 0.5s; }}
.animated-text-delay-3 {{ animation-delay: 0.75s; }}


/* Modal styles (Placeholder) */
.modal-content {{
  background-color: rgba(5, 2, 15, 0.98);
  border: 2px solid {theme["primary"]};
  border-radius: 18px;
  box-shadow: 0 0 30px rgba({primary_rgb_css}, 0.7);
  padding: 30px;
  position: relative; 
}}

.close-button {{
  position: absolute;
  top: 18px;
  right: 18px;
  cursor: pointer;
  font-size: 30px;
  color: {theme["primary"]};
  transition: color 0.3s, transform 0.3s;
}}
.close-button:hover {{
    color: {theme["secondary"]};
    transform: rotate(180deg); /* More dramatic rotation */
}}

/* Glitching effect for futuristic holodecks */
@keyframes glitch {{
  0% {{ clip-path: inset(30% 0 71% 0); transform: translate(-2px, 1px); }}
  10% {{ clip-path: inset(10% 0 31% 0); transform: translate(1px, -1px); opacity: 0.9; }}
  20% {{ clip-path: inset(82% 0 1% 0); transform: translate(2px, 2px); }}
  30% {{ clip-path: inset(23% 0 41% 0); transform: translate(-1px, -2px); opacity: 0.85; }}
  /* ... (keep the rest of the glitch keyframes) ... */
  100% {{ clip-path: inset(50% 0 51% 0); transform: translate(0,0); opacity: 1; }}
}}

.holodeck-container {{
  position: relative;
  overflow: hidden;
  border-radius: 18px; /* More rounded */
  padding: 25px; /* More padding */
  background-color: rgba(5, 2, 15, 0.75);
  border: 1px solid rgba({primary_rgb_css}, 0.35);
  box-shadow: 0 0 12px rgba({primary_rgb_css}, 0.25);
  margin-bottom: 1.8rem; 
}}

.holodeck-container:after {{
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, rgba({primary_rgb_css},0.06) 25%, transparent 25%, transparent 50%, rgba({primary_rgb_css},0.06) 50%, rgba({primary_rgb_css},0.06) 75%, transparent 75%, transparent 100%);
  background-size: 7px 7px; 
  opacity: 0.35; /* Slightly more visible */
  animation: glitch 7s infinite linear alternate-reverse; /* Slower glitch */
  pointer-events: none;
  z-index: 1; 
}}
.holodeck-container > * {{ 
    position: relative;
    z-index: 2;
}}

/* New hologram projection effect */
.hologram {{
  position: relative;
  border-radius: 12px; 
  overflow: hidden; 
  padding: 12px;
  display: inline-block; 
  background: radial-gradient(ellipse at center, rgba({primary_rgb_css},0.18) 0%, rgba({primary_rgb_css},0.07) 70%, transparent 100%);
}}

.hologram:before {{
  content: '';
  position: absolute;
  top: -100%; 
  left: 0;
  width: 100%;
  height: 300%; 
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2.5px, 
    rgba({primary_rgb_css}, 0.25) 2.5px, 
    rgba({primary_rgb_css}, 0.25) 3.5px  
  );
  animation: hologramLines 2.2s infinite linear; /* Faster lines */
  pointer-events: none;
  opacity: 0.75; /* More visible lines */
}}

@keyframes hologramLines {{
  0% {{ transform: translateY(0%); }}
  100% {{ transform: translateY(33.33%); }} 
}}
.hologram img {{ 
    opacity: 0.9; /* More opaque image */
    filter: drop-shadow(0 0 12px {theme["primary"]});
}}


/* Spell notes appearing effect */
@keyframes appearsWithSparkles {{
  0% {{ opacity: 0; filter: blur(5px); transform: translateY(18px) scale(0.93); }}
  70% {{ opacity: 0.75; filter: blur(1.5px); transform: translateY(0) scale(1.02);}}
  100% {{ opacity: 1; filter: blur(0); transform: translateY(0) scale(1);}}
}}

.spell-note {{
  animation: appearsWithSparkles 0.9s forwards ease-out; /* Slower appearance */
  position: relative;
  padding: 1.5rem; 
  border-radius: 15px; /* More rounded */
  background-color: rgba(10, 5, 25, 0.9); 
  border: 1px solid {theme["primary"]};
  margin: 1.2rem 0;
  box-shadow: 0 0 12px rgba({primary_rgb_css}, 0.35);
}}

.spell-note:before {{
  content: "🔮"; 
  position: absolute;
  top: -14px; 
  left: 18px;
  font-size: 24px; 
  color: {theme["primary"]};
  background-color: rgba(10, 5, 25, 0.98); 
  padding: 0 6px;
  border-radius: 50%;
  text-shadow: 0 0 6px {theme["primary"]};
}}

/* Expander styling */
.stExpander {{
    border: 1.5px solid {theme["primary"]}; /* Slightly thicker border */
    border-radius: 12px;
    background-color: rgba(10, 5, 25, 0.75);
    margin-bottom: 1.2rem;
}}
.stExpander header {{
    font-family: 'MedievalSharp', 'Orbitron', cursive;
    font-size: 1.3em; /* Larger expander title */
    color: {theme["text"]};
}}
.stExpander header:hover {{
    color: {theme["primary"]};
    text-shadow: 0 0 5px {theme["primary"]};
}}
.stExpander svg {{ 
    fill: {theme["primary"]};
    transform: scale(1.1); /* Larger arrow */
}}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {{
		gap: 28px; /* More gap */
        border-bottom: 2.5px solid {theme["primary"]}; /* Thicker border */
}}
.stTabs [data-baseweb="tab"] {{
		height: 55px; /* Taller tabs */
        white-space: pre-wrap;
		background-color: transparent;
        font-family: 'MedievalSharp', 'Orbitron', cursive;
        font-size: 1.2em; /* Larger tab font */
        color: {theme["text"]};
        padding-bottom: 12px; 
        transition: color 0.3s, background-color 0.3s;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background-color: rgba({primary_rgb_css}, 0.15);
    color: {theme["primary"]};
}}
.stTabs [aria-selected="true"] {{
    color: {theme["primary"]};
    font-weight: bold;
    text-shadow: 0 0 7px {theme["primary"]};
    border-bottom: 4px solid {theme["primary"]}; 
}}

.stApp > header {{
    z-index: -3 !important; 
    background-color: transparent !important;
}}

/* Futuristic Font for specific texts if needed */
.futuristic-text {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 1px;
    text-transform: uppercase; /* Optional: for a more techy feel */
}

</style>
"""

st.markdown(background_css, unsafe_allow_html=True)

# Add ambient background video based on selected house
if selected_house != "":
    video_html = f'''
    <video autoplay muted loop id="video-background">
      <source src="{theme["background_video"]}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    '''
    st.markdown(video_html, unsafe_allow_html=True)

# Magical wand and wizarding icons
WAND = "🪄"
WIZARD_HAT_ICON = "🧙"
CRYSTAL_BALL = "🔮"
OWL = "🦉"
BROOM = "🧹"
POTION = "⚗️"
SPELL_BOOK = "📖"
STARS = "✨"
LIGHTNING = "⚡"
ROCKET = "🚀" # For futuristic touch
GEAR = "⚙️" # For tech/ML
ATOM = "⚛️" # For quantum/advanced feel

# Sidebar page navigation with enhanced magical UI
st.sidebar.markdown(f"### {SPELL_BOOK} Celestial Navigation") # Changed title
page_options = ["Hogwarts Holo-Welcome", "Data Transmutation Chamber", # Renamed pages
                "Predictive Enchantment Matrix", "Quantum Market Observatory"]
page_labels_in_session = "selected_page_label_hp_fm" # unique key for session state

if page_labels_in_session not in st.session_state:
    st.session_state[page_labels_in_session] = page_options[0]


# Capture current page from radio button for direct use
current_page_selection = st.sidebar.radio(
    "Select Your Destination:",
    options=page_options,
    index=page_options.index(st.session_state[page_labels_in_session]), # Use session state for index
    help="Navigate through the different chronomantic sections of the application."
)
# Update session state if selection changes
if st.session_state[page_labels_in_session] != current_page_selection:
    st.session_state[page_labels_in_session] = current_page_selection
    # No automatic rerun needed here, page change logic below will handle it if button click caused it.

# This variable `page` will be used for routing logic.
page = st.session_state[page_labels_in_session]


# Initialize session state variables
if "df" not in st.session_state:
    st.session_state.df = None
if "ticker_data" not in st.session_state:
    st.session_state.ticker_data = None
if "spell_cast" not in st.session_state:
    st.session_state.spell_cast = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "sorting_complete" not in st.session_state:
    st.session_state.sorting_complete = False

# Function to display magical loading animation
def magical_loading(message="Engaging Warp Drive..."): # Updated message
    loading_html = f"""
    <div class="loading-magic-container">
        <div class="loading-magic"></div>
        <div class="loading-magic-text">{message}</div>
    </div>
    """
    # Use st.empty to show and then clear the loading animation
    # This needs to be handled carefully if other elements are rendered immediately after
    # For now, let's assume it's used where it can occupy space temporarily
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown(loading_html, unsafe_allow_html=True)
    time.sleep(2.5) # Simulate spell casting time
    spinner_placeholder.empty() # Clear the loading animation

# Function to load data from file upload with magical effects
def load_data_from_upload():
    st.markdown(f"### {SPELL_BOOK} Upload Ancient Data Scrolls (CSV)") # Updated title
    uploaded_file = st.file_uploader("Upload your financial dataset (CSV)", type=["csv"],
                                     help="Provide a CSV file (digital scroll) containing your financial data.")

    if uploaded_file:
        try:
            magical_loading("Decrypting ancient data streams...")
            data = pd.read_csv(uploaded_file)
            st.session_state.df = data
            st.success(f"{STARS} Data scrolls successfully decrypted and integrated! {STARS}")

            st.markdown("""
            <div class="spell-note animated-text">
                <p class="futuristic-text">Data stream decoded. Financial matrix ready for analysis and chronomantic projection.</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Data scroll decryption failed: {e}")
            st.markdown(f"""
            <div class="spell-note animated-text" style="border-color: #ff4444;">
                <p class="futuristic-text">Error: Data stream corrupted. Unable to parse digital scroll. Please verify format (CSV integrity) and try recalibration.</p>
            </div>
            """, unsafe_allow_html=True)

# Function to load data from stock with magical effects
def load_data_from_stock():
    st.markdown(f"### {CRYSTAL_BALL} Summon Quantum Market Signatures (Stocks)") # Updated title
    ticker_col, start_col, end_col = st.columns(3)

    with ticker_col:
        ticker = st.text_input("Enter Quantum Signature (Stock Symbol, e.g., QNTM)", value="MSFT", key="ticker_input_stock",
                              help="The quantum entanglement signature (stock symbol) of the entity for analysis.")
    with start_col:
        start_date = st.date_input("Initial Chrono-Marker", pd.to_datetime("2023-01-01"), key="start_date_input_stock",
                                   help="The starting temporal marker for your quantum analysis.")
    with end_col:
        end_date = st.date_input("Final Chrono-Marker", pd.to_datetime("today"), key="end_date_input_stock",
                                 help="The ending temporal marker for your quantum analysis.")

    if st.button(f"{ATOM} Summon Market Signatures"): # Updated icon
        if not ticker:
            st.warning("Please enter a Quantum Signature to summon data.")
            return
        try:
            magical_loading(f"Calibrating chronometers for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error(f"No quantum signatures found for '{ticker}' in the specified temporal window.")
                st.markdown(f"""
                <div class="spell-note animated-text" style="border-color: #ff4444;">
                    <p class="futuristic-text">The chronoscope shows no signal for this signature. Try a different quantum signature or adjust temporal parameters.</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.ticker_data = None
            else:
                data.reset_index(inplace=True)
                st.session_state.ticker_data = data
                st.success(f"{ROCKET} Quantum market signatures for {ticker} successfully acquired and materialized! {ROCKET}")

                open_price = data['Open'].iloc[0]
                close_price = data['Close'].iloc[-1]
                high_price = data['High'].max()
                low_price = data['Low'].min()
                percent_change = ((close_price - open_price) / open_price) * 100 if open_price != 0 else 0
                change_color = theme["primary"] if percent_change > 0 else "#F44336"

                st.markdown(f"""
                <div class="holodeck-container animated-text" style="margin-top: 20px;">
                    <h3 style="text-align: center; margin-bottom: 15px; color: {theme['secondary']};" class="futuristic-text">{ticker} Quantum Projection Summary {OWL}</h3>
                    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px;">
                        <div style="text-align: center; padding: 10px; background: rgba({primary_rgb_css}, 0.1); border-radius: 8px;">
                            <h4 style="color:{theme['text']}; font-size:0.9em; margin-bottom:5px;" class="futuristic-text">Initiation Value</h4>
                            <p style="font-size: 1.4rem; color:{theme['primary']};">${open_price:.2f}</p>
                        </div>
                        <div style="text-align: center; padding: 10px; background: rgba({primary_rgb_css}, 0.1); border-radius: 8px;">
                            <h4 style="color:{theme['text']}; font-size:0.9em; margin-bottom:5px;" class="futuristic-text">Termination Value</h4>
                            <p style="font-size: 1.4rem; color:{theme['primary']};">${close_price:.2f}</p>
                        </div>
                        <div style="text-align: center; padding: 10px; background: rgba({primary_rgb_css}, 0.1); border-radius: 8px;">
                            <h4 style="color:{theme['text']}; font-size:0.9em; margin-bottom:5px;" class="futuristic-text">Zenith Point</h4>
                            <p style="font-size: 1.4rem; color:{theme['primary']};">${high_price:.2f}</p>
                        </div>
                         <div style="text-align: center; padding: 10px; background: rgba({primary_rgb_css}, 0.1); border-radius: 8px;">
                            <h4 style="color:{theme['text']}; font-size:0.9em; margin-bottom:5px;" class="futuristic-text">Nadir Point</h4>
                            <p style="font-size: 1.4rem; color:{theme['primary']};">${low_price:.2f}</p>
                        </div>
                        <div style="text-align: center; padding: 10px; background: rgba({primary_rgb_css}, 0.1); border-radius: 8px;">
                            <h4 style="color:{theme['text']}; font-size:0.9em; margin-bottom:5px;" class="futuristic-text">Delta Shift</h4>
                            <p style="font-size: 1.4rem; color: {change_color}; font-weight: bold;">{percent_change:.2f}%</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Quantum entanglement failed: {e}")
            st.markdown(f"""
            <div class="spell-note animated-text" style="border-color: #ff4444;">
                <p class="futuristic-text">Error: Chronomantic analysis spell encountered interference. Please verify the quantum signature and network stability with the Celestial Cartographers Guild.</p>
            </div>
            """, unsafe_allow_html=True)

# Function to reset data with magical effects
def reset_data():
    if st.button(f"{WAND} Evanesco (Purge Data Cache)"): # Updated text
        magical_loading("Purging temporal data streams...")
        st.session_state.df = None
        st.session_state.ticker_data = None
        st.success("All data streams have been successfully purged from the chronometers!")
        time.sleep(1)
        st.experimental_rerun()

# Function for the welcome page with interactive magical elements
def welcome_page():
    col_main, col_art = st.columns([2,1])

    with col_main:
        st.markdown(f"""
        <div class="animated-text">
            <h1 style="font-size: 3rem; margin-bottom: 0.3rem;"> 
                <span class="wizard-hat">{WIZARD_HAT_ICON}</span>Welcome to the <br><span class="sparkling-text futuristic-text">Hogwarts Financial Mystics</span><span class="wand-icon">{ROCKET}</span>
            </h1>
            <p style="font-size: 1.2rem; color: {theme['secondary']}; font-family: 'Orbitron', 'MedievalSharp', cursive;" class="futuristic-text">
                Initializing Holo-Interface... Where Ancient Wizardry Meets Quantum Financial Dynamics!
            </p>
        </div>
        """, unsafe_allow_html=True)

        # "Dive into Hogwarts" Button
        st.markdown("<br>", unsafe_allow_html=True)
        cols_button_center = st.columns([0.5, 2, 0.5]) # Adjust column ratios for centering
        with cols_button_center[1]:
            if st.button(f"{ROCKET} Dive into the Financial Cosmos of Hogwarts! {ATOM}", key="dive_in_hogwarts_button_main", use_container_width=True):
                if not st.session_state.sorting_complete and not st.session_state.user_name: # If user hasn't even entered name
                    st.toast("The Sorting Hat must first calibrate your chrono-signature, young adept!", icon="🎩")
                elif not st.session_state.sorting_complete: # If name entered, but not sorted
                     st.toast("The Sorting Hat's assessment is pending. Please complete your initiation!", icon="🎩")
                else:
                    st.session_state[page_labels_in_session] = "Data Transmutation Chamber" # Navigate
                    st.experimental_rerun()


        if not st.session_state.sorting_complete:
            st.markdown("""
            <div class="spell-note animated-text animated-text-delay-1" style="margin-top:25px;">
                <p style="font-size: 1.1rem;" class="futuristic-text">Initiate Holo-Scan... A new Chronomancer detected! Before you access the deeper financial matrices, the Sorting Oracle must assess your quantum aptitude.</p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.user_name = st.text_input("Enter your Chronomancer designation (wizarding name):",
                                                        key="wizard_name_input_welcome_page", # Unique key
                                                        help="Your chosen callsign for this temporal-financial journey.",
                                                        value=st.session_state.user_name) # Persist value

            if st.session_state.user_name:
                with st.expander(f"🌌 Sorting Oracle Calibration for {st.session_state.user_name}...", expanded=True):
                    st.markdown(f"""
                    <p style="font-style: italic; color: {theme['text']};" class="futuristic-text">"Analyzing quantum signature of: {st.session_state.user_name}. To determine your core financial frequency..."</p>
                    """, unsafe_allow_html=True)

                    financial_question = st.selectbox(
                        "When faced with a newly discovered chrono-crystal (source of immense financial energy), your primary protocol is to:",
                        [
                            "Boldly channel its energy into a high-risk, high-reward temporal investment (Gryffindor Protocol)",
                            "Devise a multi-layered strategy to amplify its power through shrewd market manipulations (Slytherin Protocol)",
                            "Establish a stable energy conduit, ensuring sustainable growth and fair distribution (Hufflepuff Protocol)",
                            "Interface with ancient data archives to understand its optimal, knowledge-driven application (Ravenclaw Protocol)"
                        ],
                        index=None,
                        placeholder="Select your prime directive...",
                        help="This calibrates the Oracle to your financial wavelength."
                    )

                    if st.button("Activate Sorting Oracle!"):
                        if financial_question:
                            magical_loading("Oracle Calibrating Quantum Frequencies...")
                            if "Gryffindor" in financial_question: suggested_house = "Gryffindor"
                            elif "Slytherin" in financial_question: suggested_house = "Slytherin"
                            elif "Hufflepuff" in financial_question: suggested_house = "Hufflepuff"
                            else: suggested_house = "Ravenclaw"

                            st.balloons()
                            st.success(f"**Sorting Oracle Matrix Confirmed: {suggested_house.upper()} ALIGNMENT!**")
                            st.markdown(f"""
                            <div class="spell-note animated-text futuristic-text">
                                <p>Calibration complete, Chronomancer {st.session_state.user_name}! Your financial matrix resonates with the {suggested_house} frequency.
                                You may now attune the Holo-Interface to '{suggested_house} House' via the Celestial Navigation panel for a personalized chronomantic experience, or proceed with the universal Hogwarts quantum field.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.session_state.sorting_complete = True
                        else:
                            st.warning("The Sorting Oracle requires input for calibration!")
        else: # Sorting is complete
            st.markdown(f"""
            <div class="spell-note animated-text animated-text-delay-1" style="margin-top:25px;">
                <p style="font-size: 1.2rem;" class="futuristic-text">Welcome back, esteemed Chronomancer <strong style="color:{theme['primary']};">{st.session_state.user_name}</strong>! Ready to manipulate more temporal financial data streams? {ROCKET}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="animated-text animated-text-delay-2 futuristic-text" style="margin-top: 20px;">
            <p style="font-size: 1.05rem;">Interface with a universe of enchanted algorithms, quantum predictive models, and data-driven chronomancy. Here, the arcane echoes of the past harmonize with the technological symphonies of a magical future to unlock the temporal keys to prosperity.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_art:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        logo_to_display = theme["house_logo"] if selected_house != "None" else house_themes["None"]["house_logo"]
        caption_text = f"Holographic Projection: {selected_house} House Matrix" if selected_house != "None" else "Central Hogwarts Quantum Core"

        st.markdown(f"""
        <div class="hologram animated-text animated-text-delay-2" style="text-align: center;">
            <img src="{logo_to_display}" width="80%" class="floating-avatar" alt="{caption_text}"/>
        </div>
        <p style="text-align: center; font-style: italic; margin-top: 10px; color:{theme['secondary']}; font-family:'Orbitron', 'MedievalSharp', cursive;" class="futuristic-text">
            {caption_text}
        </p>
        """, unsafe_allow_html=True)


    st.markdown("<hr class='animated-text animated-text-delay-3'>", unsafe_allow_html=True)

    col_features, col_fortune = st.columns(2)

    with col_features:
        with st.expander(f"{ATOM} Access Advanced Holo-Features {ATOM}", expanded=False): # Updated title & icon
            st.markdown(f"""
            <div class="animated-text futuristic-text">
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 12px; font-size: 1.05em;"><span style="font-size: 1.3rem;">{SPELL_BOOK}</span>  <strong>Data Scroll Decryption:</strong> Upload financial matrices via CSV data-scrolls or interface with real-time quantum market signatures using ticker-runes.</li>
                    <li style="margin-bottom: 12px; font-size: 1.05em;"><span style="font-size: 1.3rem;">{GEAR}</span>  <strong>Predictive AI Algorithms:</strong> Deploy potent machine learning enchantments (Linear/Logistic Regression, K-Means Clustering) to project market trajectories.</li>
                    <li style="margin-bottom: 12px; font-size: 1.05em;"><span style="font-size: 1.3rem;">{CRYSTAL_BALL}</span>  <strong>Holographic Visualizations:</strong> Transmute raw data streams into captivating, interactive holo-charts that reveal cryptic financial patterns.</li>
                    <li style="margin-bottom: 12px; font-size: 1.05em;"><span style="font-size: 1.3rem;">{ROCKET}</span>  <strong>Personalized Quantum Field:</strong> Immerse yourself in a dynamic holo-interface attuned to your designated Hogwarts House frequency, augmenting your chronomantic journey.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with st.expander(f"{OWL} Hogwarts Frequencies & Financial Protocols {OWL}", expanded=False): # Updated title
            st.markdown(f"""
            <div class="futuristic-text">
                <h4 style="color: {house_themes['Gryffindor']['primary']}; text-shadow: 0 0 4px {house_themes['Gryffindor']['secondary']};">Gryffindor Protocol 🦁 – The Valiant Vector</h4>
                <p style="font-size: 0.95em;">Bold and decisive, Gryffindor Chronomancers vector resources into high-velocity, high-potential temporal ventures. They champion disruptive innovations and embrace calculated quantum leaps for potentially legendary yields.</p>

                <h4 style="color: {house_themes['Slytherin']['primary']}; text-shadow: 0 0 4px {house_themes['Slytherin']['secondary']};">Slytherin Protocol 🐍 – The Cunning Chronostrategy</h4>
                <p style="font-size: 0.95em;">Ambitious and adaptive, Slytherin strategists excel at navigating multi-dimensional market matrices. They leverage their quantum intuition to identify undervalued chrono-assets and construct financial empires through astute temporal arbitrage.</p>

                <h4 style="color: {house_themes['Hufflepuff']['primary']}; text-shadow: 0 0 4px {house_themes['Hufflepuff']['secondary']};">Hufflepuff Protocol 🦡 – The Steadfast Synchronizer</h4>
                <p style="font-size: 0.95em;">Diligent and harmonic, Hufflepuff investors build quantum wealth through consistent, ethical energy flows. They favor stable, long-term chrono-growth and value systemic integrity and diversified temporal portfolios.</p>

                <h4 style="color: {house_themes['Ravenclaw']['primary']}; text-shadow: 0 0 4px {house_themes['Ravenclaw']['secondary']};">Ravenclaw Protocol 🦅 – The Erudite Entangler</h4>
                <p style="font-size: 0.95em;">Wise and analytical, Ravenclaw financiers delve deep into quantum data streams and ancient chrono-archives. They employ sophisticated entanglement models and intellectual rigor to uncover unique market resonances and innovative temporal investment paradigms.</p>
            </div>
            """, unsafe_allow_html=True)

    with col_fortune:
        st.markdown("""
        <div class="animated-text animated-text-delay-3">
            <h3 style="text-align:center;" class="futuristic-text">Interface with the Oracle Matrix {CRYSTAL_BALL}</h3>
        </div>
        """, unsafe_allow_html=True)

        with st.form("fortune_spell_form_welcome"): # Unique key
            fortune_input = st.text_input("Pose a binary query to the Oracle Matrix regarding your financial timeline:",
                                          placeholder="Will my quantum investments achieve singularity this cycle?",
                                          help="The Oracle offers probabilistic glimpses, not deterministic futures!")
            cast_spell_button = st.form_submit_button(f"{ATOM} Query Oracle Matrix")

        if cast_spell_button:
            if fortune_input:
                st.session_state.spell_cast = True
                magical_loading("Oracle Matrix Processing Query...")
                fortunes = [
                    {"result": "Affirmative. Quantum entanglement favors your vector. High probability of positive outcome.", "color": "#4CAF50"},
                    {"result": "Negative. Temporal interference detected. Current trajectory shows low probability of success.", "color": "#F44336"},
                    {"result": "Indeterminate. The quantum foam is turbulent. Multiple timelines show equal probability. Recalibrate and re-query.", "color": "#FF9800"},
                    {"result": "Probable. Favorable chrono-currents detected. Proceed with optimized parameters.", "color": theme['primary']},
                    {"result": "Uncertain. The timeline is occluded by a probability storm. Further data required.", "color": theme['secondary']},
                    {"result": "Deferred. Temporal flux too high for stable projection. Await next celestial alignment.", "color": "#FF5722"}
                ]
                chosen_fortune = random.choice(fortunes)
                st.markdown(f"""
                <div class="spell-note animated-text futuristic-text" style="border-color: {chosen_fortune['color']}; text-align: center;">
                    <h4 style="color:{chosen_fortune['color']};">Oracle Matrix Response:</h4>
                    <p style="font-size: 1.15rem; color: {theme['text']};">{chosen_fortune['result']}</p>
                    <p style="font-style: italic; font-size: 0.9rem; color: {theme['text']}b3;">*Disclaimer: The Oracle Matrix operates on probabilistic quantum calculations. Future is not immutable.*</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("The Oracle Matrix requires a query to process!")


    st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="animated-text">
        <h2 style="text-align:center;" class="futuristic-text">Your Chronomancer Training Protocol {SPELL_BOOK}</h2>
        <p style="text-align:center; font-size: 1.05em;" class="futuristic-text">Follow this advanced training protocol to master temporal financial dynamics:</p>
    </div>
    """, unsafe_allow_html=True)


    cols_curriculum = st.columns(4)
    curriculum_steps = [
        {"title": "Phase I: Data Influx & Transmutation", "icon": POTION, "desc": "Initiate by interfacing with financial data streams via CSV chrono-scrolls or by divining live market signatures through ticker-runes in the Data Transmutation Chamber."},
        {"title": "Phase II: Holographic Charting & Analysis", "icon": CRYSTAL_BALL, "desc": "Learn to weave complex visual algorithms, transmuting raw numerical inputs into insightful holo-charts and graphs that map the hidden ley lines of quantum finance."},
        {"title": "Phase III: Predictive AI & Enchantment Matrix", "icon": GEAR, "desc": "Interface with the Predictive Enchantment Matrix. Deploy potent Machine Learning algorithms to forecast market trajectories, cluster quantum opportunities, and classify financial phenomena across timelines."},
        {"title": "Phase IV: Quantum Market Observatory & Chrono-Projection", "icon": ATOM, "desc": "Ascend to the Quantum Market Observatory. Apply your skills to interpret real-time data influx, projecting probable futures and making enlightened temporal financial decisions."}
    ]

    for i, step in enumerate(curriculum_steps):
        with cols_curriculum[i]:
            st.markdown(f"""
            <div class="spell-note animated-text animated-text-delay-{i+1} futuristic-text" style="height: 320px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h4 style="text-align:center; color:{theme['primary']};"><span style="font-size:1.5em;">{step['icon']}</span> {step['title']}</h4>
                    <p style="font-size:0.9em;">{step['desc']}</p>
                </div>
                <p style="text-align:center; font-style:italic; color:{theme['secondary']}; font-size:0.8em;">Integrate this protocol...</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="animated-text">
        <h2 style="text-align:center;" class="futuristic-text">Arsenal of Chronomantic Artifacts {LIGHTNING}</h2>
    </div>
    """, unsafe_allow_html=True)

    cols_gallery = st.columns(3)
    artifacts = [
        {"name": "The Temporal Ledger Engine", "img": "https://i.imgur.com/uQKGWJZ.gif", "desc": "A self-calibrating quantum ledger, revealing historical financial waves and projecting cyclical market resonances across timelines."},
        {"name": "The Quantum Entanglement Abacus", "img": "https://i.imgur.com/xAQbA1N.gif", "desc": "A futuristic computational device that transmutes chaotic data streams into coherent financial insights using arcane quantum algorithms."},
        {"name": "The Oracle Holo-Projector", "img": "https://i.imgur.com/N7PmfTI.gif", "desc": "A chrono-projection unit that materializes holographic simulations of potential market futures based on current quantum states and historical data matrices."}
    ]

    for i, artifact in enumerate(artifacts):
        with cols_gallery[i]:
            st.markdown(f"""
            <div class="holodeck-container animated-text animated-text-delay-{i+1} futuristic-text" style="text-align: center; height: 380px; display: flex; flex-direction: column; justify-content: space-between;"> {/* Increased height */}
                <div>
                    <img src="{artifact['img']}" width="90%" style="border-radius: 10px; margin-bottom: 10px; border: 1px solid {theme['secondary']}; box-shadow: 0 0 8px {theme['secondary']};" alt="{artifact['name']}" />
                    <h4 style="color:{theme['primary']};">{artifact['name']}</h4>
                    <p style="font-size:0.9em;">{artifact['desc']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)


# Function for data exploration page with enhanced magical elements
def data_exploration():
    st.markdown(f"""
    <div class="animated-text">
        <h1 style="text-align:center;" class="futuristic-text">The Data Transmutation Chamber {POTION}</h1>
        <p style="text-align:center; font-size: 1.1rem;" class="futuristic-text">Welcome, Chronomancer. In this chamber, raw financial data streams are transmuted into discernible quantum insights. Choose your method of data influx.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)

    tab_scrolls, tab_prophecies = st.tabs([f"{SPELL_BOOK} Ancient Data Scrolls (CSV)", f"{ATOM} Quantum Market Signatures (Stocks)"])

    with tab_scrolls:
        load_data_from_upload()
    with tab_prophecies:
        load_data_from_stock()

    st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)
    reset_data()

    df = st.session_state.get('df')
    ticker_data = st.session_state.get('ticker_data')

    if df is not None:
        st.markdown("""
        <div class="animated-text" id="data-section">
            <h2 style="text-align:center;" class="futuristic-text">Decrypted Glyphs from Ancient Data Scrolls</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<p class='animated-text futuristic-text' style='text-align:center; font-style:italic;'>Initial data sequence from the decrypted scrolls...</p>", unsafe_allow_html=True)
        st.dataframe(df.head(10))

        st.markdown("""
        <div class="animated-text">
            <h3 style="text-align:center;" class="futuristic-text">Quantum Insights Extracted</h3>
        </div>
        """, unsafe_allow_html=True)

        col_stats, col_struct = st.columns(2)
        with col_stats:
            st.markdown("""
            <div class="spell-note animated-text futuristic-text">
                <h4>Statistical Resonance (Summary)</h4>
                <p>Core quantum properties of your numerical data streams:</p>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.describe())
        with col_struct:
            st.markdown("""
            <div class="spell-note animated-text futuristic-text">
                <h4>Quantum Data Weave (Structure)</h4>
                <p>Types and integrity of the entangled data strands:</p>
            </div>
            """, unsafe_allow_html=True)
            column_info = pd.DataFrame({
                'Scroll Field': df.columns,
                'Quantum Type': df.dtypes.astype(str),
                'Non-Null Datapoints': df.count().values,
                'Data Void %': (df.isnull().mean() * 100).round(2).astype(str) + '%'
            })
            st.dataframe(column_info)

        st.markdown("""
        <div class="animated-text" id="chart-section">
            <h3 style="text-align:center;" class="futuristic-text">Holographic Visualizations & Charting Algorithms</h3>
        </div>
        """, unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        if numeric_cols:
            viz_type = st.selectbox(
                "Select Holo-Visualization Algorithm:",
                ["Distribution Density Mapping (Histograms)", "Correlation Nebula Imaging (Heatmap)", "Categorical Harmonics (Bar Charts)"],
                index=None, placeholder="Select a charting algorithm...",
                help="Select an algorithm to visualize different quantum aspects of your data."
            )

            if viz_type == "Distribution Density Mapping (Histograms)":
                selected_num_cols = st.multiselect("Select numerical data streams for density mapping:", numeric_cols, default=numeric_cols[:min(2, len(numeric_cols))],
                                                   help="Choose one or more numerical fields to visualize their distribution density.")
                if selected_num_cols:
                    for col_name in selected_num_cols:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(df[col_name].dropna(), kde=True, color=theme["primary"], ax=ax, bins=30,
                                     line_kws={'linewidth': 2.5, 'color': theme["secondary"]})
                        ax.set_title(f"Density Map of {col_name}", color=theme["text"], fontsize=16, fontfamily='Orbitron, Cinzel')
                        ax.set_xlabel(col_name, color=theme["text"], fontfamily='Orbitron, Cinzel')
                        ax.set_ylabel("Datapoint Frequency", color=theme["text"], fontfamily='Orbitron, Cinzel')
                        ax.set_facecolor('rgba(5,2,15,0.85)')
                        fig.patch.set_facecolor('rgba(5,2,15,0.85)')
                        ax.tick_params(axis='x', colors=theme["text"])
                        ax.tick_params(axis='y', colors=theme["text"])
                        ax.spines['bottom'].set_color(theme["primary"])
                        ax.spines['left'].set_color(theme["primary"])
                        ax.spines['top'].set_color('rgba(5,2,15,0.85)')
                        ax.spines['right'].set_color('rgba(5,2,15,0.85)')
                        plt.grid(axis='y', linestyle=':', alpha=0.4, color=theme["secondary"]) # Changed grid
                        st.pyplot(fig)
                        st.markdown(f"""
                        <div class="spell-note animated-text futuristic-text">
                            <p><strong>Quantum Measures for {col_name}:</strong></p>
                            <ul>
                                <li>Mean Quantum Value: {df[col_name].mean():.2f}</li>
                                <li>Median Entanglement Point: {df[col_name].median():.2f}</li>
                                <li>Standard Deviation Flux: {df[col_name].std():.2f}</li>
                                <li>Value Range Spectrum: {df[col_name].min():.2f} to {df[col_name].max():.2f}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

            elif viz_type == "Correlation Nebula Imaging (Heatmap)":
                if len(numeric_cols) > 1:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    corr_matrix_df = df[numeric_cols].corr()
                    mask = np.triu(np.ones_like(corr_matrix_df, dtype=bool))
                    cmap = sns.diverging_palette(260, 20, s=80, l=45, n=10, center="dark", as_cmap=True) # Adjusted cmap
                    sns.heatmap(corr_matrix_df, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                                annot=True, fmt=".2f", square=True, linewidths=.7, cbar_kws={"shrink": .85}, ax=ax,
                                annot_kws={"color": theme['text'], "fontfamily": "Orbitron, Cinzel"})
                    ax.set_title("Correlation Nebula Between Numerical Data Streams", color=theme["text"], fontsize=18, fontfamily='Orbitron, Cinzel')
                    ax.set_facecolor('rgba(5,2,15,0.85)')
                    fig.patch.set_facecolor('rgba(5,2,15,0.85)')
                    ax.tick_params(axis='x', colors=theme["text"], rotation=45)
                    ax.tick_params(axis='y', colors=theme["text"])
                    plt.xticks(fontfamily='Orbitron, Cinzel')
                    plt.yticks(fontfamily='Orbitron, Cinzel')
                    st.pyplot(fig)
                    upper = corr_matrix_df.where(np.triu(np.ones(corr_matrix_df.shape), k=1).astype(bool))
                    strong_corrs = upper.abs().unstack().sort_values(ascending=False).dropna()
                    if not strong_corrs.empty:
                        st.markdown("""
                        <div class="spell-note animated-text futuristic-text">
                            <h4>Strongest Quantum Entanglements Detected:</h4>
                            <p>The data streams show significant entanglement between these fields:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        for (idx1, idx2), val in strong_corrs.head(5).items():
                            st.markdown(f"""
                            <div class="animated-text futuristic-text" style="padding: 8px; background-color: rgba({primary_rgb_css}, 0.1);
                                        border-left: 3px solid {theme['primary']}; border-radius: 5px; margin-bottom: 5px;">
                                <p style="margin:0;"><strong>{idx1}</strong> & <strong>{idx2}</strong>: <span style="color: {theme['secondary']}; font-weight:bold;">{val:.2f}</span> entanglement strength</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("The Correlation Nebula algorithm requires at least two numerical data streams to map their entanglements.")

            elif viz_type == "Categorical Harmonics (Bar Charts)":
                if categorical_cols and numeric_cols:
                    cat_col_select = st.selectbox("Select a categorical data stream for harmonic analysis:", categorical_cols,
                                                  help="Choose a field with categories (text strings).")
                    val_col_select = st.selectbox("Select a numerical data stream for value analysis:", numeric_cols,
                                                  help="Choose a field with numerical values to compare across categories.")
                    if cat_col_select and val_col_select:
                        fig, ax = plt.subplots(figsize=(12, 7))
                        sns.barplot(x=cat_col_select, y=val_col_select, data=df, ax=ax,
                                    palette=[theme["primary"], theme["secondary"], '#A076F9', '#FDA7DF', '#76D7C4'], # Expanded palette
                                    estimator=np.mean, errorbar=None)
                        ax.set_title(f"Mean '{val_col_select}' by '{cat_col_select}' Harmonics", color=theme["text"], fontsize=16, fontfamily='Orbitron, Cinzel')
                        ax.set_xlabel(cat_col_select, color=theme["text"], fontfamily='Orbitron, Cinzel')
                        ax.set_ylabel(f"Mean {val_col_select}", color=theme["text"], fontfamily='Orbitron, Cinzel')
                        ax.set_facecolor('rgba(5,2,15,0.85)')
                        fig.patch.set_facecolor('rgba(5,2,15,0.85)')
                        ax.tick_params(axis='x', colors=theme["text"], rotation=45, ha="right")
                        ax.tick_params(axis='y', colors=theme["text"])
                        plt.grid(axis='y', linestyle=':', alpha=0.4, color=theme["secondary"])
                        st.pyplot(fig)
                else:
                    st.info("This harmonic analysis requires at least one categorical and one numerical data stream.")
        else:
            st.info("No numerical data streams found to deploy visualization algorithms.")

    elif ticker_data is not None:
        st.markdown("""
        <div class="animated-text" id="data-section">
            <h2 style="text-align:center;" class="futuristic-text">Quantum Market Signature Readings</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<p class='animated-text futuristic-text' style='text-align:center; font-style:italic;'>Initial telemetry from the Quantum Oracle...</p>", unsafe_allow_html=True)
        st.dataframe(ticker_data.head(10))

        st.markdown("""
        <div class="animated-text" id="chart-section">
            <h3 style="text-align:center;" class="futuristic-text">Holo-Visions from the Quantum Market Matrix {ATOM}</h3>
        </div>
        """, unsafe_allow_html=True)

        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📈 Price Vector Glyphs", "📊 Transaction Volume Runes", "🕯️ Candlestick Chronomancy"])

        ticker_data['MA20'] = ticker_data['Close'].rolling(window=20).mean()
        ticker_data['MA50'] = ticker_data['Close'].rolling(window=50).mean()

        with chart_tab1:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['Close'], mode='lines', name='Closing Price Vector',
                                     line=dict(color=theme["primary"], width=2.5, shape='spline'),
                                     hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'))
            if len(ticker_data) >= 20:
                fig_price.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['MA20'], mode='lines', name='20-Cycle Quantum Trend',
                                         line=dict(color=theme["secondary"], width=1.5, dash='dash'),
                                         hovertemplate='<b>MA20</b>: $%{y:.2f}<extra></extra>'))
            if len(ticker_data) >= 50:
                fig_price.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['MA50'], mode='lines', name='50-Cycle Arcane Resonance',
                                         line=dict(color='rgba(150, 107, 45, 0.7)', width=1.5, dash='dot'),
                                         hovertemplate='<b>MA50</b>: $%{y:.2f}<extra></extra>'))
            fig_price.update_layout(
                title=dict(text=f"{ticker_data.columns[1] if len(ticker_data.columns) > 1 else 'Stock'} Price Vector Projection", x=0.5, font=dict(family="Orbitron, Cinzel", size=20, color=theme['primary'])),
                xaxis_title_text="Chrono-Marker", yaxis_title_text="Value (Credits)",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color=theme["text"], font_family="Orbitron, Cinzel",
                hovermode="x unified",
                legend=dict(bgcolor='rgba(5, 2, 15, 0.75)', bordercolor=theme["primary"], font=dict(family="Orbitron, Cinzel")),
                xaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.25)', showgrid=True, type='date',
                           rangeselector=dict(
                               buttons=list([
                                   dict(count=1, label="1M", step="month", stepmode="backward"),
                                   dict(count=6, label="6M", step="month", stepmode="backward"),
                                   dict(count=1, label="YTD", step="year", stepmode="todate"),
                                   dict(count=1, label="1Y", step="year", stepmode="backward"),
                                   dict(step="all", label="ALL")
                               ]), font=dict(family="Orbitron, Cinzel")
                           ), rangeslider=dict(visible=True, bgcolor=f'rgba({primary_rgb_css},0.15)'),
                ),
                yaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.25)', showgrid=True)
            )
            st.plotly_chart(fig_price, use_container_width=True)

        with chart_tab2:
            fig_volume = go.Figure()
            colors_vol = [theme["primary"] if ticker_data['Close'][i] >= ticker_data['Open'][i] else '#E74C3C' for i in range(len(ticker_data))] # Brighter red
            fig_volume.add_trace(go.Bar(x=ticker_data['Date'], y=ticker_data['Volume'], name='Transaction Volume',
                                  marker_color=colors_vol, opacity=0.75,
                                  hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Volume</b>: %{y:,}<extra></extra>'))
            fig_volume.update_layout(
                title=dict(text="Transaction Volume Runes - Market Energy Flux", x=0.5, font=dict(family="Orbitron, Cinzel", size=20, color=theme['primary'])),
                xaxis_title_text="Chrono-Marker", yaxis_title_text="Volume Units",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color=theme["text"], font_family="Orbitron, Cinzel",
                hovermode="x unified",
                legend=dict(bgcolor='rgba(5, 2, 15, 0.75)', bordercolor=theme["primary"], font=dict(family="Orbitron, Cinzel")),
                xaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.25)', type='date'),
                yaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.25)'),
                bargap=0.25
            )
            st.plotly_chart(fig_volume, use_container_width=True)

        with chart_tab3:
            fig_candle = go.Figure(data=[go.Candlestick(x=ticker_data['Date'],
                open=ticker_data['Open'], high=ticker_data['High'],
                low=ticker_data['Low'], close=ticker_data['Close'],
                increasing_line_color= theme['primary'], decreasing_line_color= '#E74C3C',
                name="Price Vectors",
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>Open: $%{open:.2f}<br>High: $%{high:.2f}<br>Low: $%{low:.2f}<br>Close: $%{close:.2f}<extra></extra>')])
            if len(ticker_data) >= 20:
                fig_candle.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['MA20'], mode='lines', name='20-Cycle Quantum Trend',
                                         line=dict(color=theme["secondary"], width=1.5, dash='dash'), opacity=0.75))
            if len(ticker_data) >= 50:
                fig_candle.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['MA50'], mode='lines', name='50-Cycle Arcane Resonance',
                                         line=dict(color='rgba(150, 107, 45, 0.75)', width=1.5, dash='dot'), opacity=0.75))
            fig_candle.update_layout(
                title=dict(text="Candlestick Chronomancy - Daily Market Signatures", x=0.5, font=dict(family="Orbitron, Cinzel", size=20, color=theme['primary'])),
                xaxis_title_text="Chrono-Marker", yaxis_title_text="Value (Credits)",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color=theme["text"], font_family="Orbitron, Cinzel",
                hovermode="x unified",
                legend=dict(bgcolor='rgba(5, 2, 15, 0.75)', bordercolor=theme["primary"], font=dict(family="Orbitron, Cinzel")),
                xaxis_rangeslider_visible=False,
                xaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.25)', type='date'),
                yaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.25)')
            )
            st.plotly_chart(fig_candle, use_container_width=True)

    elif not df and not ticker_data: # If no data is loaded at all
        st.markdown(f"""
        <div class="spell-note animated-text futuristic-text" style="text-align:center;">
            <p style="font-size: 1.2em;">{SPELL_BOOK} The Transmutation Chamber awaits your command, Chronomancer! {ATOM}</p>
            <p>Please initiate data influx using ancient CSV scrolls or by divining quantum market signatures above to begin your temporal analysis.</p>
            <img src="https://i.imgur.com/gbsU7V1.gif" alt="Magical Book" style="width:150px; margin-top:15px; border-radius:10px; opacity:0.7;">
        </div>
        """, unsafe_allow_html=True)


# Placeholder for Machine Learning Spells page
def machine_learning_spells():
    st.markdown(f"""
    <div class="animated-text">
        <h1 style="text-align:center;" class="futuristic-text">The Predictive Enchantment Matrix {GEAR}</h1>
        <p style="text-align:center; font-size: 1.1rem;" class="futuristic-text">Interface with potent AI algorithms and arcane predictive enchantments!</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="spell-note animated-text futuristic-text" style="text-align:center;">
        <h3 style="color:{theme['primary']};">Matrix Under Quantum Calibration {BROOM}</h3>
        <p>Our Magi-Tech engineers are diligently calibrating these advanced predictive algorithms (Temporal Linear Projection, Logistic Quantum States, K-Means Singularity Clustering).</p>
        <p>Soon, you'll deploy enchantments to project market trajectories, classify chronomantic entities, and uncover hidden quantum clusters in your financial data streams.</p>
        <img src="https://i.imgur.com/9jY2L4K.gif" alt="Magic construction" style="width: 200px; margin-top: 20px; border-radius:10px; opacity:0.8;">
        <p style="font-style:italic; margin-top:15px;">Return when the quantum flux has stabilized!</p>
    </div>
    """, unsafe_allow_html=True)

# Placeholder for Market Divination Observatory page
def market_divination_observatory():
    st.markdown(f"""
    <div class="animated-text">
        <h1 style="text-align:center;" class="futuristic-text">The Quantum Market Observatory {OWL}</h1>
        <p style="text-align:center; font-size: 1.1rem;" class="futuristic-text">Peer through the Chrono-Telescope and observe the celestial dance of quantum market forces!</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="spell-note animated-text futuristic-text" style="text-align:center;">
        <h3 style="color:{theme['primary']};">Chrono-Telescopes Undergoing Attunement {LIGHTNING}</h3>
        <p>Our chief Astro-Quantomancer is currently aligning the crystal lenses for optimal market scrying across multiple timelines.</p>
        <p>This observatory will soon offer live data influx, advanced holo-charting systems, and perhaps even a glimpse into the immediate quantum future of your chosen financial entities.</p>
        <img src="https://i.imgur.com/lqzJCQW.gif" alt="Observatory" style="width: 250px; margin-top: 20px; border-radius:10px; opacity:0.8;">
        <p style="font-style:italic; margin-top:15px;">Return when the cosmic resonances are optimal for chrono-projection!</p>
    </div>
    """, unsafe_allow_html=True)


# Page routing
if page == "Hogwarts Holo-Welcome":
    welcome_page()
elif page == "Data Transmutation Chamber":
    data_exploration()
elif page == "Predictive Enchantment Matrix":
    machine_learning_spells()
elif page == "Quantum Market Observatory":
    market_divination_observatory()

# Footer with a magical touch
st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)
st.markdown(f"""
<p style="text-align:center; font-family: 'Orbitron', 'MedievalSharp', cursive; color:{theme['secondary']}; font-size:0.9em;" class="futuristic-text">
    System Engineered with Arcane Code & Pythonic Incantations by a Humble Chronomancer <span class="wand-icon">{ATOM}</span> <br>
    May your investments achieve quantum entanglement with prosperity and your returns accelerate beyond light speed!
</p>
""", unsafe_allow_html=True)
