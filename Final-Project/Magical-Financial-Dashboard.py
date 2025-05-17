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
        "background": "https://i.imgur.com/Dh5nWF4.gif",  # Magical Hogwarts with stars animation
        "primary": "#fedd00", # Gold
        "secondary": "#662d91", # Purple
        "text": "#eee7db", # Parchment
        "button_bg": "linear-gradient(45deg, #662d91, #fedd00)",
        "button_hover_bg": "linear-gradient(45deg, #fedd00, #662d91)",
        "house_logo": "https://i.imgur.com/Bt5Uyvw.png",  # Hogwarts crest
        "background_video": "https://i.imgur.com/xnTzwxu.mp4"  # Ambient magical background video
    },
    "Gryffindor": {
        "background": "https://i.imgur.com/v3vgOmS.gif",  # Gryffindor common room animated
        "primary": "#AE0001",  # Scarlet Red (official is #740001, this is brighter)
        "secondary": "#EEBA30",  # Gold
        "text": "#fff2cc", # Light Gold
        "button_bg": "linear-gradient(45deg, #AE0001, #EEBA30)",
        "button_hover_bg": "linear-gradient(45deg, #EEBA30, #AE0001)",
        "house_logo": "https://i.imgur.com/nCU2QKo.png",  # Gryffindor crest
        "background_video": "https://i.imgur.com/nXEZb5S.mp4"  # Gryffindor themed magic video
    },
    "Slytherin": {
        "background": "https://i.imgur.com/1zHdKtU.gif",  # Slytherin dungeon animated
        "primary": "#1A472A",  # Dark Green (official is #1A472A)
        "secondary": "#AAAAAA",  # Silver (official is #555555, this is brighter)
        "text": "#d0f0c0", # Pale Green
        "button_bg": "linear-gradient(45deg, #1A472A, #AAAAAA)",
        "button_hover_bg": "linear-gradient(45deg, #AAAAAA, #1A472A)",
        "house_logo": "https://i.imgur.com/DZ9tEb2.png",  # Slytherin crest
        "background_video": "https://i.imgur.com/tWEp9VH.mp4"  # Slytherin themed magic video
    },
    "Hufflepuff": {
        "background": "https://i.imgur.com/OqBzgel.gif",  # Cozy Hufflepuff common room
        "primary": "#FFDB00",  # Canary Yellow (official is #FFD800)
        "secondary": "#372E29",  # Black (using a dark brown for better aesthetics with yellow)
        "text": "#fff8e1", # Cream
        "button_bg": "linear-gradient(45deg, #372E29, #FFDB00)",
        "button_hover_bg": "linear-gradient(45deg, #FFDB00, #372E29)",
        "house_logo": "https://i.imgur.com/vQT68mS.png",  # Hufflepuff crest
        "background_video": "https://i.imgur.com/Xqzc2OQ.mp4"  # Hufflepuff themed magic video
    },
    "Ravenclaw": {
        "background": "https://i.imgur.com/60r4UcN.gif",  # Ravenclaw tower with stars
        "primary": "#0E1A40",  # Midnight Blue (official is #0E1A40)
        "secondary": "#946B2D",  # Bronze
        "text": "#E2F1FF", # Sky Blue
        "button_bg": "linear-gradient(45deg, #0E1A40, #946B2D)",
        "button_hover_bg": "linear-gradient(45deg, #946B2D, #0E1A40)",
        "house_logo": "https://i.imgur.com/RoTJCGM.png",  # Ravenclaw crest
        "background_video": "https://i.imgur.com/C5UmTTK.mp4"  # Ravenclaw themed magic video
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
    box-shadow: 0 0 6px {theme["primary"]}, inset 0 0 6px {theme["primary"]}, 0 0 12px rgba({primary_rgb_css}, 0.4);
  }}
  50% {{
    border-image-source: linear-gradient(45deg, {theme["secondary"]}, {theme["primary"]}, {theme["secondary"]});
    box-shadow: 0 0 10px {theme["secondary"]}, inset 0 0 10px {theme["secondary"]}, 0 0 18px rgba({secondary_rgb_css}, 0.4);
  }}
  100% {{
    border-image-source: linear-gradient(45deg, {theme["primary"]}, {theme["secondary"]}, {theme["primary"]});
    box-shadow: 0 0 6px {theme["primary"]}, inset 0 0 6px {theme["primary"]}, 0 0 12px rgba({primary_rgb_css}, 0.4);
  }}
}}

@keyframes magicSparkle {{
  0% {{ background-position: 0% 0%; opacity: 0.2; }}
  50% {{ background-position: 100% 100%; opacity: 0.4; }}
  100% {{ background-position: 0% 0%; opacity: 0.2; }}
}}

/* Main app styling */
.stApp {{
    background-image: url('{theme["background"]}');
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
    opacity: 0.4; /* Adjust opacity as needed */
}}

/* Floating magical containers for main content blocks */
/* This targets Streamlit's main content block */
.main .block-container {{
  position: relative;
  background-color: rgba(10, 5, 20, 0.92); /* Darker, more magical base */
  padding: 25px !important;
  border-radius: 25px;
  border: 3px solid transparent;
  animation: neonBorderMove 5s ease-in-out infinite, floatingElement 8s ease-in-out infinite alternate;
  background-image:
    linear-gradient(rgba(10, 5, 20, 0.92), rgba(10, 5, 20, 0.92)),
    linear-gradient(45deg, {theme["primary"]}, {theme["secondary"]}, {theme["primary"]});
  background-origin: border-box;
  background-clip: padding-box, border-box;
  backdrop-filter: blur(3px);
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
  margin-top: 2rem; /* Add some space from the top */
  margin-bottom: 2rem;
}}

.main .block-container:before {{ /* Sparkle overlay for main content */
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('https://i.imgur.com/WJIc0JL.gif'); /* Subtle sparkle gif */
  background-size: 300% 300%; /* Make sparkles finer */
  opacity: 0.08; /* Very subtle */
  border-radius: 22px; /* Inside the border */
  z-index: -1;
  animation: magicSparkle 15s infinite linear;
  pointer-events: none;
}}


/* Futuristic titles with magical glow */
h1, h2 {{
    font-family: 'Cinzel', serif;
    color: {theme["primary"]};
    text-shadow:
      0 0 8px {theme["primary"]},
      0 0 15px {theme["primary"]},
      0 0 20px rgba(255,255,255,0.3);
    letter-spacing: 1.5px;
    position: relative;
}}

h1 {{ font-size: 2.5rem; }}
h2 {{ font-size: 2rem; }}

/* Magical animations on hover for titles */
h1:hover, h2:hover {{
    animation: titleGlow 2s infinite;
}}

@keyframes titleGlow {{
  0% {{ text-shadow: 0 0 8px {theme["primary"]}, 0 0 15px {theme["primary"]}, 0 0 20px rgba(255,255,255,0.3); }}
  50% {{ text-shadow: 0 0 12px {theme["primary"]}, 0 0 25px {theme["primary"]}, 0 0 35px {theme["secondary"]}, 0 0 40px rgba(255,255,255,0.5); }}
  100% {{ text-shadow: 0 0 8px {theme["primary"]}, 0 0 15px {theme["primary"]}, 0 0 20px rgba(255,255,255,0.3); }}
}}

h3 {{
    font-family: 'MedievalSharp', cursive;
    color: {theme["secondary"]};
    text-shadow: 0 0 6px {theme["secondary"]}, 0 0 10px rgba({secondary_rgb_css}, 0.7);
    font-size: 1.6rem;
}}

/* Enchanted buttons with magical hover effects */
.stButton>button {{
  background: {theme["button_bg"]};
  border: 2px solid transparent; /* Initial transparent border */
  border-radius: 12px;
  color: white;
  font-family: 'Cinzel', serif;
  font-weight: bold;
  padding: 12px 28px;
  font-size: 17px;
  cursor: pointer;
  transition: all 0.4s ease;
  box-shadow:
    0 0 8px rgba({primary_rgb_css}, 0.7),
    inset 0 0 8px rgba({primary_rgb_css}, 0.5);
  animation: neonBorderMove 4s ease-in-out infinite alternate; /* Use alternate for smoother transition */
  background-origin: border-box;
  background-clip: padding-box, border-box;
  position: relative;
  z-index: 1;
  text-shadow: 0 0 4px black;
  letter-spacing: 0.5px;
}}

.stButton>button:hover {{
  background: {theme["button_hover_bg"]};
  box-shadow: 0 0 25px 8px {theme["primary"]}, 0 0 15px {theme["secondary"]};
  transform: scale(1.1) rotate(-1deg);
  color: white;
}}

/* Add magical runes/symbols around the button on hover */
.stButton>button:hover:before {{
  content: "✦"; /* Sparkle */
  position: absolute;
  top: -18px;
  left: -18px;
  font-size: 20px;
  color: {theme["secondary"]};
  animation: floatingSymbol 2.5s infinite ease-in-out;
  text-shadow: 0 0 5px {theme["secondary"]};
}}

.stButton>button:hover:after {{
  content: "✧"; /* Another sparkle */
  position: absolute;
  bottom: -18px;
  right: -18px;
  font-size: 20px;
  color: {theme["primary"]};
  animation: floatingSymbol 2.5s infinite ease-in-out reverse;
  text-shadow: 0 0 5px {theme["primary"]};
}}

@keyframes floatingSymbol {{
  0% {{ transform: translate(0, 0) rotate(0deg); opacity: 0.7; }}
  50% {{ transform: translate(4px, -4px) rotate(180deg); opacity: 1;}}
  100% {{ transform: translate(0, 0) rotate(360deg); opacity: 0.7; }}
}}

/* Magic glowing inputs */
.stTextInput>div>input, .stSelectbox>div>div, .stMultiSelect>div>div, .stDateInput>div>div>input {{
  background-color: rgba(15, 8, 30, 0.75); /* Darker input */
  color: {theme["text"]}; /* Input text color */
  border-radius: 10px;
  border: 2px solid {theme["primary"]};
  padding: 10px;
  font-size: 15px;
  font-family: 'Cinzel', serif;
  transition: all 0.3s ease;
  box-shadow: 0 0 5px rgba({primary_rgb_css}, 0.5);
}}
.stTextInput>div>input::placeholder {{
  color: rgba({primary_rgb_css}, 0.6);
}}
.stTextInput>div>input:focus, .stSelectbox>div>div:focus-within, .stMultiSelect>div>div:focus-within, .stDateInput>div>div>input:focus {{
  border: 2px solid {theme["secondary"]};
  box-shadow: 0 0 12px {theme["primary"]}, 0 0 8px {theme["secondary"]};
  background-color: rgba(20, 12, 40, 0.85);
}}
/* Style selectbox dropdown options */
.st-emotion-cache-10oheav {{ /* This class might change with Streamlit versions, inspect if needed */
    background-color: rgba(10, 5, 20, 0.95) !important;
    border: 1px solid {theme["primary"]} !important;
    color: {theme["text"]} !important;
}}
.st-emotion-cache-10oheav:hover {{
    background-color: rgba({primary_rgb_css}, 0.3) !important;
}}


/* Magical sidebar with floating particles */
.css-1d391kg .sidebar .sidebar-content {{ /* Target sidebar */
  background-image: url('{theme["background"]}'); /* Match main background */
  background-size: cover;
  background-repeat: no-repeat;
  position: relative;
  overflow: hidden; /* Important for pseudo-elements */
  border-right: 2px solid {theme["primary"]};
  box-shadow: 5px 0 15px rgba({primary_rgb_css}, 0.3);
}}

.css-1d391kg .sidebar .sidebar-content:before {{
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('https://i.imgur.com/aE3BnKy.gif'); /* Subtle moving particles */
  opacity: 0.1;
  z-index: 0; /* Behind content */
  pointer-events: none;
}}
.css-1d391kg .sidebar .sidebar-content > div {{ /* Ensure sidebar content is above pseudo-element */
    position: relative;
    z-index: 1;
}}

/* Sidebar title color */
.css-1d391kg .sidebar .sidebar-content .css-1aumxhk {{ /* Specific selector for sidebar title */
  color: {theme["primary"]} !important;
  text-shadow: 0 0 5px {theme["primary"]};
}}
.css-1d391kg .sidebar .sidebar-content .stRadio label span{{ /* Radio button labels in sidebar */
    color: {theme["text"]} !important;
    font-size: 1.05em;
}}
.css-1d391kg .sidebar .sidebar-content .stRadio label span:hover{{
    color: {theme["primary"]} !important;
}}


/* Animated magic wand icon with glowing particles */
@keyframes wand-wiggle {{
  0% {{ transform: rotate(0deg); }}
  25% {{ transform: rotate(15deg); }}
  50% {{ transform: rotate(-15deg); }}
  75% {{ transform: rotate(15deg); }}
  100% {{ transform: rotate(0deg); }}
}}

.wand-icon {{
  display: inline-block;
  animation: wand-wiggle 2.5s infinite ease-in-out;
  font-size: 1.6rem;
  margin-left: 8px;
  position: relative;
}}

.wand-icon:after {{
  content: '✦';
  position: absolute;
  color: {theme["primary"]};
  text-shadow: 0 0 8px {theme["primary"]};
  border-radius: 50%;
  top: -5px;
  right: -15px;
  animation: glow 1.5s infinite alternate;
  font-size: 0.8em;
}}
@keyframes glow {{
    0% {{ opacity: 0.5; transform: scale(0.8); }}
    100% {{ opacity: 1; transform: scale(1.2); }}
}}

/* Magical scrollbar styles */
::-webkit-scrollbar {{
  width: 12px;
}}
::-webkit-scrollbar-track {{
  background: rgba(10, 5, 20, 0.7);
  border-radius: 10px;
  box-shadow: inset 0 0 5px rgba(0,0,0,0.3);
}}
::-webkit-scrollbar-thumb {{
  background: linear-gradient(60deg, {theme["primary"]}, {theme["secondary"]});
  border-radius: 10px;
  border: 2px solid rgba(10, 5, 20, 0.7);
}}
::-webkit-scrollbar-thumb:hover {{
  background: linear-gradient(60deg, {theme["secondary"]}, {theme["primary"]});
  box-shadow: 0 0 10px {theme["primary"]};
}}

/* Magical tooltip style */
.tooltip {{
  position: relative;
  display: inline-block;
  cursor: help; /* Changed cursor */
}}

.tooltip .tooltiptext {{
  visibility: hidden;
  width: 220px;
  background-color: rgba(10, 5, 20, 0.95);
  color: {theme["text"]};
  text-align: center;
  border-radius: 8px;
  padding: 12px;
  position: absolute;
  z-index: 100; /* Ensure tooltip is on top */
  bottom: 130%; /* Position above the element */
  left: 50%;
  margin-left: -110px; /* Center the tooltip */
  opacity: 0;
  transition: opacity 0.4s, transform 0.4s;
  transform: translateY(10px);
  border: 1px solid {theme["primary"]};
  box-shadow: 0 0 12px rgba({primary_rgb_css}, 0.7);
  font-size: 0.9em;
}}

.tooltip:hover .tooltiptext {{
  visibility: visible;
  opacity: 1;
  transform: translateY(0);
}}

/* Dynamic floating graphs and charts */
.js-plotly-plot {{ /* Target Plotly chart container */
  animation: floatingElement 12s infinite ease-in-out alternate;
  border-radius: 15px; /* Rounded corners for charts */
  overflow: hidden; /* Clip any overflowing elements from pulse */
}}

/* Magical table styles */
.dataframe {{
  font-family: 'Cinzel', serif;
  border-collapse: separate;
  border-spacing: 0;
  border-radius: 12px;
  overflow: hidden;
  border: 2px solid {theme["primary"]};
  background-color: rgba(15, 8, 30, 0.8);
  box-shadow: 0 0 10px rgba({primary_rgb_css}, 0.3);
  margin: 1em 0; /* Add some margin */
}}

.dataframe th {{
  background: linear-gradient(45deg, {theme["primary"]}, {theme["secondary"]});
  color: white;
  padding: 14px;
  text-shadow: 0 0 5px black;
  font-size: 1.1em;
  text-align: left;
}}

.dataframe td {{
  padding: 12px;
  border-bottom: 1px solid rgba({secondary_rgb_css}, 0.2);
  color: {theme["text"]};
  font-size: 0.95em;
}}

.dataframe tbody tr:hover {{
  background-color: rgba({primary_rgb_css}, 0.15);
}}
.dataframe tbody tr:nth-child(even) {{
    background-color: rgba({primary_rgb_css}, 0.05);
}}
.dataframe tbody tr:nth-child(even):hover {{
    background-color: rgba({primary_rgb_css}, 0.2);
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
    padding: 20px;
}}

.loading-magic {{
  display: inline-block;
  width: 70px;
  height: 70px;
  border: 5px solid rgba({primary_rgb_css}, 0.2); /* Softer base */
  border-radius: 50%;
  border-top-color: {theme["primary"]};
  animation: magicLoading 1.2s infinite linear;
  position: relative;
  margin-bottom: 15px; /* Space for text */
}}

.loading-magic:before {{ /* Inner spinning element */
  content: '';
  position: absolute;
  top: 5px; left: 5px; right: 5px; bottom: 5px; /* Smaller */
  border-radius: 50%;
  border: 4px solid transparent;
  border-top-color: {theme["secondary"]};
  animation: magicLoading 1.8s infinite linear reverse; /* Different speed/direction */
}}
.loading-magic-text {{
    color: {theme["text"]};
    font-family: 'MedievalSharp', cursive;
    font-size: 1.2em;
    text-shadow: 0 0 5px {theme["primary"]};
}}


/* Pulsing effect for plotly graphs */
@keyframes graphPulse {{
  0% {{ box-shadow: 0 0 4px rgba({primary_rgb_css},0.4), 0 0 8px rgba({secondary_rgb_css},0.2); }}
  50% {{ box-shadow: 0 0 12px rgba({primary_rgb_css},0.7), 0 0 18px rgba({secondary_rgb_css},0.5); }}
  100% {{ box-shadow: 0 0 4px rgba({primary_rgb_css},0.4), 0 0 8px rgba({secondary_rgb_css},0.2); }}
}}

.js-plotly-plot .plotly {{ /* Target the inner plotly div */
  border-radius: 15px; /* Ensure rounded corners for the plot itself */
  animation: graphPulse 3.5s infinite ease-in-out;
}}

/* Floating avatar for the welcome page */
.floating-avatar {{
  animation: floatingElement 7s infinite ease-in-out;
  border-radius: 50%; /* If image is square, makes it circle */
  box-shadow: 0 0 15px {theme["primary"]}, 0 0 25px rgba({primary_rgb_css}, 0.5);
  padding: 5px; /* Optional: if you want a border-like effect from the shadow */
  background-color: rgba({primary_rgb_css}, 0.1); /* Slight background if image has transparency */
}}

/* Sparkling text effect */
@keyframes sparkle {{
  0% {{ background-position: 200% center; }} /* Start from off-screen */
  100% {{ background-position: -200% center; }} /* Move across */
}}

.sparkling-text {{
  background: linear-gradient(90deg, {theme["text"]}, {theme["primary"]}, {theme["secondary"]}, {theme["primary"]}, {theme["text"]});
  background-size: 300% auto; /* Wider gradient for smoother animation */
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: sparkle 5s linear infinite;
  font-weight: bold; /* Make it stand out */
}}

/* Animated wizard hat icon */
@keyframes wizardHat {{
  0% {{ transform: translateY(0) rotate(0deg); }}
  25% {{ transform: translateY(-8px) rotate(-4deg); }}
  50% {{ transform: translateY(0) rotate(0deg); }}
  75% {{ transform: translateY(-8px) rotate(4deg); }}
  100% {{ transform: translateY(0) rotate(0deg); }}
}}

.wizard-hat {{
  display: inline-block;
  font-size: 1.8rem; /* Adjusted size */
  animation: wizardHat 3.5s infinite ease-in-out;
  margin-right: 5px; /* Spacing */
}}

/* Magical dividers */
hr {{
  border: 0;
  height: 3px; /* Thicker */
  background-image: linear-gradient(to right, rgba(0,0,0,0), {theme["primary"]}, {theme["secondary"]}, {theme["primary"]}, rgba(0,0,0,0));
  margin: 2em 0; /* More spacing */
  position: relative;
  box-shadow: 0 0 5px {theme["primary"]};
}}

hr:before {{
  content: '✨'; /* Star sparkle */
  position: absolute;
  left: 50%;
  top: 50%; /* Vertically center */
  transform: translate(-50%, -50%);
  background: rgba(10, 5, 20, 0.9); /* Match container background */
  padding: 0 15px;
  color: {theme["secondary"]};
  font-size: 1.5em;
  text-shadow: 0 0 5px {theme["secondary"]};
}}

/* Magical progress bars */
.stProgress > div > div > div > div {{
  background: linear-gradient(90deg, {theme["primary"]}, {theme["secondary"]}); /* Gradient for progress */
  border-radius: 8px; /* Match outer radius */
}}

.stProgress > div > div > div {{
  background-color: rgba({secondary_rgb_css}, 0.2); /* Softer track */
  border-radius: 8px;
}}

/* Chart hover tooltip effect for Plotly */
.plotly-graph-div .hovertext {{ /* Plotly's default hover label class */
  background-color: rgba(10, 5, 20, 0.95) !important;
  border: 1px solid {theme["primary"]} !important;
  border-radius: 8px !important;
  box-shadow: 0 0 10px rgba({primary_rgb_css}, 0.7) !important;
  color: {theme["text"]} !important; /* Ensure text is readable */
  font-family: 'Cinzel', serif !important;
}}
/* Style hover label text */
.plotly-graph-div .hovertext .nums, .plotly-graph-div .hovertext .name {{
    color: {theme["text"]} !important;
}}


/* Animated text effect */
@keyframes fadeInUp {{
  from {{
    opacity: 0;
    transform: translate3d(0, 25px, 0);
  }}
  to {{
    opacity: 1;
    transform: translate3d(0, 0, 0);
  }}
}}

.animated-text {{
  opacity: 0; /* Start hidden */
  animation: fadeInUp 0.8s forwards ease-out;
}}
.animated-text-delay-1 {{ animation-delay: 0.2s; }}
.animated-text-delay-2 {{ animation-delay: 0.4s; }}
.animated-text-delay-3 {{ animation-delay: 0.6s; }}


/* Modal styles (Placeholder, Streamlit doesn't have native modals) */
.modal-content {{
  background-color: rgba(10, 5, 20, 0.97);
  border: 2px solid {theme["primary"]};
  border-radius: 15px;
  box-shadow: 0 0 25px rgba({primary_rgb_css}, 0.6);
  padding: 25px;
  position: relative; /* For absolute positioning of close button */
}}

.close-button {{
  position: absolute;
  top: 15px;
  right: 15px;
  cursor: pointer;
  font-size: 28px;
  color: {theme["primary"]};
  transition: color 0.3s, transform 0.3s;
}}
.close-button:hover {{
    color: {theme["secondary"]};
    transform: rotate(90deg);
}}

/* Glitching effect for futuristic holodecks */
@keyframes glitch {{
  0% {{ clip-path: inset(30% 0 71% 0); transform: translateX(-2px); }}
  10% {{ clip-path: inset(10% 0 31% 0); transform: translateX(1px); }}
  20% {{ clip-path: inset(82% 0 1% 0); transform: translateX(2px); }}
  30% {{ clip-path: inset(23% 0 41% 0); transform: translateX(-1px); }}
  40% {{ clip-path: inset(13% 0 1% 0); transform: translateX(1px); }}
  50% {{ clip-path: inset(55% 0 28% 0); transform: translateX(-2px); }}
  60% {{ clip-path: inset(5% 0 57% 0); transform: translateX(2px); }}
  70% {{ clip-path: inset(64% 0 7% 0); transform: translateX(-1px); }}
  80% {{ clip-path: inset(38% 0 23% 0); transform: translateX(1px); }}
  90% {{ clip-path: inset(28% 0 43% 0); transform: translateX(-2px); }}
  100% {{ clip-path: inset(50% 0 51% 0); transform: translateX(0); }}
}}

.holodeck-container {{
  position: relative;
  overflow: hidden;
  border-radius: 15px;
  padding: 20px;
  background-color: rgba(10, 5, 20, 0.7);
  border: 1px solid rgba({primary_rgb_css}, 0.3);
  box-shadow: 0 0 10px rgba({primary_rgb_css}, 0.2);
  margin-bottom: 1.5rem; /* Spacing */
}}

.holodeck-container:after {{
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, rgba({primary_rgb_css},0.05) 25%, transparent 25%, transparent 50%, rgba({primary_rgb_css},0.05) 50%, rgba({primary_rgb_css},0.05) 75%, transparent 75%, transparent 100%);
  background-size: 8px 8px; /* Scanline size */
  opacity: 0.3;
  animation: glitch 8s infinite linear alternate-reverse;
  pointer-events: none;
  z-index: 1; /* Above content but below text if needed */
}}
.holodeck-container > * {{ /* Ensure content is above glitch effect */
    position: relative;
    z-index: 2;
}}

/* New hologram projection effect */
.hologram {{
  position: relative;
  border-radius: 10px; /* Can be 50% for circular holograms */
  overflow: hidden; /* Important for line effect */
  padding: 10px;
  display: inline-block; /* Or block, depending on usage */
  background: radial-gradient(ellipse at center, rgba({primary_rgb_css},0.15) 0%, rgba({primary_rgb_css},0.05) 70%, transparent 100%);
}}

.hologram:before {{
  content: '';
  position: absolute;
  top: -100%; /* Start lines off-screen */
  left: 0;
  width: 100%;
  height: 300%; /* Make lines long enough to scroll through */
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px, /* Space between lines */
    rgba({primary_rgb_css}, 0.2) 2px, /* Line color and thickness */
    rgba({primary_rgb_css}, 0.2) 3px  /* Line thickness */
  );
  animation: hologramLines 2.5s infinite linear;
  pointer-events: none;
  opacity: 0.7;
}}

@keyframes hologramLines {{
  0% {{ transform: translateY(0%); }}
  100% {{ transform: translateY(33.33%); }} /* Moves one full set of lines (300% height / 3 = 100% visible area) */
}}
.hologram img {{ /* Style image inside hologram */
    opacity: 0.85;
    filter: drop-shadow(0 0 10px {theme["primary"]});
}}


/* Spell notes appearing effect */
@keyframes appearsWithSparkles {{
  0% {{ opacity: 0; filter: blur(4px); transform: translateY(15px) scale(0.95); }}
  70% {{ opacity: 0.7; filter: blur(1px); transform: translateY(0) scale(1.02);}}
  100% {{ opacity: 1; filter: blur(0); transform: translateY(0) scale(1);}}
}}

.spell-note {{
  animation: appearsWithSparkles 0.8s forwards ease-out;
  position: relative;
  padding: 1.2rem; /* More padding */
  border-radius: 12px;
  background-color: rgba(15, 8, 30, 0.85); /* Slightly darker than holodeck */
  border: 1px solid {theme["primary"]};
  margin: 1rem 0;
  box-shadow: 0 0 10px rgba({primary_rgb_css}, 0.3);
}}

.spell-note:before {{
  content: "🔮"; /* Crystal ball icon */
  position: absolute;
  top: -12px; /* Position icon slightly outside */
  left: 15px;
  font-size: 22px; /* Icon size */
  color: {theme["primary"]};
  background-color: rgba(15, 8, 30, 0.95); /* Match background to "sit" on border */
  padding: 0 5px;
  border-radius: 50%;
  text-shadow: 0 0 5px {theme["primary"]};
}}

/* Expander styling */
.stExpander {{
    border: 1px solid {theme["primary"]};
    border-radius: 10px;
    background-color: rgba(15, 8, 30, 0.7);
    margin-bottom: 1rem;
}}
.stExpander header {{
    font-family: 'MedievalSharp', cursive;
    font-size: 1.2em;
    color: {theme["text"]};
}}
.stExpander header:hover {{
    color: {theme["primary"]};
}}
.stExpander svg {{ /* Expander arrow icon */
    fill: {theme["primary"]};
}}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {{
		gap: 24px;
        border-bottom: 2px solid {theme["primary"]};
}}
.stTabs [data-baseweb="tab"] {{
		height: 50px;
        white-space: pre-wrap;
		background-color: transparent;
        font-family: 'MedievalSharp', cursive;
        font-size: 1.1em;
        color: {theme["text"]};
        padding-bottom: 10px; /* Add padding to push text up from border */
}}
.stTabs [data-baseweb="tab"]:hover {{
    background-color: rgba({primary_rgb_css}, 0.1);
    color: {theme["primary"]};
}}
.stTabs [aria-selected="true"] {{
    color: {theme["primary"]};
    font-weight: bold;
    text-shadow: 0 0 5px {theme["primary"]};
    border-bottom: 3px solid {theme["primary"]}; /* Highlight selected tab */
}}

/* Ensure Streamlit header has lower z-index than video background overlay */
.stApp > header {{
    z-index: -3 !important; /* Behind video and main background */
    background-color: transparent !important;
}}

</style>
"""

st.markdown(background_css, unsafe_allow_html=True)

# Add ambient background video based on selected house
if selected_house != "": # Check if a house is selected (or None for default)
    video_html = f'''
    <video autoplay muted loop id="video-background">
      <source src="{theme["background_video"]}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    '''
    st.markdown(video_html, unsafe_allow_html=True)

# Magical wand and wizarding icons
WAND = "🪄"
WIZARD_HAT_ICON = "🧙" # Using single character for better alignment in some cases
CRYSTAL_BALL = "🔮"
OWL = "🦉"
BROOM = "🧹"
POTION = "⚗️"
SPELL_BOOK = "📖"
STARS = "✨"
LIGHTNING = "⚡"

# Sidebar page navigation with enhanced magical UI
st.sidebar.markdown(f"### {SPELL_BOOK} Magical Navigation")
page_options = ["Hogwarts Welcome", "Data Exploration Chamber",
                "Machine Learning Spells", "Market Divination Observatory"]
page = st.sidebar.radio("", page_options,
    help="Navigate through the different magical sections of the application.")


# Initialize session state variables
if "df" not in st.session_state:
    st.session_state.df = None
if "ticker_data" not in st.session_state:
    st.session_state.ticker_data = None
if "spell_cast" not in st.session_state: # For the welcome page fortune spell
    st.session_state.spell_cast = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "sorting_complete" not in st.session_state:
    st.session_state.sorting_complete = False

# Function to display magical loading animation
def magical_loading(message="Casting spell..."):
    loading_html = f"""
    <div class="loading-magic-container">
        <div class="loading-magic"></div>
        <div class="loading-magic-text">{message}</div>
    </div>
    """
    with st.empty(): # Use st.empty to show and then clear the loading animation
        st.markdown(loading_html, unsafe_allow_html=True)
        time.sleep(2.5) # Simulate spell casting time
        st.empty() # Clear the loading animation

# Function to load data from file upload with magical effects
def load_data_from_upload():
    st.markdown(f"### {CRYSTAL_BALL} Upload your ancient financial scrolls")
    uploaded_file = st.file_uploader("Upload your financial dataset (CSV)", type=["csv"],
                                     help="Provide a CSV file containing your financial data, like an ancient scroll filled with numbers.")

    if uploaded_file:
        try:
            magical_loading("Decoding ancient financial scrolls...")
            data = pd.read_csv(uploaded_file)
            st.session_state.df = data
            st.success(f"{STARS} The magical scrolls have been successfully decoded! {STARS}")

            st.markdown("""
            <div class="spell-note animated-text">
                <p>Your financial data has been successfully transformed into magical insights, ready for divination!</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"The magical scroll seems to be corrupted: {e}")
            st.markdown(f"""
            <div class="spell-note animated-text" style="border-color: #ff4444;">
                <p>Alas! Your magical scroll contains enchantments we cannot decipher. Please try a different scroll format (ensure it's a valid CSV).</p>
            </div>
            """, unsafe_allow_html=True)

# Function to load data from stock with magical effects
def load_data_from_stock():
    st.markdown(f"### {CRYSTAL_BALL} Summon Stock Market Prophecies")

    ticker_col, start_col, end_col = st.columns(3)

    with ticker_col:
        ticker = st.text_input("Enter Stock Symbol Rune (e.g. AAPL)", value="GOOGL", key="ticker_input", # Changed key
                              help="The magical rune (stock symbol) of the company you wish to analyze.")

    with start_col:
        start_date = st.date_input("Ancient Date", pd.to_datetime("2023-01-01"), key="start_date_input", # Changed key
                                   help="The starting date for your prophecy.")

    with end_col:
        end_date = st.date_input("Modern Date", pd.to_datetime("today"), key="end_date_input", # Changed key
                                 help="The ending date for your prophecy.")

    if st.button(f"{WAND} Summon Market Data"):
        if not ticker:
            st.warning("Please enter a Stock Symbol Rune to summon data.")
            return
        try:
            magical_loading(f"Divining the future of {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error(f"No prophecies found for the magical rune '{ticker}' in the specified time frame.")
                st.markdown(f"""
                <div class="spell-note animated-text" style="border-color: #ff4444;">
                    <p>The crystal ball shows naught for this symbol. Perhaps try a different magical rune or adjust the prophecy's time period.</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.ticker_data = None
            else:
                data.reset_index(inplace=True)
                st.session_state.ticker_data = data
                st.success(f"{STARS} The market divination for {ticker} has been successfully manifested! {STARS}")

                open_price = data['Open'].iloc[0]
                close_price = data['Close'].iloc[-1]
                high_price = data['High'].max()
                low_price = data['Low'].min()
                percent_change = ((close_price - open_price) / open_price) * 100 if open_price != 0 else 0

                change_color = theme["primary"] if percent_change > 0 else "#F44336" # Use theme primary for positive

                st.markdown(f"""
                <div class="holodeck-container animated-text" style="margin-top: 20px;">
                    <h3 style="text-align: center; margin-bottom: 15px; color: {theme['secondary']};">{ticker} Prophecy Summary {OWL}</h3>
                    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px;">
                        <div style="text-align: center; padding: 10px; background: rgba({primary_rgb_css}, 0.1); border-radius: 8px;">
                            <h4 style="color:{theme['text']}; font-size:0.9em; margin-bottom:5px;">Opening Spell</h4>
                            <p style="font-size: 1.4rem; color:{theme['primary']};">${open_price:.2f}</p>
                        </div>
                        <div style="text-align: center; padding: 10px; background: rgba({primary_rgb_css}, 0.1); border-radius: 8px;">
                            <h4 style="color:{theme['text']}; font-size:0.9em; margin-bottom:5px;">Closing Revelation</h4>
                            <p style="font-size: 1.4rem; color:{theme['primary']};">${close_price:.2f}</p>
                        </div>
                        <div style="text-align: center; padding: 10px; background: rgba({primary_rgb_css}, 0.1); border-radius: 8px;">
                            <h4 style="color:{theme['text']}; font-size:0.9em; margin-bottom:5px;">Peak Enchantment</h4>
                            <p style="font-size: 1.4rem; color:{theme['primary']};">${high_price:.2f}</p>
                        </div>
                         <div style="text-align: center; padding: 10px; background: rgba({primary_rgb_css}, 0.1); border-radius: 8px;">
                            <h4 style="color:{theme['text']}; font-size:0.9em; margin-bottom:5px;">Lowest Ebb</h4>
                            <p style="font-size: 1.4rem; color:{theme['primary']};">${low_price:.2f}</p>
                        </div>
                        <div style="text-align: center; padding: 10px; background: rgba({primary_rgb_css}, 0.1); border-radius: 8px;">
                            <h4 style="color:{theme['text']}; font-size:0.9em; margin-bottom:5px;">Fortune's Shift</h4>
                            <p style="font-size: 1.4rem; color: {change_color}; font-weight: bold;">{percent_change:.2f}%</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"The crystal ball is cloudy: {e}")
            st.markdown(f"""
            <div class="spell-note animated-text" style="border-color: #ff4444;">
                <p>Our divination spell encountered interference. Please verify the magical rune symbol and ensure the Ministry of Magic's internet connection is stable.</p>
            </div>
            """, unsafe_allow_html=True)

# Function to reset data with magical effects
def reset_data():
    if st.button(f"{WAND} Evanesco (Clear All Data)"):
        magical_loading("Vanishing all traces of previous spells...")
        st.session_state.df = None
        st.session_state.ticker_data = None
        st.success("All data has vanished as if by magic!")
        time.sleep(1) # Allow success message to be seen
        st.experimental_rerun()

# Function for the welcome page with interactive magical elements
def welcome_page():
    col_main, col_art = st.columns([2,1])

    with col_main:
        st.markdown(f"""
        <div class="animated-text">
            <h1 style="font-size: 2.8rem; margin-bottom: 0.2rem;">
                <span class="wizard-hat">{WIZARD_HAT_ICON}</span>Welcome to the <br><span class="sparkling-text">Hogwarts Financial Mystics</span><span class="wand-icon">{WAND}</span>
            </h1>
            <p style="font-size: 1.1rem; color: {theme['secondary']}; font-family: 'MedievalSharp', cursive;">Where Ancient Wizardry Meets Futuristic Finance!</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.sorting_complete:
            st.markdown("""
            <div class="spell-note animated-text animated-text-delay-1">
                <p style="font-size: 1.1rem;">Ah, a new apprentice! Before you delve into the arcane arts of financial divination, the Sorting Hat must assess your disposition.</p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.user_name = st.text_input("Pray, tell us your wizarding name:", key="wizard_name_input_welcome",
                                                        help="Your chosen name for this magical journey.")

            if st.session_state.user_name:
                with st.expander(f"🎩 The Sorting Hat Ponders for {st.session_state.user_name}...", expanded=True):
                    st.markdown(f"""
                    <p style="font-style: italic; color: {theme['text']};">"Hmm, a keen mind, {st.session_state.user_name}. Now, to gauge your financial spirit..."</p>
                    """, unsafe_allow_html=True)

                    financial_question = st.selectbox(
                        "When faced with a vault of Gringotts gold, your first instinct is to:",
                        [
                            "Bravely invest in a daring new magical enterprise (Gryffindor Potential)",
                            "Craft a shrewd plan to multiply it through cunning ventures (Slytherin Potential)",
                            "Secure it steadily, ensuring fair growth for all involved (Hufflepuff Potential)",
                            "Research ancient texts for forgotten investment wisdom (Ravenclaw Potential)"
                        ],
                        index=None, # No default selection
                        placeholder="Choose your financial approach...",
                        help="This will help the Sorting Hat guide you."
                    )

                    if st.button("Let the Sorting Hat Decide!"):
                        if financial_question:
                            magical_loading("The Sorting Hat is Deliberating...")
                            if "Gryffindor" in financial_question: suggested_house = "Gryffindor"
                            elif "Slytherin" in financial_question: suggested_house = "Slytherin"
                            elif "Hufflepuff" in financial_question: suggested_house = "Hufflepuff"
                            else: suggested_house = "Ravenclaw"

                            st.balloons()
                            st.success(f"**The Sorting Hat declares: {suggested_house.upper()}!**")
                            st.markdown(f"""
                            <div class="spell-note animated-text">
                                <p>Excellent, {st.session_state.user_name}! Your financial spirit aligns with the qualities of {suggested_house}.
                                You may now select '{suggested_house} House' from the sidebar for a personalized experience, or continue with the grand Hogwarts theme!</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.session_state.sorting_complete = True
                        else:
                            st.warning("The Sorting Hat requires an answer to make its choice!")
        else:
            st.markdown(f"""
            <div class="spell-note animated-text animated-text-delay-1">
                <p style="font-size: 1.2rem;">Welcome back, esteemed financial wizard <strong style="color:{theme['primary']};">{st.session_state.user_name}</strong>! Ready to weave more financial spells? {STARS}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="animated-text animated-text-delay-2" style="margin-top: 20px;">
            <p style="font-size: 1.05rem;">Embark on a journey through enchanted charts, potent predictions, and data-driven divination. Here, the wisdom of ages past converges with the technological marvels of a magical future to unlock the secrets of prosperity.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_art:
        st.markdown("<br><br>", unsafe_allow_html=True) # Spacer
        logo_to_display = theme["house_logo"] if selected_house != "None" else house_themes["None"]["house_logo"]
        caption_text = f"The Noble Crest of {selected_house} House" if selected_house != "None" else "The Grand Hogwarts Crest"

        st.markdown(f"""
        <div class="hologram animated-text animated-text-delay-2" style="text-align: center;">
            <img src="{logo_to_display}" width="80%" class="floating-avatar" alt="{caption_text}"/>
        </div>
        <p style="text-align: center; font-style: italic; margin-top: 10px; color:{theme['secondary']}; font-family:'MedievalSharp', cursive;">
            {caption_text}
        </p>
        """, unsafe_allow_html=True)


    st.markdown("<hr class='animated-text animated-text-delay-3'>", unsafe_allow_html=True)

    col_features, col_fortune = st.columns(2)

    with col_features:
        with st.expander(f"{STARS} Unveil Our Magical Features {STARS}", expanded=False):
            st.markdown(f"""
            <div class="animated-text">
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 12px; font-size: 1.05em;"><span style="font-size: 1.3rem;">{CRYSTAL_BALL}</span>  <strong>Data Divination Scrolls:</strong> Upload financial data from ancient CSV scrolls or summon real-time stock market prophecies using ticker runes.</li>
                    <li style="margin-bottom: 12px; font-size: 1.05em;"><span style="font-size: 1.3rem;">{WIZARD_HAT_ICON}</span>  <strong>Predictive Enchantments:</strong> Cast powerful machine learning spells (Linear/Logistic Regression, K-Means Clustering) to foresee market movements.</li>
                    <li style="margin-bottom: 12px; font-size: 1.05em;"><span style="font-size: 1.3rem;">{POTION}</span>  <strong>Visual Hexes & Charms:</strong> Transfigure raw data into captivating, interactive charts and graphs that reveal hidden financial patterns.</li>
                    <li style="margin-bottom: 12px; font-size: 1.05em;"><span style="font-size: 1.3rem;">{BROOM}</span>  <strong>Personalized House Aether:</strong> Immerse yourself in a dynamic interface themed to your chosen Hogwarts House, enhancing your mystical journey.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with st.expander(f"{OWL} Hogwarts Houses & Their Financial Magic {OWL}", expanded=False):
            st.markdown(f"""
            <div>
                <h4 style="color: {house_themes['Gryffindor']['primary']}; text-shadow: 0 0 4px {house_themes['Gryffindor']['secondary']};">Gryffindor 🦁 – The Courageous Capitalist</h4>
                <p style="font-size: 0.95em;">Bold and chivalrous, Gryffindor financiers make daring investments in high-growth ventures. They champion innovation and aren't afraid of calculated risks for potentially legendary returns.</p>

                <h4 style="color: {house_themes['Slytherin']['primary']}; text-shadow: 0 0 4px {house_themes['Slytherin']['secondary']};">Slytherin 🐍 – The Cunning Cultivator</h4>
                <p style="font-size: 0.95em;">Ambitious and resourceful, Slytherin strategists excel at navigating complex market dynamics. They leverage their shrewdness to identify undervalued assets and build empires through astute financial maneuvers.</p>

                <h4 style="color: {house_themes['Hufflepuff']['primary']}; text-shadow: 0 0 4px {house_themes['Hufflepuff']['secondary']};">Hufflepuff 🦡 – The Steadfast Steward</h4>
                <p style="font-size: 0.95em;">Diligent and patient, Hufflepuff investors build wealth through consistent, ethical practices. They favor stable, long-term growth and value fairness and diversification in their financial endeavors.</p>

                <h4 style="color: {house_themes['Ravenclaw']['primary']}; text-shadow: 0 0 4px {house_themes['Ravenclaw']['secondary']};">Ravenclaw 🦅 – The Erudite Economist</h4>
                <p style="font-size: 0.95em;">Wise and analytical, Ravenclaw financiers delve deep into data and research. They employ sophisticated models and intellectual rigor to uncover unique market insights and innovative investment strategies.</p>
            </div>
            """, unsafe_allow_html=True)

    with col_fortune:
        st.markdown("""
        <div class="animated-text animated-text-delay-3">
            <h3 style="text-align:center;">Try a Simple Divination Spell {CRYSTAL_BALL}</h3>
        </div>
        """, unsafe_allow_html=True)

        with st.form("fortune_spell_form"): # Added key to form
            fortune_input = st.text_input("Ask the Oracle a yes/no question about your financial future:",
                                          placeholder="Will my Gringotts vault overflow this year?",
                                          help="The Oracle offers glimpses, not guarantees!")
            cast_spell_button = st.form_submit_button(f"{WAND} Consult the Oracle")

        if cast_spell_button:
            if fortune_input:
                st.session_state.spell_cast = True # Mark that spell was cast
                magical_loading("The Oracle is peering into the mists of time...")

                fortunes = [
                    {"result": "The runes glow brightly! Signs point to YES, prosperity is on your horizon!", "color": "#4CAF50"},
                    {"result": "A shadow falls upon the crystal ball... Alas, the omens suggest NO for now.", "color": "#F44336"},
                    {"result": "The future is as murky as the Black Lake. The Oracle whispers: MAYBE.", "color": "#FF9800"},
                    {"result": "Favorable winds are blowing! A resounding YES, but exercise wisdom.", "color": theme['primary']},
                    {"result": "The path is unclear, like a Floo journey gone awry. The answer is veiled.", "color": theme['secondary']},
                    {"result": "The stars are not aligned for this query. The Oracle advises patience: NOT YET.", "color": "#FF5722"}
                ]
                chosen_fortune = random.choice(fortunes)

                st.markdown(f"""
                <div class="spell-note animated-text" style="border-color: {chosen_fortune['color']}; text-align: center;">
                    <h4 style="color:{chosen_fortune['color']};">The Oracle Has Spoken!</h4>
                    <p style="font-size: 1.15rem; color: {theme['text']};">{chosen_fortune['result']}</p>
                    <p style="font-style: italic; font-size: 0.9rem; color: {theme['text']}b3;">*Remember, young wizard, the future is ever in motion. This is but one possible thread in the tapestry of time.*</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("You must pose a question to the Oracle!")


    st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="animated-text">
        <h2 style="text-align:center;">Your Financial Wizardry Curriculum {SPELL_BOOK}</h2>
        <p style="text-align:center; font-size: 1.05em;">Follow this enchanted path to master the art of financial divination:</p>
    </div>
    """, unsafe_allow_html=True)

    cols_curriculum = st.columns(4)
    curriculum_steps = [
        {"title": "Module I: Data Transfiguration", "icon": POTION, "desc": "Begin by summoning financial data via ancient CSV scrolls or by divining live market trends through potent ticker runes in the Data Exploration Chamber."},
        {"title": "Module II: Charting Charms", "icon": CRYSTAL_BALL, "desc": "Learn to weave visual spells, transforming raw numbers into insightful charts and graphs that reveal the hidden ley lines of finance."},
        {"title": "Module III: Predictive Potions", "icon": WIZARD_HAT_ICON, "desc": "Brew powerful Machine Learning Spells to forecast market movements, cluster opportunities, and classify financial phenomena."},
        {"title": "Module IV: Oracle's Observatory", "icon": OWL, "desc": "Ascend to the Market Divination Observatory to apply your skills, interpreting real-time data streams and making enlightened financial prophecies."}
    ]

    for i, step in enumerate(curriculum_steps):
        with cols_curriculum[i]:
            st.markdown(f"""
            <div class="spell-note animated-text animated-text-delay-{i+1}" style="height: 280px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h4 style="text-align:center; color:{theme['primary']};"><span style="font-size:1.5em;">{step['icon']}</span> {step['title']}</h4>
                    <p style="font-size:0.9em;">{step['desc']}</p>
                </div>
                <p style="text-align:center; font-style:italic; color:{theme['secondary']}; font-size:0.8em;">Master this art...</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="animated-text">
        <h2 style="text-align:center;">Gallery of Mystical Financial Artifacts {LIGHTNING}</h2>
    </div>
    """, unsafe_allow_html=True)

    cols_gallery = st.columns(3)
    artifacts = [
        {"name": "The Chronos Ledger", "img": "https://i.imgur.com/uQKGWJZ.gif", "desc": "An ancient, self-updating ledger that reveals historical financial trends and whispers of cyclical market patterns."},
        {"name": "The Quantum Abacus", "img": "https://i.imgur.com/xAQbA1N.gif", "desc": "A futuristic calculating device that transmutes complex data streams into golden insights using arcane algorithms."},
        {"name": "The Seer's Simulacrum", "img": "https://i.imgur.com/N7PmfTI.gif", "desc": "An oracle orb that projects holographic simulations of potential market futures based on current enchantments and historical data."}
    ]

    for i, artifact in enumerate(artifacts):
        with cols_gallery[i]:
            st.markdown(f"""
            <div class="holodeck-container animated-text animated-text-delay-{i+1}" style="text-align: center; height: 350px; display: flex; flex-direction: column; justify-content: space-between;">
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
        <h1 style="text-align:center;">The Data Exploration Chamber {POTION}</h1>
        <p style="text-align:center; font-size: 1.1rem;">Welcome, apprentice! In this chamber, raw financial data is transmuted into discernible insights. Choose your method of summoning.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)

    tab_scrolls, tab_prophecies = st.tabs([f"{SPELL_BOOK} Ancient Scrolls (CSV)", f"{CRYSTAL_BALL} Market Divination (Stocks)"])

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
            <h2 style="text-align:center;">Deciphered Glyphs from the Ancient Scrolls</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<p class='animated-text' style='text-align:center; font-style:italic;'>A glimpse into the first few enchantments...</p>", unsafe_allow_html=True)
        st.dataframe(df.head(10))

        st.markdown("""
        <div class="animated-text">
            <h3 style="text-align:center;">Magical Insights Extracted</h3>
        </div>
        """, unsafe_allow_html=True)

        col_stats, col_struct = st.columns(2)
        with col_stats:
            st.markdown("""
            <div class="spell-note animated-text">
                <h4>Statistical Runes (Summary)</h4>
                <p>The core magical properties of your numeric data:</p>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.describe())
        with col_struct:
            st.markdown("""
            <div class="spell-note animated-text">
                <h4>Arcane Data Weave (Structure)</h4>
                <p>The types and integrity of the data strands:</p>
            </div>
            """, unsafe_allow_html=True)
            column_info = pd.DataFrame({
                'Scroll Column': df.columns,
                'Magical Type': df.dtypes.astype(str),
                'Non-Null Glyphs': df.count().values,
                'Missing Glyphs %': (df.isnull().mean() * 100).round(2).astype(str) + '%'
            })
            st.dataframe(column_info)

        st.markdown("""
        <div class="animated-text" id="chart-section">
            <h3 style="text-align:center;">Visual Incantations & Charting Charms</h3>
        </div>
        """, unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        if numeric_cols:
            viz_type = st.selectbox(
                "Choose your visualization spell:",
                ["Distribution Divination (Histograms)", "Correlation Constellations (Heatmap)", "Categorical Comparisons (Bar Charts)"],
                index=None, placeholder="Select a charting charm...",
                help="Select a spell to visualize different aspects of your data."
            )

            if viz_type == "Distribution Divination (Histograms)":
                selected_num_cols = st.multiselect("Select numeric scrolls for distribution divination:", numeric_cols, default=numeric_cols[:min(2, len(numeric_cols))],
                                                   help="Choose one or more numeric columns to see their distribution.")
                if selected_num_cols:
                    for col_name in selected_num_cols:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(df[col_name].dropna(), kde=True, color=theme["primary"], ax=ax, bins=30,
                                     line_kws={'linewidth': 2, 'color': theme["secondary"]})
                        ax.set_title(f"Distribution of {col_name}", color=theme["text"], fontsize=16, fontfamily='Cinzel')
                        ax.set_xlabel(col_name, color=theme["text"], fontfamily='Cinzel')
                        ax.set_ylabel("Frequency of Glyphs", color=theme["text"], fontfamily='Cinzel')
                        ax.set_facecolor('rgba(10,5,20,0.8)') # Dark background
                        fig.patch.set_facecolor('rgba(10,5,20,0.8)')
                        ax.tick_params(axis='x', colors=theme["text"])
                        ax.tick_params(axis='y', colors=theme["text"])
                        ax.spines['bottom'].set_color(theme["primary"])
                        ax.spines['left'].set_color(theme["primary"])
                        ax.spines['top'].set_color('rgba(10,5,20,0.8)')
                        ax.spines['right'].set_color('rgba(10,5,20,0.8)')
                        plt.grid(axis='y', linestyle='--', alpha=0.3, color=theme["secondary"])
                        st.pyplot(fig)
                        st.markdown(f"""
                        <div class="spell-note animated-text">
                            <p><strong>Mystical Measures for {col_name}:</strong></p>
                            <ul>
                                <li>Average Arcane Value (Mean): {df[col_name].mean():.2f}</li>
                                <li>Median Magical Point: {df[col_name].median():.2f}</li>
                                <li>Standard Deviation Sigil: {df[col_name].std():.2f}</li>
                                <li>Range of Enchantment: {df[col_name].min():.2f} to {df[col_name].max():.2f}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

            elif viz_type == "Correlation Constellations (Heatmap)":
                if len(numeric_cols) > 1:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    corr_matrix_df = df[numeric_cols].corr()
                    mask = np.triu(np.ones_like(corr_matrix_df, dtype=bool))
                    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark", as_cmap=True) # Custom cmap
                    sns.heatmap(corr_matrix_df, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                                annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax,
                                annot_kws={"color": theme['text'], "fontfamily": "Cinzel"})
                    ax.set_title("Correlation Constellations Between Numeric Scrolls", color=theme["text"], fontsize=18, fontfamily='Cinzel')
                    ax.set_facecolor('rgba(10,5,20,0.8)')
                    fig.patch.set_facecolor('rgba(10,5,20,0.8)')
                    ax.tick_params(axis='x', colors=theme["text"], rotation=45)
                    ax.tick_params(axis='y', colors=theme["text"])
                    plt.xticks(fontfamily='Cinzel')
                    plt.yticks(fontfamily='Cinzel')
                    st.pyplot(fig)

                    # Highlight strongest correlations
                    upper = corr_matrix_df.where(np.triu(np.ones(corr_matrix_df.shape), k=1).astype(bool))
                    strong_corrs = upper.abs().unstack().sort_values(ascending=False).dropna()
                    if not strong_corrs.empty:
                        st.markdown("""
                        <div class="spell-note animated-text">
                            <h4>Strongest Magical Affinities Revealed:</h4>
                            <p>The scrolls show strong connections between these elements:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        for (idx1, idx2), val in strong_corrs.head(5).items():
                            st.markdown(f"""
                            <div class="animated-text" style="padding: 8px; background-color: rgba({primary_rgb_css}, 0.1);
                                        border-left: 3px solid {theme['primary']}; border-radius: 5px; margin-bottom: 5px;">
                                <p style="margin:0;"><strong>{idx1}</strong> & <strong>{idx2}</strong>: <span style="color: {theme['secondary']}; font-weight:bold;">{val:.2f}</span> correlation strength</p>
                            </div>
                            """, unsafe_allow_html=True)

                else:
                    st.info("The Correlation Constellation spell requires at least two numeric scrolls to map their affinities.")

            elif viz_type == "Categorical Comparisons (Bar Charts)":
                if categorical_cols and numeric_cols:
                    cat_col_select = st.selectbox("Select a categorical scroll for comparison:", categorical_cols,
                                                  help="Choose a column with categories (text).")
                    val_col_select = st.selectbox("Select a numeric scroll for values:", numeric_cols,
                                                  help="Choose a column with numbers to compare across categories.")
                    if cat_col_select and val_col_select:
                        fig, ax = plt.subplots(figsize=(12, 7))
                        sns.barplot(x=cat_col_select, y=val_col_select, data=df, ax=ax,
                                    palette=[theme["primary"], theme["secondary"], '#A076F9', '#FDA7DF'], # Example palette
                                    estimator=np.mean, errorbar=None) # Use mean, remove error bars for clarity
                        ax.set_title(f"Average '{val_col_select}' by '{cat_col_select}' Categories", color=theme["text"], fontsize=16, fontfamily='Cinzel')
                        ax.set_xlabel(cat_col_select, color=theme["text"], fontfamily='Cinzel')
                        ax.set_ylabel(f"Average {val_col_select}", color=theme["text"], fontfamily='Cinzel')
                        ax.set_facecolor('rgba(10,5,20,0.8)')
                        fig.patch.set_facecolor('rgba(10,5,20,0.8)')
                        ax.tick_params(axis='x', colors=theme["text"], rotation=45, ha="right")
                        ax.tick_params(axis='y', colors=theme["text"])
                        plt.grid(axis='y', linestyle='--', alpha=0.3, color=theme["secondary"])
                        st.pyplot(fig)
                else:
                    st.info("This charm requires at least one categorical and one numeric scroll.")
        else:
            st.info("No numeric scrolls found to cast visualization spells.")

    elif ticker_data is not None:
        st.markdown("""
        <div class="animated-text" id="data-section">
            <h2 style="text-align:center;">Market Prophecy Readings</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<p class='animated-text' style='text-align:center; font-style:italic;'>The initial visions from the Oracle...</p>", unsafe_allow_html=True)
        st.dataframe(ticker_data.head(10))

        st.markdown("""
        <div class="animated-text" id="chart-section">
            <h3 style="text-align:center;">Visions from the Market Crystal Ball {CRYSTAL_BALL}</h3>
        </div>
        """, unsafe_allow_html=True)

        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📈 Price Journey Glyphs", "📊 Trading Volume Runes", "🕯️ Candlestick Scrying"])

        # Calculate moving averages once
        ticker_data['MA20'] = ticker_data['Close'].rolling(window=20).mean()
        ticker_data['MA50'] = ticker_data['Close'].rolling(window=50).mean()

        with chart_tab1: # Price Journey
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['Close'], mode='lines', name='Closing Price',
                                     line=dict(color=theme["primary"], width=2.5, shape='spline'),
                                     hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'))
            if len(ticker_data) >= 20:
                fig_price.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['MA20'], mode='lines', name='20-Day Prophetic Trend',
                                         line=dict(color=theme["secondary"], width=1.5, dash='dash'),
                                         hovertemplate='<b>MA20</b>: $%{y:.2f}<extra></extra>'))
            if len(ticker_data) >= 50:
                fig_price.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['MA50'], mode='lines', name='50-Day Arcane Cycle',
                                         line=dict(color='rgba(150, 107, 45, 0.7)', width=1.5, dash='dot'), # Bronze-like for Ravenclaw secondary
                                         hovertemplate='<b>MA50</b>: $%{y:.2f}<extra></extra>'))

            fig_price.update_layout(
                title=dict(text=f"{ticker_data['Close'].name if 'Close' in ticker_data else 'Stock'} Price Journey Prophecy", x=0.5, font=dict(family="Cinzel", size=20, color=theme['primary'])),
                xaxis_title_text="Date Rune", yaxis_title_text="Price (in Galleons)",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color=theme["text"], font_family="Cinzel",
                hovermode="x unified",
                legend=dict(bgcolor='rgba(10, 5, 20, 0.7)', bordercolor=theme["primary"], font=dict(family="Cinzel")),
                xaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.2)', showgrid=True, type='date',
                           rangeselector=dict(
                               buttons=list([
                                   dict(count=1, label="1m", step="month", stepmode="backward"),
                                   dict(count=6, label="6m", step="month", stepmode="backward"),
                                   dict(count=1, label="YTD", step="year", stepmode="todate"),
                                   dict(count=1, label="1y", step="year", stepmode="backward"),
                                   dict(step="all")
                               ])
                           ), rangeslider=dict(visible=True, bgcolor=f'rgba({primary_rgb_css},0.1)'),
                ),
                yaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.2)', showgrid=True)
            )
            st.plotly_chart(fig_price, use_container_width=True)

        with chart_tab2: # Trading Volume
            fig_volume = go.Figure()
            colors = [theme["primary"] if ticker_data['Close'][i] >= ticker_data['Open'][i] else '#F44336' for i in range(len(ticker_data))]
            fig_volume.add_trace(go.Bar(x=ticker_data['Date'], y=ticker_data['Volume'], name='Trading Volume',
                                  marker_color=colors, opacity=0.7,
                                  hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Volume</b>: %{y:,}<extra></extra>'))
            fig_volume.update_layout(
                title=dict(text="Trading Volume Runes - Market Energy Flow", x=0.5, font=dict(family="Cinzel", size=20, color=theme['primary'])),
                xaxis_title_text="Date Rune", yaxis_title_text="Volume Transacted",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color=theme["text"], font_family="Cinzel",
                hovermode="x unified",
                legend=dict(bgcolor='rgba(10, 5, 20, 0.7)', bordercolor=theme["primary"], font=dict(family="Cinzel")),
                xaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.2)', type='date'),
                yaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.2)'),
                bargap=0.3
            )
            st.plotly_chart(fig_volume, use_container_width=True)

        with chart_tab3: # Candlestick
            fig_candle = go.Figure(data=[go.Candlestick(x=ticker_data['Date'],
                open=ticker_data['Open'], high=ticker_data['High'],
                low=ticker_data['Low'], close=ticker_data['Close'],
                increasing_line_color= theme['primary'], decreasing_line_color= '#F44336', # Red for decreasing
                name="Price Candles",
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>Open: $%{open:.2f}<br>High: $%{high:.2f}<br>Low: $%{low:.2f}<br>Close: $%{close:.2f}<extra></extra>')])

            if len(ticker_data) >= 20:
                fig_candle.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['MA20'], mode='lines', name='20-Day Prophetic Trend',
                                         line=dict(color=theme["secondary"], width=1.5, dash='dash'), opacity=0.7))
            if len(ticker_data) >= 50:
                fig_candle.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['MA50'], mode='lines', name='50-Day Arcane Cycle',
                                         line=dict(color='rgba(150, 107, 45, 0.7)', width=1.5, dash='dot'), opacity=0.7))

            fig_candle.update_layout(
                title=dict(text="Candlestick Scrying - Daily Market Omens", x=0.5, font=dict(family="Cinzel", size=20, color=theme['primary'])),
                xaxis_title_text="Date Rune", yaxis_title_text="Price (in Galleons)",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color=theme["text"], font_family="Cinzel",
                hovermode="x unified",
                legend=dict(bgcolor='rgba(10, 5, 20, 0.7)', bordercolor=theme["primary"], font=dict(family="Cinzel")),
                xaxis_rangeslider_visible=False, # Common to disable for candlestick clarity
                xaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.2)', type='date'),
                yaxis=dict(gridcolor=f'rgba({secondary_rgb_css}, 0.2)')
            )
            st.plotly_chart(fig_candle, use_container_width=True)

    elif not df and not ticker_data:
        st.markdown(f"""
        <div class="spell-note animated-text" style="text-align:center;">
            <p style="font-size: 1.2em;">{SPELL_BOOK} The Chamber awaits your command, young wizard! {CRYSTAL_BALL}</p>
            <p>Please summon data using the ancient scrolls (CSV) or by divining from the stock market prophecies above to begin your exploration.</p>
            <img src="https://i.imgur.com/gbsU7V1.gif" alt="Magical Book" style="width:150px; margin-top:15px; border-radius:10px; opacity:0.7;">
        </div>
        """, unsafe_allow_html=True)


# Placeholder for Machine Learning Spells page
def machine_learning_spells():
    st.markdown(f"""
    <div class="animated-text">
        <h1 style="text-align:center;">The Chamber of Machine Learning Spells {WIZARD_HAT_ICON}</h1>
        <p style="text-align:center; font-size: 1.1rem;">Harness the power of arcane algorithms and potent predictive enchantments!</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="spell-note animated-text" style="text-align:center;">
        <h3 style="color:{theme['primary']};">Under Arcane Construction {BROOM}</h3>
        <p>The house-elves are diligently working to perfect these powerful spells (Linear Regression, Logistic Regression, K-Means Clustering).</p>
        <p>Soon, you'll be able to cast enchantments to predict trends, classify mystical entities, and uncover hidden clusters in your financial data.</p>
        <img src="https://i.imgur.com/9jY2L4K.gif" alt="Magic construction" style="width: 200px; margin-top: 20px; border-radius:10px; opacity:0.8;">
        <p style="font-style:italic; margin-top:15px;">Please check back after the next full moon!</p>
    </div>
    """, unsafe_allow_html=True)

# Placeholder for Market Divination Observatory page
def market_divination_observatory():
    st.markdown(f"""
    <div class="animated-text">
        <h1 style="text-align:center;">The Market Divination Observatory {OWL}</h1>
        <p style="text-align:center; font-size: 1.1rem;">Peer through the great telescope of finance and observe the celestial dance of market forces!</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='animated-text'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="spell-note animated-text" style="text-align:center;">
        <h3 style="color:{theme['primary']};">Telescopes Being Calibrated {LIGHTNING}</h3>
        <p>Our chief astrologer, Professor Trelawney's great-grand-nephew, is currently aligning the lenses for optimal market scrying.</p>
        <p>This observatory will soon offer live data streams, advanced charting tools, and perhaps even a glimpse into the immediate future of your chosen stocks.</p>
        <img src="https://i.imgur.com/lqzJCQW.gif" alt="Observatory" style="width: 250px; margin-top: 20px; border-radius:10px; opacity:0.8;">
        <p style="font-style:italic; margin-top:15px;">Return when the stars are right for prophecy!</p>
    </div>
    """, unsafe_allow_html=True)


# Page routing
if page == "Hogwarts Welcome":
    welcome_page()
elif page == "Data Exploration Chamber":
    data_exploration()
elif page == "Machine Learning Spells":
    machine_learning_spells()
elif page == "Market Divination Observatory":
    market_divination_observatory()

# Footer with a magical touch
st.markdown("---")
st.markdown(f"""
<p style="text-align:center; font-family: 'MedievalSharp', cursive; color:{theme['secondary']}; font-size:0.9em;">
    Crafted with Magic & Python by a Humble Apprentice <span class="wand-icon">✨</span> <br>
    May your Galleons multiply and your investments soar like a Golden Snitch!
</p>
""", unsafe_allow_html=True)
