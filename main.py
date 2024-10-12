import streamlit as st
import scrapetube
import random
import numpy as np
## for getting title
import requests
from bs4 import BeautifulSoup

from pytube import YouTube

import embedding_bucketing.embedding_model_test as em
from config import openai_api_key

import ao_core as ao
from arch_recommender import arch

# Initialize global variables
if "videos_in_list" not in st.session_state:
    st.session_state.videos_in_list = []
if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = []
if "current_binary_input" not in st.session_state:
    st.session_state.current_binary_input = []


display_video = False

if "agent" not in st.session_state:
    print("-------creating agent-------")
    st.session_state.agent = ao.Agent(arch, notes=[])

max_distance = 20 # setting it high for no auto bucketing
amount_of_binary_digits = 10
type_of_distance_calc = "COSINE SIMILARITY"
start_Genre = genres = ["Drama", "Comedy", "Action", "Romance", "Documentary", "Music", "Gaming", "Entertainment", "News", "Thriller", "Horror", "Science Fiction", "Fantasy", "Adventure", "Mystery", "Animation", "Family", "Historical", "Biography", "Superhero"
]
em.config(openai_api_key) # configuring openai client for embedding model
cache_file_name = "genre_embedding_cache.json"
cache, genre_buckets = em.init(cache_file_name, start_Genre)


st.set_page_config(page_title="DemoRS", layout="wide")

# Predefined list of random search terms
random_search_terms = ['funny', 'gaming', 'science', 'technology', 'news', 'random', 'adventure', "programming", "computer science"]


def get_random_youtube_link():
    print("Attempting to get a random video link...")
    
    # Select a random search term
    search_term = random.choice(random_search_terms)
    
    # Get videos from scrapetube
    videos = scrapetube.get_search(query=search_term, limit=10)
    
    # Shuffle and pick a random video
    video_list = list(videos)
    random.shuffle(video_list)
    
    if video_list:
        random_video = random.choice(video_list)
        video_id = random_video.get('videoId')
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    
    return None

def get_title_from_url(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    link = soup.find_all(name="title")[0]
    title = str(link)
    title = title.replace("<title>","")
    title = title.replace("</title>","")
    print("title", title)
    return title

def get_length_from_url(url): # returns if the video is short, medium or long in binary
    yt = YouTube(url)
    try:
        length = yt.length
    except Exception as e:
        print("error in getting length", e)
        length = 0
    length_binary = []
    if length < 5:
        length_binary = [0, 0]
    elif length >= 5 and length < 20:
        length_binary = [0, 1]
    else:
        length_binary = [1, 1]


    return length, length_binary

def get_video_data_from_url(url):
    length, length_binary = get_length_from_url(url)
    title = get_title_from_url(url)
    closest_genre, genre_binary_encoding = embedding_bucketing_response(title, max_distance, genre_buckets, type_of_distance_calc, amount_of_binary_digits)
    genre_binary_encoding = genre_binary_encoding.tolist()
    print("Closest genre to title", title, "is", closest_genre)
    return length, length_binary, closest_genre, genre_binary_encoding

def embedding_bucketing_response(uncategorized_input, max_distance, bucket_list, type_of_distance_calc, amount_of_binary_digits):
    sort_response = em.auto_sort(uncategorized_input, max_distance, bucket_list, type_of_distance_calc, amount_of_binary_digits) 

    closest_distance = sort_response[0]
    closest_bucket   = sort_response[1]  # which bucket the uncategorized_input was placed in
    bucket_id        = sort_response[2]  # the id of the closest_bucket
    bucket_binary    = sort_response[3]  # binary representation of the id for INPUT into api.aolabs.ai

    return closest_bucket, bucket_binary # returning the closest bucket and its binary encoding


def next_video():  # function return closest genre and binary encoding of next video and displays it 
    display_video = False
    length, length_binary, closest_genre, genre_binary_encoding = get_video_data_from_url(st.session_state.videos_in_list[0])
    st.write("Genre:", closest_genre, "Length:", length)
    binary_input_to_agent = genre_binary_encoding+ length_binary
    st.write("binary input:", binary_input_to_agent)
    st.session_state.current_binary_input = binary_input_to_agent # storing the current binary input to reduce redundant calls
    st.session_state.recommendation_result = agent_response(binary_input_to_agent)
    recommended = "undefined"
    if st.session_state.recommendation_result[0] == 0:
        recommended = "Not recommended for you"
    else:
        recommended = "Recommended for you"
    st.write("Recommendation result: ", recommended)

    st.video(st.session_state.videos_in_list[0])
    return closest_genre, genre_binary_encoding

def train_agent(user_response):
    binary_input = st.session_state.current_binary_input
    if user_response == "pleasure":
        Cpos = True 
        Cneg = False
    elif user_response == "pain":
        Cneg = True
        Cpos = False
    st.session_state.agent.next_state(INPUT=binary_input, Cpos=Cpos, Cneg=Cneg, print_result=False)


def agent_response(binary_input): # function to get agent response on next video
    #input = get_agent_input()
    st.session_state.agent.next_state( INPUT=binary_input, print_result=False)
    response = st.session_state.agent.story[st.session_state.agent.state-1, st.session_state.agent.arch.Z__flat]
    return response



# Title of the app
st.title("Recommender")

big_left, big_right = st.columns(2)

with big_left:
    # Input for the number of links
    count = st.text_input("How many links to load", value='0')
    count = int(count) 
    url = st.text_input("Enter a youtube video to test", value=None)
    if url !=None:
        print("Adding url")
        try:
            st.session_state.videos_in_list.insert(0, url)
            next_video()
        except Exception as e:
            st.write("Error url not recognised")
    # Start button logic
    if st.button("Start"):
        if count > 0:
            st.write(f"Loading {count} links...")
            for i in range(count):
                data = get_random_youtube_link()
                while not data:  # Retry until a valid link is retrieved
                    data = get_random_youtube_link()
                if data not in st.session_state.videos_in_list:
                    st.session_state.videos_in_list.append(data)
            st.write(f"Loaded {count} videos.")
            genre, genre_binary_encoding = next_video()




with big_right:
    small_right, small_left = st.columns(2)
    with small_right:
        if st.button("Pleasure"):#
            train_agent(user_response="pleasure")
            if len(st.session_state.videos_in_list) > 0:
                st.session_state.videos_in_list.pop(0)  # Remove the first video from the list
                display_video = True
            else:
                st.write("The list is empty, cannot pop any more items.")

    with small_left:
        if st.button("Pain"):
            train_agent(user_response="pain")
            if len(st.session_state.videos_in_list) > 0:
                st.session_state.videos_in_list.pop(0)  # Remove the first video from the list
                display_video = True
            else:
                st.write("The list is empty, cannot pop any more items.")


    if st.session_state.videos_in_list:
        if display_video == True:
            genre, genre_binary_encoding = next_video()
    else:
        st.write("No more videos in the list.")