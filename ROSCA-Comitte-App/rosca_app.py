import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os

# Define the file name for saving and loading ROSCA data
file_name = "rosca_data.csv"

# Function to load ROSCA data from CSV file if it exists
def load_rosca_data():
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        return {
            "committees": {
                row["Committee"]: {
                    "members": row["Members"].split(','),
                    "contribution": row["Contribution"],
                    "rotation": row["Rotation"],
                    "funds": row["Funds"],
                    "history": row["History"].split(',') if pd.notna(row["History"]) else []
                }
                for _, row in df.iterrows()
            }
        }
    return {"committees": {}}

# Function to save ROSCA data to a CSV file
def save_rosca_data(data):
    committees = data["committees"]
    df = pd.DataFrame([
        {
            "Committee": name,
            "Members": ",".join(committee["members"]),
            "Contribution": committee["contribution"],
            "Rotation": committee["rotation"],
            "Funds": committee["funds"],
            "History": ",".join(map(str, committee["history"]))
        }
        for name, committee in committees.items()
    ])
    df.to_csv(file_name, index=False)

# Load existing ROSCA data (if any) at the start of the app
rosca_data = load_rosca_data()

# Streamlit UI setup
st.title("ROSCA Management App")

# Sidebar for navigation
menu = ["Create Committee", "Manage Committee", "Visualize Data"]
choice = st.sidebar.selectbox("Menu", menu)

# Function to create a new committee
def create_committee(name, members, contribution):
    rosca_data["committees"][name] = {
        "members": members,
        "contribution": contribution,
        "rotation": 0,
        "funds": 0,
        "history": []
    }
    save_rosca_data(rosca_data)

# Function to make a contribution
def make_contribution(committee_name):
    if committee_name in rosca_data["committees"]:
        committee = rosca_data["committees"][committee_name]
        committee["funds"] += committee["contribution"]
        committee["history"].append(committee["contribution"])
        committee["rotation"] = (committee["rotation"] + 1) % len(committee["members"])
        save_rosca_data(rosca_data)

# Function to draw spinning wheel
def draw_wheel(members):
    fig, ax = plt.subplots()
    num_members = len(members)
    wedges, texts = ax.pie([1] * num_members, labels=members, startangle=90, wedgeprops={"edgecolor": "black"})
    plt.title("Spinning Wheel")
    return fig

# Function to simulate spinning wheel and pick a random member
def spin_wheel(members):
    st.write("Spinning the wheel...")
    time.sleep(2)  # Simulate spinning time
    winner = random.choice(members)
    st.success(f"Winner: {winner} gets the money first!")
    return winner

if choice == "Create Committee":
    st.header("Create a New Committee")
    name = st.text_input("Committee Name")
    members_input = st.text_input("Members (comma-separated)")
    members = [m.strip() for m in members_input.split(',')] if members_input else []
    contribution = st.slider("Monthly Contribution", 0, 1000, 100)

    if st.button("Create Committee"):
        if name and members:
            create_committee(name, members, contribution)
            st.success(f"Committee '{name}' created successfully!")
        else:
            st.error("Please provide a committee name and at least one member.")

elif choice == "Manage Committee":
    st.header("Manage Committee")
    committee_name = st.selectbox("Select Committee", list(rosca_data["committees"].keys()))

    if committee_name:
        committee = rosca_data["committees"][committee_name]
        st.write(f"Members: {', '.join(committee['members'])}")
        st.write(f"Monthly Contribution: {committee['contribution']}")
        st.write(f"Total Funds: {committee['funds']}")
        st.write(f"Current Rotation: {committee['members'][committee['rotation']]}")

        if st.button("Make Contribution"):
            make_contribution(committee_name)
            st.success("Contribution made successfully!")

        # Spinning wheel feature
        st.subheader("Spin the Wheel")
        fig = draw_wheel(committee["members"])
        st.pyplot(fig)

        if st.button("Spin Now"):
            winner = spin_wheel(committee["members"])
            committee["rotation"] = committee["members"].index(winner)
            save_rosca_data(rosca_data)
            st.success(f"{winner} will get the money first!")

elif choice == "Visualize Data":
    st.header("Visualize Financial Data")
    committee_name = st.selectbox("Select Committee", list(rosca_data["committees"].keys()))

    if committee_name:
        committee = rosca_data["committees"][committee_name]

        st.subheader("Contribution History")
        fig, ax = plt.subplots()
        ax.plot(committee['history'], marker='o')
        ax.set_title("Contribution History")
        ax.set_xlabel("Contribution Number")
        ax.set_ylabel("Amount")
        st.pyplot(fig)

        st.subheader("Rotation Chart")
        rotation_data = {member: 0 for member in committee['members']}
        rotation_data[committee['members'][committee['rotation']]] = committee['funds']
        st.bar_chart(rotation_data)

        st.subheader("Members and Contributions")
        member_data = pd.DataFrame({
            'Member': committee['members'],
            'Contribution': [committee['contribution']] * len(committee['members'])
        })
        st.table(member_data)
