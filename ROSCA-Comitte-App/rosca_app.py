import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
import time
import os

# Define the file name for saving and loading ROSCA data
file_name = "rosca_data.csv"

# Function to load ROSCA data from CSV file if it exists
def load_rosca_data():
    if os.path.exists(file_name):
        try:
            df = pd.read_csv(file_name)
            return {
                "committees": {
                    row["Committee"]: {
                        "members": str(row["Members"]).split(',') if pd.notna(row["Members"]) else [],
                        "contribution": float(row["Contribution"]) if pd.notna(row["Contribution"]) else 0.0,
                        "rotation": int(row["Rotation"]) if pd.notna(row["Rotation"]) else 0,
                        "funds": float(row["Funds"]) if pd.notna(row["Funds"]) else 0.0,
                        "history": str(row["History"]).split(',') if pd.notna(row["History"]) else []
                    }
                    for _, row in df.iterrows()
                }
            }
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return {"committees": {}}
    return {"committees": {}}

# Function to save ROSCA data to a CSV file
def save_rosca_data(data):
    try:
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
    except Exception as e:
        st.error(f"Error saving data: {e}")

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

# Function to delete a committee
def delete_committee(committee_name):
    if committee_name in rosca_data["committees"]:
        del rosca_data["committees"][committee_name]
        save_rosca_data(rosca_data)
        st.success(f"Committee '{committee_name}' deleted successfully!")
        st.experimental_rerun()  # Refresh the app to update the committee list

# Function to draw a spinning wheel using Plotly
def draw_wheel(members, rotation=0, highlight_index=None):
    num_members = len(members)
    if num_members == 0:
        return None

    angles = [(i * (360 / num_members)) for i in range(num_members)]
    colors = ['#1f77b4' if i != highlight_index else '#ffeb3b' for i in range(num_members)]  # Blue and yellow

    fig = go.Figure()

    for i, member in enumerate(members):
        fig.add_trace(go.Barpolar(
            r=[1],
            theta=[angles[i]],
            width=[360 / num_members],
            marker_color=colors[i],
            marker_line_color="black",
            marker_line_width=2,
            name=member,
            text=member,  # Display member name on the wheel
            hoverinfo="text"
        ))

    fig.update_layout(
        title="Spinning Wheel 🎰",
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(
                rotation=rotation,  # Rotate the wheel
                direction="clockwise"
            )
        ),
        showlegend=True,
        height=500,
        width=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig

# Function to animate spinning wheel and select winner
def animated_spin_wheel(members):
    if not members:
        st.error("No members in this committee to spin the wheel!")
        return None

    placeholder = st.empty()  # Placeholder for the wheel
    num_spins = random.randint(20, 30)  # More spins for smoother effect
    rotation = 0
    speed = 10  # Initial speed of rotation (degrees per frame)

    for i in range(num_spins):
        rotation = (rotation + speed) % 360  # Continuous rotation
        speed += random.uniform(0.5, 1.5)  # Gradually increase speed
        if i > num_spins * 0.7:  # Slow down towards the end
            speed *= 0.9

        fig = draw_wheel(members, rotation=rotation)
        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.05)  # Smooth animation

    # Final spin to select winner
    winner = random.choice(members)
    highlight_index = members.index(winner)
    for i in range(10):  # Slow final rotation to highlight winner
        rotation = (rotation + speed * 0.5) % 360
        fig = draw_wheel(members, rotation=rotation, highlight_index=highlight_index)
        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.1)

    st.success(f"🎉 Winner: {winner} gets the money first! 🎊")
    return winner

# Load existing ROSCA data (if any) at the start of the app
rosca_data = load_rosca_data()

# Streamlit UI setup
st.title("ROSCA Management App 🎡")

# Sidebar for navigation
menu = ["Create Committee", "Manage Committee", "Visualize Data"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Create Committee":
    st.header("Create a New Committee")
    name = st.text_input("Committee Name")
    members_input = st.text_input("Members (comma-separated, e.g., John,Jane,Bob)")
    members = [m.strip() for m in members_input.split(',')] if members_input else []
    contribution = st.slider("Monthly Contribution", 0, 1000, 100)

    if st.button("Create Committee"):
        if not name:
            st.error("Committee name cannot be empty!")
        elif not members or members == ['']:
            st.error("Please provide at least one member!")
        elif name in rosca_data["committees"]:
            st.error("Committee name already exists! Choose a different name.")
        else:
            create_committee(name, members, contribution)
            st.success(f"Committee '{name}' created successfully!")

elif choice == "Manage Committee":
    st.header("Manage Committee")
    if not rosca_data["committees"]:
        st.warning("No committees available. Create a committee first!")
    else:
        committee_name = st.selectbox("Select Committee", list(rosca_data["committees"].keys()))

        if committee_name:
            committee = rosca_data["committees"][committee_name]
            st.write(f"**Members**: {', '.join(committee['members'])}")
            st.write(f"**Monthly Contribution**: ${committee['contribution']:.2f}")
            st.write(f"**Total Funds**: ${committee['funds']:.2f}")
            st.write(f"**Current Rotation**: {committee['members'][committee['rotation']]}")

            if st.button("Make Contribution"):
                make_contribution(committee_name)
                st.success("Contribution made successfully!")

            # Spinning wheel feature
            st.subheader("Spin the Wheel to Decide Who Gets the Money First")
            if st.button("Spin Now"):
                winner = animated_spin_wheel(committee["members"])
                if winner:
                    committee["rotation"] = committee["members"].index(winner)
                    save_rosca_data(rosca_data)

            # Delete committee option
            st.subheader("Delete Committee")
            if st.button(f"Delete '{committee_name}'"):
                if st.checkbox("Are you sure? This action cannot be undone."):
                    delete_committee(committee_name)

elif choice == "Visualize Data":
    st.header("Visualize Financial Data")
    if not rosca_data["committees"]:
        st.warning("No committees available. Create a committee first!")
    else:
        committee_name = st.selectbox("Select Committee", list(rosca_data["committees"].keys()))

        if committee_name:
            committee = rosca_data["committees"][committee_name]

            # Contribution History Graph
            st.subheader("Contribution History")
            if committee['history']:
                history = [float(h) for h in committee['history']]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(history))), y=history, mode='lines+markers', name='Contributions'))
                fig.update_layout(
                    title="Contribution History",
                    xaxis_title="Contribution Number",
                    yaxis_title="Amount ($)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No contributions made yet.")

            # Rotation Chart
            st.subheader("Rotation Chart")
            rotation_data = {member: 0 for member in committee['members']}
            rotation_data[committee['members'][committee['rotation']]] = committee['funds']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(rotation_data.values()), y=list(rotation_data.keys()), orientation='h'))
            fig.update_layout(
                title="Current Rotation Funds",
                xaxis_title="Funds ($)",
                yaxis_title="Member",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Members and Contributions Table
            st.subheader("Members and Contributions")
            member_data = pd.DataFrame({
                'Member': committee['members'],
                'Contribution': [committee['contribution']] * len(committee['members'])
            })
            st.table(member_data)

# Button to Export Data
if st.button("📂 Export & Download CSV"):
    if not rosca_data["committees"]:
        st.error("No data to export!")
    else:
        committees = rosca_data["committees"]
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
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="rosca_data.csv", mime="text/csv")