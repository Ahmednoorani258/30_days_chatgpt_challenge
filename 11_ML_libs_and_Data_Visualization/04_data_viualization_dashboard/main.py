import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# load data
df = pd.read_csv(
    "11_ML_libs_and_Data_Visualization/04_data_viualization_dashboard/Titanic-Dataset.csv"
)

st.sidebar.title("Filters")
pclass = st.sidebar.multiselect("Passenger Class", df["Pclass"].unique())
gender = st.sidebar.multiselect("Gender", df["Sex"].unique())
vis_options = st.sidebar.multiselect(
    "Select Visualization",
    ["Survived vs Age", "Age Distribution", "Fare Distribution", "Survival Rate"],
    default="Age Distribution",
)

# filter data)

filtered_df = df[df["Pclass"].isin(pclass)] if pclass else df
filtered_df = filtered_df[filtered_df["Sex"].isin(gender)] if gender else filtered_df

# Dashboard

st.title("Titanic Data Dashboard")
st.markdown(
    "Explore the Titanic dataset by filtering by passenger class and gender, and visualizing different aspects of the data."
)


# statistics
st.header(f"Statistics for Passenger Class: {pclass} \n Gender: {gender} ")

num_passengers = len(filtered_df)
avg_age = filtered_df["Age"].mean()
avg_fare = filtered_df["Fare"].mean()
survival_rate = filtered_df["Survived"].mean() * 100
st.write(f"Number of passengers: {num_passengers}")
st.write(f"Average age: {avg_age:.2f} years")
st.write(f"Average fare: £{avg_fare:.2f}")
st.write(f"Survival rate: {survival_rate:.2f}%")


# visualization

for vis in vis_options:

    # Check if the current visualization option is "Survived vs Age"
    if vis == "Survived vs Age":
        # Display a subheader in the Streamlit app with the text "Survived vs Age"
        st.subheader("Survived vs Age")
        # Create a new Matplotlib figure and axis object for the plot
        fig, ax = plt.subplots()
        # Create a box plot showing Age distribution by Survived status
        sns.boxplot(x="Survived", y="Age", data=filtered_df, ax=ax)
        # Set the title, including the selected Pclass and optionally Gender
        ax.set_title(
            f"Survived vs Age for Class {pclass}"
            + (f" and Gender {gender}" if gender != "All" else "")
        )
        # Label the x-axis as "Survived" with custom labels "No" and "Yes"
        ax.set_xlabel("Survived")
        ax.set_xticklabels(["No", "Yes"])
        # Label the y-axis as "Age"
        ax.set_ylabel("Age")
        # Render the plot in the Streamlit app
        st.pyplot(fig)
    elif vis == "Age Distribution":
        st.subheader("Age Distribution")

        # Create a new Matplotlib figure and axis object to hold the plot
        fig, ax = plt.subplots()
        # Plot a histogram of the 'Age' column from filtered_df using Seaborn, with a KDE (smooth curve) overlay
        sns.histplot(filtered_df["Age"], kde=True, ax=ax)
        # Set the title of the plot, dynamically including the selected Pclass and optionally Gender
        ax.set_title(
            f"Age Distribution for Class {pclass}"
            + (f" and Gender {gender}" if gender != "All" else "")
        )
        # Label the x-axis as "Age" for clarity
        ax.set_xlabel("Age")
        # Label the y-axis as "Count" to indicate the frequency of ages
        ax.set_ylabel("Count")
        # Render the Matplotlib figure in the Streamlit app
        st.pyplot(fig)
    elif vis == "Fare Distribution":
        st.subheader("Fare Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df["Fare"], kde=True, ax=ax)
        ax.set_title(
            f"Fare Distribution for Class {pclass}"
            + (f" and Gender {gender}" if gender != "All" else "")
        )
        ax.set_xlabel("Fare")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    elif vis == "Survival Rate":
        st.subheader("Survival Rate")
        fig, ax = plt.subplots()
        survival_counts = filtered_df["Survived"].value_counts(normalize=True) * 100
        sns.barplot(x=survival_counts.index, y=survival_counts.values, ax=ax)
        ax.set_title(
            f"Survival Rate for Class {pclass}"
            + (f" and Gender {gender}" if gender != "All" else "")
        )
        ax.set_xlabel("Survived")
        ax.set_ylabel("Percentage (%)")
        ax.set_xticklabels(["No", "Yes"])
        st.pyplot(fig)
