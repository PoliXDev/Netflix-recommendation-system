Netflix recommendation system

General Summary:

This is a Netflix recommendation system developed in Python as part of a study project at the Conquerblocks Academy, leveraging natural language processing and machine learning techniques.

Main Components:

NetflixRecommender (Recommendation Engine)
Loads data from a CSV file (netflixData.csv).

-Uses TF-IDF and cosine similarity to generate recommendations.
-Processes features such as Production Country, Director, and Cast.
-Generates recommendations based on similar titles.

Graphical User Interface (GUI):

-Implemented with PyQt5.
-features two main screens:
-IntroScreen: Introductory screen with project information.
-MainScreen: Main screen with search functionality.
-Visual Style (NetflixStyle)
-Mimics Netflix's aesthetics.
-Utilizes Netflix's characteristic colors (red, black, gray).
-Defines styles for buttons, search fields, and lists.

Workflow:

-The user launches the application.
-The introduction screen is displayed.
-Clicking "Start Program":
The main screen appears.
The user can search for titles.
The system displays up to 10 similar recommendations.

Technical Features

-Uses sklearn for text processing and similarity calculations.
-Robust error handling.
-Responsive and user-friendly interface.
-Modular, object-oriented design.

Technologies Used

-Python
-PyQt5 for the graphical interface.
-pandas for data handling.
-scikit-learn for the recommendation engine.
-Custom Netflix-inspired style system.
