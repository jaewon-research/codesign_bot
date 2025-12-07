# Codesign Bot Simulation
This repository is a fork of https://github.com/camel-ai/oasis. In this project we are creating a simulated social network with a small number of users (< 1k) and observing behaviors of simulated agents.

# Contributions
We introduce a new chronological recommendation system parameterized by max connection degree (to control the maximum connection degree of posts shown to agents 1=follower, 2=mutual, etc)

We create a simulated social graph for the agents using the Barabási-Albert model.

# TODO
Create a list of simulated agent profiles that will be used to inform agent behaviors during the simulation


# Simulation Variables
- Duration: Two weeks
- Number of users: 1000
- Bi-directional followers: User1 can follow User2 without User2 following User1
- Max connection degree: (parameterized) [1,2,3]
- refresh_rec_post_count: TBD (The number of posts returned by the social media internal recommendation system per refresh)
- Daily questions (same question per day for all users - see questions.tsv)
- max_rec_post_len: The maximum number of posts per user in the recommendation table (buffer)
- available actions
    -post
    -comment
    -like
    -direct message
    -update profile
        -update interests #hashtags (tbd predefined list or text)
    -follow new people
        -request
    -change feed sort
        -1st degree (close friend, friend, follower)
        -2nd degree only (only direct mutuals)
        -mutual interests
    -add as friend
        -make close friends
        -apply "close friend" to past posts (or only current)
    -view profile
        -status update (only shown on profile page)
    -view friends
        -list of friends with 1 (most recent) status update from each
        -profile information (name, profile picture, interests, etc)
    -daily questions
        - 5 daily questions (can be shared)
        - 7 days worth of question
            -click button to send to friend
    -view notifications
        -comment on your post
        -reply to your comment
        -emoji reactions
    -send question to friends (as notification)
        -notification on friends response


#Experiment
simulate same group of personas
-show them both versions of the platform (in both orders)
-run N times (with different sets of personas)

Control version:
-Two feeds
    -1st degree feed (close friends, friends, followers) - chronological
    -discover feed (everyone else n=infinity twitter style feed)

Non-Control version:
-Three feeds
    -1st degree feed (close friends, friends, followers) - chronological
    -2nd degree feed (mutuals only - reddit sort)
    -2nd degree + 3rd degree (reddit sort) (possibly vary to include 4th + 5th, etc)
-profile feed
    -list of friends + close friends
        -status update

-additional features
    -emoji reactions
    -private comments (only OP can see)
    -change text of daily question "Share with your friends"
    -send prompts directly to other friends (notification)
    -possibly see how other people responded to daily questions
