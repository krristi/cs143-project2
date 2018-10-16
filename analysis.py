#!/usr/bin/env python3


# May first need:
# In your VM: sudo apt-get install libgeos-dev (brew install on Mac)
# pip3 install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np

from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

"""
IMPORTANT
This is EXAMPLE code.
There are a few things missing:
1) You may need to play with the colors in the US map.
2) This code assumes you are running in Jupyter Notebook or on your own system.
   If you are using the VM, you will instead need to play with writing the images
   to PNG files with decent margins and sizes.
3) The US map only has code for the Positive case. I leave the negative case to you.
4) Alaska and Hawaii got dropped off the map, but it's late, and I want you to have this
   code. So, if you can fix Hawaii and Alask, ExTrA CrEdIt. The source contains info
   about adding them back.
"""


"""
PLOT 1: SENTIMENT OVER TIME (TIME SERIES PLOT)
"""
# Assumes a file called time_data.csv that has columns
# date, Positive, Negative. Use absolute path.

ts = pd.read_csv("time_data.csv")
# Remove erroneous row.
ts = ts[ts['date'] != '2018-12-31']

plt.figure(figsize=(12,5))
ts.date = pd.to_datetime(ts['date'], format='%Y-%m-%d')
ts.set_index(['date'],inplace=True)

ax = ts.plot(title="President Trump Sentiment on /r/politics Over Time",
        color=['green', 'red'],
       ylim=(0, 1.05))
ax.plot()
plt.savefig("djt_part1.png")

"""
PLOT 2: SENTIMENT BY STATE (POSITIVE AND NEGATIVE SEPARATELY)
# This example only shows positive, I will leave negative to you.
"""

# This assumes you have a CSV file called "state_data.csv" with the columns:
# state, Positive, Negative
#
# You should use the FULL PATH to the file, just in case.

state_data = pd.read_csv("state_data.csv")

"""
You also need to download the following files. Put them somewhere convenient:
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx
IF YOU USE WGET (CONVERT TO CURL IF YOU USE THAT) TO DOWNLOAD THE ABOVE FILES, YOU NEED TO USE 
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx?raw=true"
The rename the files to get rid of the ?raw=true
"""

# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
pos_data = dict(zip(state_data.state, state_data.Positive))
neg_data = dict(zip(state_data.state, state_data.Negative))

# choose a color for each state based on sentiment.
pos_colors = {}
neg_colors = {}
statenames = []
pos_cmap = plt.cm.Greens # use 'hot' colormap
neg_cmap = plt.cm.Reds


pos_max_key = max(pos_data, key=pos_data.get)
pos_min_key = min(pos_data, key=pos_data.get)

neg_max_key = max(neg_data, key=neg_data.get)
neg_min_key = min(neg_data, key=neg_data.get)

granularity = 3
decimal_places = 1 / (10 ** granularity)
pos_vmin_round = round(pos_data[pos_min_key], granularity)
pos_vmax_round = round(pos_data[pos_max_key], granularity)
neg_vmin_round = round(neg_data[neg_min_key], granularity)
neg_vmax_round = round(neg_data[neg_max_key], granularity)

pos_vmin = (pos_vmin_round - decimal_places) if (pos_vmin_round - decimal_places) > 0 else pos_vmin_round
pos_vmax = (pos_vmax_round + decimal_places) if (pos_vmax_round + decimal_places) > 0 else pos_vmax_round # set range.
neg_vmin = (neg_vmin_round - decimal_places) if (neg_vmin_round - decimal_places) < 1 else neg_vmin_round # set range.
neg_vmax = (neg_vmax_round + decimal_places) if (neg_vmax_round + decimal_places) < 1 else neg_vmax_round


for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = pos_data[statename]
        pos_colors[statename] = pos_cmap(np.sqrt((pos - pos_vmin)/(pos_vmax - pos_vmin)))[:3]
        neg = neg_data[statename]
        neg_colors[statename] = neg_cmap(np.sqrt((neg - neg_vmin)/(neg_vmax - neg_vmin)))[:3]
    statenames.append(statename)


for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = pos_data[statename]
        pos_colors[statename] = pos_cmap(np.sqrt((pos - pos_vmin)/(pos_vmax - pos_vmin)))[:3]
        neg = neg_data[statename]
        neg_colors[statename] = neg_cmap(np.sqrt((neg - neg_vmin)/(neg_vmax - neg_vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.


# pos minus neg
diff_state = {}
diff_cmap = plt.cm.RdYlGn
diff_colors = {}

for state in pos_data:
    if state not in ['District of Columbia', 'Puerto Rico']:
        diff_state[state] = pos_data[state] - neg_data[state]

diff_max_key = max(diff_state, key=diff_state.get)
diff_min_key = min(diff_state, key=diff_state.get)
diff_vmin_round = round(diff_state[diff_min_key], granularity)
diff_vmax_round = round(diff_state[diff_max_key], granularity)


diff_vmin = (diff_vmin_round - decimal_places) if (diff_vmin_round - decimal_places) > -1 else diff_vmin_round
diff_vmax = (diff_vmax_round + decimal_places) if (diff_vmax_round + decimal_places) < 1  else diff_vmax_round


for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        diff = diff_state[statename]
        #diff_colors[statename] = diff_cmap(diff)[:3]
        diff_colors[statename] = diff_cmap(np.sqrt((diff - diff_vmin)/(diff_vmax - diff_vmin))*2-1)[:3]
    statenames.append(statename)



# POSITIVE MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(pos_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Positive Trump Sentiment Across the US')
plt.legend().remove()
plt.savefig("djt_mycoolmap.png")

# NEGATIVE MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(neg_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Negative Trump Sentiment Across the US')
plt.legend().remove()
plt.savefig("djt_mymapisbetterthanyours.png")

#DIFF MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(diff_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Difference Trump Sentiment Across the US')
plt.legend().remove()
plt.savefig("djt_diffmap_thing.png")



#ax2 = plt.gca() 
#for nshape, seg in enumerate(m.states):
#   if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
#        color2 = rgb2hex(neg_colors[statenames[nshape]])
#        poly2 = Polygon(seg, facecolor=color2, edgecolor=color2)
#        ax2.add_patch(poly2)
#plt.title('Negative Trump Sentiment Across the US')
#plt.savefig("mymapisbetterthanyours.png")


# SOURCE: https://stackoverflow.com/questions/39742305/how-to-use-basemap-python-to-plot-us-with-50-states
# (this misses Alaska and Hawaii. If you can get them to work, EXTRA CREDIT)

"""
PART 4 SHOULD BE DONE IN SPARK
"""


"""
PLOT 5A: SENTIMENT BY STORY SCORE
"""
# What is the purpose of this? It helps us determine if the story score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called submission_score.csv with the following coluns
# submission_score, Positive, Negative

story = pd.read_csv("submission_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['submission_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['submission_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('President Trump Sentiment by Submission Score')
plt.ylabel("Percent Sentiment")
plt.savefig("djt_plot5a.png")

"""
PLOT 5B: SENTIMENT BY COMMENT SCORE
"""
# What is the purpose of this? It helps us determine if the comment score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called comment_score.csv with the following columns
# comment_score, Positive, Negative

story = pd.read_csv("comment_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['comment_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['comment_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('President Trump Sentiment by Comment Score')
plt.ylabel("Percent Sentiment")
plt.savefig("djt_plot5b.png")









#GRAND OLD PARTY

"""
PLOT 1: SENTIMENT OVER TIME (TIME SERIES PLOT)
"""
# Assumes a file called time_data.csv that has columns
# date, Positive, Negative. Use absolute path.

ts = pd.read_csv("gop_time_data.csv")
# Remove erroneous row.
ts = ts[ts['date'] != '2018-12-31']

plt.figure(figsize=(12,5))
ts.date = pd.to_datetime(ts['date'], format='%Y-%m-%d')
ts.set_index(['date'],inplace=True)

ax = ts.plot(title="GOP Sentiment on /r/politics Over Time",
        color=['green', 'red'],
       ylim=(0, 1.05))
ax.plot()
plt.savefig("gop_part1.png")

"""
PLOT 2: SENTIMENT BY STATE (POSITIVE AND NEGATIVE SEPARATELY)
# This example only shows positive, I will leave negative to you.
"""

# This assumes you have a CSV file called "state_data.csv" with the columns:
# state, Positive, Negative
#
# You should use the FULL PATH to the file, just in case.

state_data = pd.read_csv("gop_state_data.csv")

"""
You also need to download the following files. Put them somewhere convenient:
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx
IF YOU USE WGET (CONVERT TO CURL IF YOU USE THAT) TO DOWNLOAD THE ABOVE FILES, YOU NEED TO USE 
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx?raw=true"
The rename the files to get rid of the ?raw=true
"""

# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
pos_data = dict(zip(state_data.state, state_data.Positive))
neg_data = dict(zip(state_data.state, state_data.Negative))

# choose a color for each state based on sentiment.
pos_colors = {}
neg_colors = {}
statenames = []
pos_cmap = plt.cm.Greens # use 'hot' colormap
neg_cmap = plt.cm.Reds


pos_max_key = max(pos_data, key=pos_data.get)
pos_min_key = min(pos_data, key=pos_data.get)

neg_max_key = max(neg_data, key=neg_data.get)
neg_min_key = min(neg_data, key=neg_data.get)

granularity = 3
decimal_places = 1 / (10 ** granularity)
pos_vmin_round = round(pos_data[pos_min_key], granularity)
pos_vmax_round = round(pos_data[pos_max_key], granularity)
neg_vmin_round = round(neg_data[neg_min_key], granularity)
neg_vmax_round = round(neg_data[neg_max_key], granularity)

pos_vmin = (pos_vmin_round - decimal_places) if (pos_vmin_round - decimal_places) > 0 else pos_vmin_round
pos_vmax = (pos_vmax_round + decimal_places) if (pos_vmax_round + decimal_places) > 0 else pos_vmax_round # set range.
neg_vmin = (neg_vmin_round - decimal_places) if (neg_vmin_round - decimal_places) < 1 else neg_vmin_round # set range.
neg_vmax = (neg_vmax_round + decimal_places) if (neg_vmax_round + decimal_places) < 1 else neg_vmax_round


for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = pos_data[statename]
        pos_colors[statename] = pos_cmap(np.sqrt((pos - pos_vmin)/(pos_vmax - pos_vmin)))[:3]
        neg = neg_data[statename]
        neg_colors[statename] = neg_cmap(np.sqrt((neg - neg_vmin)/(neg_vmax - neg_vmin)))[:3]
    statenames.append(statename)


for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = pos_data[statename]
        pos_colors[statename] = pos_cmap(np.sqrt((pos - pos_vmin)/(pos_vmax - pos_vmin)))[:3]
        neg = neg_data[statename]
        neg_colors[statename] = neg_cmap(np.sqrt((neg - neg_vmin)/(neg_vmax - neg_vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.


# pos minus neg
diff_state = {}
diff_cmap = plt.cm.RdYlGn
diff_colors = {}

for state in pos_data:
    if state not in ['District of Columbia', 'Puerto Rico']:
        diff_state[state] = pos_data[state] - neg_data[state]

diff_max_key = max(diff_state, key=diff_state.get)
diff_min_key = min(diff_state, key=diff_state.get)
diff_vmin_round = round(diff_state[diff_min_key], granularity)
diff_vmax_round = round(diff_state[diff_max_key], granularity)


diff_vmin = (diff_vmin_round - decimal_places) if (diff_vmin_round - decimal_places) > -1 else diff_vmin_round
diff_vmax = (diff_vmax_round + decimal_places) if (diff_vmax_round + decimal_places) < 1  else diff_vmax_round


for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        diff = diff_state[statename]
        diff_colors[statename] = diff_cmap(np.sqrt((diff - diff_vmin)/(diff_vmax - diff_vmin))*2-1)[:3]
    statenames.append(statename)



# POSITIVE MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(pos_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Positive GOP Sentiment Across the US')
plt.legend().remove()
plt.savefig("gop_mycoolmap.png")

# NEGATIVE MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(neg_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Negative GOP Sentiment Across the US')
plt.legend().remove()
plt.savefig("gop_mymapisbetterthanyours.png")

#DIFF MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(diff_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Difference in GOP Sentiment Across the US')
plt.legend().remove()
plt.savefig("gop_diffmap_thing.png")



#ax2 = plt.gca() 
#for nshape, seg in enumerate(m.states):
#   if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
#        color2 = rgb2hex(neg_colors[statenames[nshape]])
#        poly2 = Polygon(seg, facecolor=color2, edgecolor=color2)
#        ax2.add_patch(poly2)
#plt.title('Negative Trump Sentiment Across the US')
#plt.savefig("mymapisbetterthanyours.png")


# SOURCE: https://stackoverflow.com/questions/39742305/how-to-use-basemap-python-to-plot-us-with-50-states
# (this misses Alaska and Hawaii. If you can get them to work, EXTRA CREDIT)

"""
PART 4 SHOULD BE DONE IN SPARK
"""


"""
PLOT 5A: SENTIMENT BY STORY SCORE
"""
# What is the purpose of this? It helps us determine if the story score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called submission_score.csv with the following coluns
# submission_score, Positive, Negative

story = pd.read_csv("gop_submission_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['submission_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['submission_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('GOP Sentiment by Submission Score')
plt.ylabel("Percent Sentiment")
plt.savefig("gop_plot5a.png")

"""
PLOT 5B: SENTIMENT BY COMMENT SCORE
"""
# What is the purpose of this? It helps us determine if the comment score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called comment_score.csv with the following columns
# comment_score, Positive, Negative

story = pd.read_csv("gop_comment_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['comment_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['comment_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('GOP Sentiment by Comment Score')
plt.ylabel("Percent Sentiment")
plt.savefig("gop_plot5b.png")







#DEMOCRATS 4 LYFE

"""
PLOT 1: SENTIMENT OVER TIME (TIME SERIES PLOT)
"""
# Assumes a file called time_data.csv that has columns
# date, Positive, Negative. Use absolute path.

ts = pd.read_csv("dem_time_data.csv")
# Remove erroneous row.
ts = ts[ts['date'] != '2018-12-31']

plt.figure(figsize=(12,5))
ts.date = pd.to_datetime(ts['date'], format='%Y-%m-%d')
ts.set_index(['date'],inplace=True)

ax = ts.plot(title="Democrat Sentiment on /r/politics Over Time",
        color=['green', 'red'],
       ylim=(0, 1.05))
ax.plot()
plt.savefig("dem_part1.png")

"""
PLOT 2: SENTIMENT BY STATE (POSITIVE AND NEGATIVE SEPARATELY)
# This example only shows positive, I will leave negative to you.
"""

# This assumes you have a CSV file called "state_data.csv" with the columns:
# state, Positive, Negative
#
# You should use the FULL PATH to the file, just in case.

state_data = pd.read_csv("dem_state_data.csv")

"""
You also need to download the following files. Put them somewhere convenient:
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx
IF YOU USE WGET (CONVERT TO CURL IF YOU USE THAT) TO DOWNLOAD THE ABOVE FILES, YOU NEED TO USE 
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx?raw=true"
The rename the files to get rid of the ?raw=true
"""

# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
pos_data = dict(zip(state_data.state, state_data.Positive))
neg_data = dict(zip(state_data.state, state_data.Negative))

# choose a color for each state based on sentiment.
pos_colors = {}
neg_colors = {}
statenames = []
pos_cmap = plt.cm.Greens # use 'hot' colormap
neg_cmap = plt.cm.Reds


pos_max_key = max(pos_data, key=pos_data.get)
pos_min_key = min(pos_data, key=pos_data.get)

neg_max_key = max(neg_data, key=neg_data.get)
neg_min_key = min(neg_data, key=neg_data.get)

granularity = 3
decimal_places = 1 / (10 ** granularity)
pos_vmin_round = round(pos_data[pos_min_key], granularity)
pos_vmax_round = round(pos_data[pos_max_key], granularity)
neg_vmin_round = round(neg_data[neg_min_key], granularity)
neg_vmax_round = round(neg_data[neg_max_key], granularity)

pos_vmin = (pos_vmin_round - decimal_places) if (pos_vmin_round - decimal_places) > 0 else pos_vmin_round
pos_vmax = (pos_vmax_round + decimal_places) if (pos_vmax_round + decimal_places) > 0 else pos_vmax_round # set range.
neg_vmin = (neg_vmin_round - decimal_places) if (neg_vmin_round - decimal_places) < 1 else neg_vmin_round # set range.
neg_vmax = (neg_vmax_round + decimal_places) if (neg_vmax_round + decimal_places) < 1 else neg_vmax_round


for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = pos_data[statename]
        pos_colors[statename] = pos_cmap(np.sqrt((pos - pos_vmin)/(pos_vmax - pos_vmin)))[:3]
        neg = neg_data[statename]
        neg_colors[statename] = neg_cmap(np.sqrt((neg - neg_vmin)/(neg_vmax - neg_vmin)))[:3]
    statenames.append(statename)


for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = pos_data[statename]
        pos_colors[statename] = pos_cmap(np.sqrt((pos - pos_vmin)/(pos_vmax - pos_vmin)))[:3]
        neg = neg_data[statename]
        neg_colors[statename] = neg_cmap(np.sqrt((neg - neg_vmin)/(neg_vmax - neg_vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.


# pos minus neg
diff_state = {}
diff_cmap = plt.cm.RdYlGn
diff_colors = {}

for state in pos_data:
    if state not in ['District of Columbia', 'Puerto Rico']:
        diff_state[state] = pos_data[state] - neg_data[state]

diff_max_key = max(diff_state, key=diff_state.get)
diff_min_key = min(diff_state, key=diff_state.get)
diff_vmin_round = round(diff_state[diff_min_key], granularity)
diff_vmax_round = round(diff_state[diff_max_key], granularity)


diff_vmin = (diff_vmin_round - decimal_places) if (diff_vmin_round - decimal_places) > -1 else diff_vmin_round
diff_vmax = (diff_vmax_round + decimal_places) if (diff_vmax_round + decimal_places) < 1  else diff_vmax_round


for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        diff = diff_state[statename]
        diff_colors[statename] = diff_cmap(np.sqrt((diff - diff_vmin)/(diff_vmax - diff_vmin))*2-1)[:3]
    statenames.append(statename)



# POSITIVE MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(pos_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Positive Democrat Sentiment Across the US')
plt.legend().remove()
plt.savefig("dem_mycoolmap.png")

# NEGATIVE MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(neg_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Negative Democrat Sentiment Across the US')
plt.legend().remove()
plt.savefig("dem_mymapisbetterthanyours.png")

#DIFF MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(diff_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Difference in Democrat Sentiment Across the US')
plt.legend().remove()
plt.savefig("dem_diffmap_thing.png")



#ax2 = plt.gca() 
#for nshape, seg in enumerate(m.states):
#   if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
#        color2 = rgb2hex(neg_colors[statenames[nshape]])
#        poly2 = Polygon(seg, facecolor=color2, edgecolor=color2)
#        ax2.add_patch(poly2)
#plt.title('Negative Trump Sentiment Across the US')
#plt.savefig("mymapisbetterthanyours.png")


# SOURCE: https://stackoverflow.com/questions/39742305/how-to-use-basemap-python-to-plot-us-with-50-states
# (this misses Alaska and Hawaii. If you can get them to work, EXTRA CREDIT)

"""
PART 4 SHOULD BE DONE IN SPARK
"""


"""
PLOT 5A: SENTIMENT BY STORY SCORE
"""
# What is the purpose of this? It helps us determine if the story score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called submission_score.csv with the following coluns
# submission_score, Positive, Negative

story = pd.read_csv("dem_submission_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['submission_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['submission_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('Democrat Sentiment by Submission Score')
plt.ylabel("Percent Sentiment")
plt.savefig("dem_plot5a.png")

"""
PLOT 5B: SENTIMENT BY COMMENT SCORE
"""
# What is the purpose of this? It helps us determine if the comment score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called comment_score.csv with the following columns
# comment_score, Positive, Negative

story = pd.read_csv("dem_comment_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['comment_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['comment_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('Democrat Sentiment by Comment Score')
plt.ylabel("Percent Sentiment")
plt.savefig("dem_plot5b.png")
