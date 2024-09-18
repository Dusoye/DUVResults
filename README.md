### DUV ultramarathon dataset scrape

The [DUV](https://statistik.d-u-v.org/) (Deutsche Ultramarathon-Vereinigung) has an incredible dataset of almost 100,000 ultramarathon results dating back to 1798, and so this python script scrapes the set by year to provide a comprehensive dataset to analyse. It aims to include all races that are greater than 45km in length, or for timed events it includes all performances that were over 45km. Backyard/elimination races, relay races and virtual runs are not included. It also doesn't include any runner who DNF's the race, which would also be interesting to see. There will be some occasions where the data is inaccurate and so some cleaning will be required of the collated data.

More information about the data can be found at https://statistik.d-u-v.org/faq.php 

The columns per results table can differ slightly, depending on the type of event, but includes:

- Rank
- Performance
- "Original nameSurnamefirst name"
- Club
- Nat.
- YOB
- M/F
- Rank M/F
- Cat
- Cat. Rank
- Avg.Speed km/h
- Age graded performance
- Runner ID
- Event
- Date
- Distance
- Finishers
- Winner Time
- Elevation Gain
- Event ID
- "Surnamefirst name"
- hours

scrape_async.py scrapes multiple years in parallel using asyncio, whilst scrape.py goes through the scraping linearly.

The Clean data notebook cleans and engineers some features for the dataset, resulting in the following columns:

- Runner ID: Unique identifier for each runner
- First Name: Runner's first name
- Surname: Runner's last name
- Nat.: Nationality of the runner
- M/F: Gender of the runner (Male/Female)
- Age: Age of the runner at the time of the race
- Age Group: Categorized age group of the runner (e.g., 20-29, 30-39, etc.)
- Cat: Race category the runner competed in
- YOB: Year of Birth of the runner
- Race Count: Number of races the runner has participated in (including the current race)
- Cumulative Distance KM: Total distance in kilometers the runner has raced up to and including the current race
- Avg Winner Percentage: Average of the runner's performance relative to the winner across all their races
- Event ID: Unique identifier for each race event
- Event: Name of the race event
- Event Type: Type of race (Distance or Time based)
- Date: Date of the race
- Race Location: Location where the race took place
- Elevation Gain: Total elevation gain of the race course in meters
- Elevation Gain per KM: Average elevation gain per kilometer for the race
- Finishers: String representation of total finishers, including breakdown by gender
- Total Finishers: Total number of runners who completed the race
- Male Finishers: Number of male runners who completed the race
- Female Finishers: Number of female runners who completed the race
- Rank: Overall finishing position of the runner in the race
- Rank M/F: Finishing position of the runner within their gender category
- Cat. Rank: Finishing position of the runner within their specific race category
- Finish Percentage: Runner's finishing position as a percentage of total finishers
- Winner Percentage: Runner's performance as a percentage relative to the winner's performance
- Distance/Time: The set distance or time for the race
- Distance KM: Race distance in kilometers
- Terrain: Type of terrain for the race (e.g., trail, road, track)
- Time Seconds Finish: Runner's finishing time in seconds
- Distance Finish: Distance covered by the runner in time-based races
- Average Speed: Runner's average speed in meters per second
- Avg.Speed km/h: Runner's average speed in kilometers per hour
