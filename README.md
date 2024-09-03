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

scrape_async.py scrapes multiple years in parallel using asyncio, whilst scrape.py goes through the scraping linearly