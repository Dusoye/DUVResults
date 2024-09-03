import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd
import re

def fetch_html(url):
    """Fetch the content of a URL with retries and backoff."""
    session = requests.Session()
    # Setup retry strategy
    retries = Retry(
        total=5,  # Total retries
        backoff_factor=1,  # Time between retries, exponential backoff factor
        status_forcelist=[500, 502, 503, 504, 429],  # Retry on these status codes
    )
    # Mount it for both http and https connections
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        response = session.get(url, timeout=10)  # 10 seconds timeout for the request
        if response.ok:
            return BeautifulSoup(response.content, 'html.parser')
        else:
            response.raise_for_status()  # This will raise an error for 4XX client errors
    except requests.RequestException as e:
        print(f"Error fetching the URL {url}: {e}")
        return None

def extract_page_count(soup):
    """Extract the number of pages from the pagination element."""
    pagination = soup.find('div', class_='pagination')
    return int(pagination.find_all('a')[-2].text) if pagination else 1

def extract_event_links(base_url, soup):
    """ Extract event links from the page. """
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if 'getresultevent.php?event=' in href:
            links.append(base_url + href)
    return links

def extract_event_details(soup):
    """Extract 'Event', 'Date', 'Finishers' and 'Distance' from the HTML content."""
    details = {}
    try:
        info_rows = soup.find_all('tr')  # Find all table rows in the page
        for row in info_rows:
            # Look for rows where the first cell contains the labels we're interested in
            header_cell = row.find('td')
            if header_cell and header_cell.find('b'):  # Check for bold tags which might contain labels
                label = header_cell.get_text(strip=True).rstrip(':')
                value_cell = header_cell.find_next_sibling('td')  # Get the next sibling cell for the value
                if label in ['Date', 'Event', 'Distance', 'Finishers'] and value_cell:
                    details[label] = value_cell.get_text(strip=True)
    except Exception as e:
        print(f"Error extracting event details: {e}")
    return details

def extract_elevation_gain(soup):
    """Extract elevation gain from the event detail page."""
    elevation_row = soup.find('b', string=re.compile('Elevation gain/loss: '))
    if elevation_row:
        elevation_data = elevation_row.find_next('td')
        if elevation_data:
            return elevation_data.text.strip()
    return 'N/A'

def extract_event_id(url):
    """Extract event ID from the URL."""
    match = re.search(r'event=(\d+)', url)
    return match.group(1) if match else 'N/A'

def fetch_event_all_data(table_soup, event_details, elevation_gain, event_id):
    """Extract data and runner IDs from the event table, including event details and elevation gain."""
    data = []
    headers = [th.text.strip() for th in table_soup.find_all('th')]
    # Append the event details headers
    headers.extend(['Runner ID', 'Event', 'Date', 'Distance', 'Finishers', 'Winner Time', 'Elevation Gain', 'Event ID'])
    rows = table_soup.find_all('tr')[1:]  # Skip header row
    winner_time = None

    # Extract winners time
    for row in rows:
        cols = row.find_all('td')
        rank = int(cols[0].text.strip()) if cols[0].text.strip().isdigit() else None

        if rank == 1:
            winner_time = cols[1].text.strip() 
            break

    for row in rows:
        cols = row.find_all('td')
        row_data = [col.text.strip() for col in cols]
        # Get runner ID
        link = cols[2].find('a', href=True)  # Assuming the third column has the link
        runner_id = link['href'].split('runner=')[-1] if link else 'No ID'
        # Include event details, runner ID, elevation gain, and event ID
        row_data.extend([
            runner_id, 
            event_details.get('Event', 'N/A'), 
            event_details.get('Date', 'N/A'), 
            event_details.get('Distance', 'N/A'), 
            event_details.get('Finishers', 'N/A'), 
            winner_time or 'N/A',
            elevation_gain,
            event_id
        ])
        data.append(row_data)

    return pd.DataFrame(data, columns=headers)

def extract_specific_links(soup, base_url, path_starts_with):
    """Extract specific links that start with a given path from the parsed HTML."""
    links = []
    if soup:
        # Find all 'a' tags with an 'href' attribute
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Check if the href starts with the desired path
            if href.startswith(path_starts_with):
                full_link = base_url + href
                links.append(full_link)
    return links

# Scraping year by year for every race
base_url = "https://statistik.d-u-v.org/"
start_year = 2024
end_year = 2024

for year in range(start_year, end_year + 1):
    page_url = f"{base_url}geteventlist.php?year={year}&dist=all&country=all&surface=all&sort=1&page=1"
    first_page = fetch_html(page_url)

    if first_page:
        num_pages = extract_page_count(first_page)
        all_data = []

        for page in range(1, num_pages + 1):
            page_url = f"{base_url}geteventlist.php?year={year}&dist=all&country=all&surface=all&sort=1&page={page}"
            print(page_url)
            page_soup = fetch_html(page_url)
            if page_soup:
                event_links = extract_event_links(base_url, page_soup)
                for event_link in event_links:
                    event_page = fetch_html(event_link)
                    if event_page:
                        event_details = extract_event_details(event_page)
                        table_soup = event_page.find('table', {'id': 'Resultlist'})
                        
                        # Extract event ID
                        event_id = extract_event_id(event_link)
                        
                        # Fetch elevation gain from event detail page
                        event_detail_url = f"{base_url}eventdetail.php?event={event_id}"
                        event_detail_page = fetch_html(event_detail_url)
                        elevation_gain = extract_elevation_gain(event_detail_page) if event_detail_page else 'N/A'
                        
                        if table_soup:
                            event_data = fetch_event_all_data(table_soup, event_details, elevation_gain, event_id)
                            if not event_data.empty:
                                all_data.append(event_data)
        
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df.to_csv(f'./output/all_events_data_{year}.csv', index=False)
            print(f"Saved all event data for {year} to 'all_events_data_{year}.csv'.")
        else:
            print(f"No data was extracted for {year}.")
