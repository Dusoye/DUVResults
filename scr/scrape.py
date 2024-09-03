import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor
import time

async def fetch_html(url, session):
    """Fetch the content of a URL asynchronously."""
    try:
        async with session.get(url, timeout=30) as response:
            if response.status == 200:
                return await response.text()
            else:
                print(f"Error fetching {url}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return None

def parse_html(html):
    """Parse HTML content using BeautifulSoup."""
    return BeautifulSoup(html, 'lxml') if html else None

async def extract_page_count(session, url):
    """Extract the number of pages from the pagination element."""
    html = await fetch_html(url, session)
    if html:
        soup = BeautifulSoup(html, 'lxml')
        pagination = soup.find('div', class_='pagination')
        return int(pagination.find_all('a')[-2].text) if pagination else 1
    return 1

async def extract_event_links(session, base_url, page_url):
    """Extract event links from the page."""
    html = await fetch_html(page_url, session)
    if html:
        soup = BeautifulSoup(html, 'lxml')
        return [base_url + link['href'] for link in soup.find_all('a', href=True) if 'getresultevent.php?event=' in link['href']]
    return []

async def extract_elevation_gain(soup):
    """Extract elevation gain from the event detail page."""
    elevation_row = soup.find('b', string=re.compile('Elevation gain/loss: '))
    if elevation_row:
        elevation_data = elevation_row.find_next('td')
        if elevation_data:
            return elevation_data.text.strip()
    return 'N/A'

async def extract_event_details(soup):
    """Extract 'Event', 'Date', 'Finishers' and 'Distance' from the HTML content."""
    details = {}
    try:
        info_rows = soup.find_all('tr')
        for row in info_rows:
            header_cell = row.find('td')
            if header_cell and header_cell.find('b'):
                label = header_cell.get_text(strip=True).rstrip(':')
                value_cell = header_cell.find_next_sibling('td')
                if label in ['Date', 'Event', 'Distance', 'Finishers'] and value_cell:
                    details[label] = value_cell.get_text(strip=True)
    except Exception as e:
        print(f"Error extracting event details: {e}")
    return details

async def extract_event_id(url):
    """Extract event ID from the URL."""
    match = re.search(r'event=(\d+)', url)
    return match.group(1) if match else 'N/A'

async def fetch_event_all_data(table_soup, event_details, elevation_gain, event_id):
    """Extract data and runner IDs from the event table, including event details and elevation gain."""
    data = []
    headers = [th.text.strip() for th in table_soup.find_all('th')]
    headers.extend(['Runner ID', 'Event', 'Date', 'Distance', 'Finishers', 'Winner Time', 'Elevation Gain', 'Event ID'])
    rows = table_soup.find_all('tr')[1:]
    winner_time = None

    for row in rows:
        cols = row.find_all('td')
        rank = int(cols[0].text.strip()) if cols[0].text.strip().isdigit() else None
        if rank == 1:
            winner_time = cols[1].text.strip()
            break

    for row in rows:
        cols = row.find_all('td')
        row_data = [col.text.strip() for col in cols]
        link = cols[2].find('a', href=True)
        runner_id = link['href'].split('runner=')[-1] if link else 'No ID'
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

async def process_event(session, event_link, base_url):
    """Process a single event asynchronously."""
    event_html = await fetch_html(event_link, session)
    if not event_html:
        return None

    event_soup = parse_html(event_html)
    event_details = extract_event_details(event_soup)
    table_soup = event_soup.find('table', {'id': 'Resultlist'})

    event_id = extract_event_id(event_link)
    event_detail_url = f"{base_url}eventdetail.php?event={event_id}"
    event_detail_html = await fetch_html(event_detail_url, session)
    event_detail_soup = parse_html(event_detail_html)
    elevation_gain = extract_elevation_gain(event_detail_soup) if event_detail_soup else 'N/A'

    if table_soup:
        return fetch_event_all_data(table_soup, event_details, elevation_gain, event_id)
    return None

async def scrape_year(year, base_url):
    """Scrape data for a specific year."""
    async with aiohttp.ClientSession() as session:
        first_page_url = f"{base_url}geteventlist.php?year={year}&dist=all&country=all&surface=all&sort=1&page=1"
        num_pages = await extract_page_count(session, first_page_url)
        
        all_event_links = []
        for page in range(1, num_pages + 1):
            page_url = f"{base_url}geteventlist.php?year={year}&dist=all&country=all&surface=all&sort=1&page={page}"
            event_links = await extract_event_links(session, base_url, page_url)
            all_event_links.extend(event_links)

        tasks = [process_event(session, link, base_url) for link in all_event_links]
        results = await asyncio.gather(*tasks)
        
        return [result for result in results if result is not None]

async def scrape_year(year, base_url):
    async with aiohttp.ClientSession() as session:
        first_page_url = f"{base_url}geteventlist.php?year={year}&dist=all&country=all&surface=all&sort=1&page=1"
        num_pages = await extract_page_count(session, first_page_url)
        all_data = []

        for page in range(1, num_pages + 1):
            page_url = f"{base_url}geteventlist.php?year={year}&dist=all&country=all&surface=all&sort=1&page={page}"
            print(f"Processing year {year}, page {page}")
            event_links = await extract_event_links(session, base_url, page_url)
            for event_link in event_links:
                event_html = await fetch_html(event_link, session)
                if event_html:
                    event_soup = BeautifulSoup(event_html, 'lxml')
                    event_details = await extract_event_details(event_soup)
                    table_soup = event_soup.find('table', {'id': 'Resultlist'})
                    event_id = await extract_event_id(event_link)
                    event_detail_url = f"{base_url}eventdetail.php?event={event_id}"
                    event_detail_html = await fetch_html(event_detail_url, session)
                    elevation_gain = await extract_elevation_gain(BeautifulSoup(event_detail_html, 'lxml')) if event_detail_html else 'N/A'
                    if table_soup:
                        event_data = await fetch_event_all_data(table_soup, event_details, elevation_gain, event_id)
                        if not event_data.empty:
                            all_data.append(event_data)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df.to_csv(f'./output/all_events_data_{year}.csv', index=False)
            print(f"Saved all event data for {year} to 'all_events_data_{year}.csv'.")
        else:
            print(f"No data was extracted for {year}.")

async def main():
    base_url = "https://statistik.d-u-v.org/"
    start_year = 2020
    end_year = 2023
    
    start_time = time.time()
    
    tasks = [scrape_year(year, base_url) for year in range(start_year, end_year + 1)]
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
