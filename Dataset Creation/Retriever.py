class Retriever:
  def __init__(self):
    pass


  def get_reviews_ids_from_movie_id(self, movie_id):

    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from webdriver_manager.chrome import ChromeDriverManager
    import re

    options = webdriver.ChromeOptions()
    options.headless = True
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = f'https://www.imdb.com/title/tt{movie_id}/reviews/'
    text_to_find = '/review/rw'

    try:
        driver.get(url)

        # Cerca e clicca il bottone "Tutto" o "Altri"
        try:
            # Aspetta e clicca il bottone "Tutto"
            load_all_button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, '//button//span[contains(text(), "Tutto")]'))
            )
            driver.execute_script("arguments[0].click();", load_all_button)
        except TimeoutException:
            # Se "Tutto" non è presente, prova con "Altri"
            try:
                load_all_button_altro = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.XPATH, '//button//span[contains(text(), "Altri")]'))
                )
                driver.execute_script("arguments[0].click();", load_all_button_altro)
            except TimeoutException:
                print('Less than 25 reviews available - skipped')
                driver.quit()
                return None

        # Aspetta che il caricamento delle recensioni sia completato
        WebDriverWait(driver, 3600).until(
            EC.invisibility_of_element_located((By.CLASS_NAME, "ipc-see-more"))
        )

    except Exception as e:
        print('execution interrupted - was taking too long')
        

    # Estrai i numeri dalle recensioni
    html = driver.page_source
    pattern = re.compile(rf'{re.escape(text_to_find)}(\d+)')
    matches = pattern.findall(html)

    driver.quit()
    return set(matches) if matches else None



  def get_reviews_dataframe_from_set_of_review_ids(self, review_ids):
    import re
    import tqdm
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    
    # Initialize a list to store the review data
    reviews_data = []

    # Base URL for the reviews
    base_url = "https://www.imdb.com/review/"

    # Make sure the browser language is set in EN
    headers = {
    "Accept-Language": "en-US,en;q=0.5",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }


    # Loop through the review IDs with tqdm for the progress bar
    for review_id in tqdm.tqdm(review_ids, desc='retrieving reviews', unit='review'):
        review_id_str = str(review_id)
        url = base_url + f"rw{review_id_str}"
    
        try:
            # Send a GET request to the URL
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                continue  # Skip if the page does not exist

            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract movie data (title and ID)
            movie_tag = soup.find('a', href=lambda x: x and x.startswith('/title/tt'))
            movie_title = movie_tag.text.strip() if movie_tag else None  # Extract movie name
            movie_id = movie_tag['href'].split('/')[2].split('?')[0] if movie_tag else None  # Extract movie ID

            # Validate that the title is a movie
            if movie_id:
                title_url = f"https://www.imdb.com/title/{movie_id}/"
                title_response = requests.get(title_url)
                title_soup = BeautifulSoup(title_response.text, 'html.parser')

            # Extract review details
            rating_tag = soup.find('span', {'class': 'rating-other-user-rating'})
            rating = int(rating_tag.find('span').text) if rating_tag else None
            review_date = soup.find('span', {'class': 'review-date'}).text.strip() if soup.find('span', {'class': 'review-date'}) else None
            review_title_tag = soup.find('a', {'class': 'title'})
            review_title = review_title_tag.text.strip() if review_title_tag else None
            review_text = soup.find('div', {'class': 'text show-more__control'}).text.strip() if soup.find('div', {'class': 'text show-more__control'}) else None

            # Extract helpfulness votes
            helpfulness_tag = soup.find('div', {'class': 'actions text-muted'})
            if helpfulness_tag:
                helpfulness_text = helpfulness_tag.text.strip()  # Get the text content
                match = re.search(r'(\d+) out of (\d+) found this helpful', helpfulness_text)
                if match:
                     helpful_votes = int(match.group(1))  # M: Number of positive voters
                     total_votes = int(match.group(2))   # N: Total number of voters
                else:
                    helpful_votes = None
                    total_votes = None
            else:
                helpful_votes = None
                total_votes = None


            # Append the extracted data to the list
            reviews_data.append({
                "Review_ID": review_id_str,
                "Movie_ID": movie_id,
                "Movie_Title": movie_title,
                "Rating": rating,
                "Review_Date": review_date,
                "Review_Title": review_title,
                "Review_Text": review_text,
                "Helpful_Votes": helpful_votes,
                "Total_Votes": total_votes
            })

        except Exception as e:
            print(f"Error processing review ID {review_id_str}: {e}")

    # Convert the list of dictionaries to a DataFrame
    reviews_df = pd.DataFrame(reviews_data)

    # Save the dataframe to a CSV file
    #reviews_df.to_csv("imdb_reviews_movies_only.csv", index=False)

    #print("Finished scraping. The data is saved to 'imdb_reviews_movies_only.csv'.")

    return reviews_df


