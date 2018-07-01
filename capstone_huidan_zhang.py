"""
ALY6983 ST: Python for Data Science
Capstone Project

Northeastern University
Instructor: Joel Schwartz


Written by: Huidan Zhang
June 2018

Python Version: 3.6
"""

"""
My function does web-scrapping on the Security Exchange Commission (SEC)’s Edgar database and extract a full list 
of a public company’s subsidiaries, including the subsidiaries’ names and countries of jurisdiction. A public company 
is required to file Exhibit-21 as a part of its annual report (Form 10-K), where it lists out the subsidiaries of the 
public company. Such information is available on https://www.sec.gov/edgar/searchedgar/companysearch.html by 
searching for a public company by its stock ticker. A user has to navigate through several webpages to get the 
Exhibit 21 in a year of 10-K filing. The Exhibit 21 is also in html format. My function facilitates this process by 
scraping the previous pages and generating the URL that eventually goes to Exhibit 21, then scrapes the html 
subsidiaries list into a clean dataframe. An option is also provided whether or not to save the dataframe as a local 
csv file. Due to the fact that the format of Exhibit 21 is pretty flexible, it requires some exception processing and 
cleaning to output any company’s subsidiaries in the same format. My function has generalized this process to get any 
public company’s list of subsidiaries, only by inputting the ticker. 

This project is a good start for any further analysis on subsidiaries, or scraping other important information 
available in the e-filings of a public company. 
"""

"""
Please import all the functions and packages. Any function is dependent on its previous functions. All functions have 
to be imported to get the "main" function work. 
"""


import pandas as pd
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim


def parse_html(url):
    """
    This function performs the preparation steps in parsing an HTML page.

    :param url: the URL of the page to be parsed
    :return: the parsed HTML page, a bs4.beautifulsoup object
    """

    r = requests.get(url, timeout=10)  # get the HTML page with 10 seconds timeout

    # check the HTTP status code and proceed accordingly
    if r.status_code != 200:
        return print("The page is not correctly connected.")  # the page is not correctly connected

    else:
        soup = BeautifulSoup(r.content, "html.parser")  # use "BeautifulSoup" to parse the html page
        return soup


def company_url(ticker):
    """
    This function gets the URL link to a public company's information page on SEC's Edgar database. The public company
    is designated by its stock ticker.

    :param ticker: public company's stock ticker
    :return: the URL of company information page on SEC
    """
    # insert the ticker symbol into the SEC's standardized link
    url = "https://www.sec.gov/cgi-bin/browse-edgar?CIK={}&owner=exclude&action=getcompany".format(ticker)

    # when the URL is supplied with an incorrect ticker, the website will always show a warning message have an "h1" tag
    # if the ticker is correct, the web-page doesn't have "h1" tag.
    if parse_html(url).find("h1") is None:
        return url  # catch un-matching ticker

    else:
        return "No matching Ticker Symbol."  # return the URL of company information page on SEC


def get_cik(ticker):
    """
    This function takes a ticker and finds the CIK number (a type of reference number at SEC) of that public company
    from its company's info page (derived from company_url() function). The CIK number is needed in the URL that goes
    to this company's list of filings (the next function).

    :param ticker: public company's stock ticker
    :return: the public company's CIK number (a type of reference number) at SEC
    """

    if company_url(ticker) == "No matching Ticker Symbol.":
        return "No matching Ticker Symbol."  # catch un-matching ticker

    else:
        # the CIK number is embedded in a string in an "a" tag in the HTML of the company's info page
        # use BeautifulSoup web-scrapping functions to locate the tag and extract part of the string
        # the location of the CIK number in any company's info page is identical
        cik = parse_html(company_url(ticker)).find("span", {"class": "companyName"}).a.text[:-26]  # extract the CIK
        return cik  # return the company's CIK number


def get_filings_url(ticker, filing_type="10-K", date="", results_count="10"):
    """
    This function takes a company's ticker and gets the URL that goes to a public company's page of filings. When
    defaulted, the page contains the list of the 10 most recent 10-K filings of the company designated by the ticker.

    :param ticker: public company's stock ticker
    :param filing_type: filing type available at SEC, such as 10-K, 10-Q, etc. Defaulted as "10-K".
    :param date: the filings prior to this date (YYYYMMDD). Defaulted as no specified date.
    :param results_count: the number of results per page (10, 20, 40, 80 or 100). Defaulted  as 10.
    :return: url that goes to the filings list web-page that match the above conditions in SEC's EDGAR database
    """

    # insert the parameters into the SEC's URL to get the URL that goes to the company's filings list page
    url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={type}&dateb={date}&owner=exclude&" \
          "count={count}".format(cik=get_cik(ticker), type=filing_type, date=date, count=results_count)

    soup = parse_html(url)
    t = soup.find("table", {"class": "tableFile2"})

    if company_url(ticker) == "No matching Ticker Symbol.":
        return "No matching Ticker Symbol."  # catch un-matching ticker

    elif len(t.find_all("tr")) == 1:
        return "No records."  # if there is no filings

    else:
        return url  # return the SEC URL


def latest_filing_url(ticker):
    """
    This function takes a company's ticker and extract a URL of a company's most recent filing (specified by
    get_filings_url() function). By default, it extracts the URL that goes to this company's most recent 10-K filing,
    where it contains the list of exhibits in this 10-K.

    :param ticker: public company's stock ticker
    :return: url that goes to a SEC's Edgar page containing the content of the company's most recent filing
    """

    if company_url(ticker) == "No matching Ticker Symbol.":
        return "No matching Ticker Symbol."  # catch un-matching ticker

    elif get_filings_url(ticker) == "No records.":
        return "No records."  # if there is no filings

    else:
        # Scrape the filings list webpage to extract the link that goes to the most recent filing
        url = get_filings_url(ticker)  # gets the URL of filings list
        soup = parse_html(url)  # prepare to parse the webpage

        filings_table = soup.find_all("table", {"class": "tableFile2"})[0]  # find the table that contains the filings
        href_tags = filings_table.find_all("a", href=True)  # extract all the links of the filings
        link = "https://www.sec.gov" + href_tags[0]["href"]  # "[0]" gets the link to the most recent (1st) 10-k filing

        return link


def get_ex21_url(ticker):
    """
    This function parses the page of a company's filing and extracts the URL to Exhibit 21.

    :param ticker: public company's stock ticker
    :return: url that goes to the Exhibit.21 (subsidiary info) of the 10-K filing of a company on SEC EDGAR database
    """

    if company_url(ticker) == "No matching Ticker Symbol.":
        return "No matching Ticker Symbol."  # catch un-matching ticker

    elif get_filings_url(ticker) == "No records.":
        return "No records."  # if there is no filings

    else:
        # the content of a filing is stored in a table format on the page
        latest_filing_table = pd.read_html(latest_filing_url(ticker), header=0)[0]  # extract the HTML table
        latest_filing_table.columns = ['Seq', 'Description', 'Document', 'Type', 'Size']  # rename the column indexes

        ex21_row = latest_filing_table.loc[latest_filing_table["Type"].str.contains("EX-21") == True].reset_index(
            drop=True)  # extract the row that stores Exhibit 21
        ex21_doc = ex21_row.loc[0, "Document"]  # keep only the "Document" cell (string)

        # the "Document" cell contains the link to the details of Exhibit 21 (list of subsidiaries).
        soup = parse_html(latest_filing_url(ticker))  # parse the HTML using BeautifulSoup
        link = soup.body.find(text=ex21_doc).parent["href"]  # find the link embedded in the designated location
        complete_link = "http://www.sec.gov" + link  # get the complete URL

        return complete_link


def huidan_main(ticker, save_as_csv=False, filename=""):
    """
    This is the main function, which utilizes the results of previous functions to scrape the HTML table of a public
    company's subsidiaries. This function cleans the HTML table and output a consistent dataframe for any company.

    :param ticker: public company's stock ticker
    :param save_as_csv: (optional)save the output dataframe as csv, defaulted at False
    :param filename: (only requires when save_as_csv=True)name of the csv file
    :return:
    """

    if company_url(ticker) == "No matching Ticker Symbol.":
        return "No matching Ticker Symbol."  # catch un-matching ticker

    elif get_filings_url(ticker) == "No records.":
        return "No records."  # if there is no filings

    else:
        ex21_url = get_ex21_url(ticker)  # the url to be parsed: the url where the EX.21 presents

        try:
            tables = pd.read_html(ex21_url)  # a list of tables of subsidiaries in the webpage

            # clean the html tables
            for i in range(len(tables)):  # for each table
                tables[i] = tables[i].dropna(how="all", axis='columns')  # remove empty columns
                tables[i] = tables[i].dropna(how="all", axis='rows')  # remove empty rows
                tables[i] = tables[i].T.reset_index(drop=True).T  # reset columns indexes
                tables[i] = tables[i].reset_index(drop=True)  # reset row indexes

                # Exhibit.21 requires a company to report the subsidiary name and its jurisdiction country,
                # and optionally its vote(in %, eg. 100) to that subsidiary.
                # table may contains columns of indexes (primary key), sub's name, jurisdiction and vote
                # for consistent outcomes, only columns of keep sub's name and jurisdiction

                if len(tables[i].columns) > 1:  # when not all information is stored in one column

                    # remove Column 0 if it is numeric (original index in HTML table)
                    for r in range(min(5, len(tables[i]))):
                        if type(tables[i].iat[r, 0]) == int or type(tables[i].iat[r, 0]) == float:
                            tables[i] = tables[i].drop(0, axis=1)
                        break

                    tables[i] = tables[i].T.reset_index(drop=True).T  # reset column indexes

                    if len(tables[i].columns) > 1:  # check remaining columns
                        tables[i] = tables[i][[0, 1]]  # keeps only the first 2 columns where it should be sub name and country

                    # table format B: Column 0 - subsidiary name and country; Column 1 - vote
                    # "vote" column is generally numeric
                        if map(lambda x: x.numeric(), tables[i][1]) is True:
                            tables[i] = tables[i].drop(1, 1)  # remove Column 1 if it is for "vote"

                        if len(tables[i].columns) > 1:  # check remaining columns

                            # remove the original title of the HTML table by iterating through typical words in title
                            for word in ["jurisdiction", "organization", "incorporation", "corporation", "subsidiar",
                                         "formation"]:
                                try:
                                    for r in range(len(tables[i])):
                                        if word in str(tables[i].iat[r, 1]).lower():
                                            tables[i] = tables[i].drop(tables[i].index[r])

                                except:
                                    tables[i] = tables[i].reset_index(drop=True)  # reset row indexes

                            # check if Column 1 is actually a valid location for jurisdiction
                            # to get rid of other unexpected information
                            geolocator = Nominatim()  # use geo info provided by Nominatim
                            try:
                                for r in range(min(5, len(tables[i]))):
                                    if geolocator.geocode(tables[i].iat[r, 1]) is None:  # not a valid location
                                        tables[i] = tables[i].drop(tables[i].index[r])  # drop rows with invalid location

                            except:
                                break

            # store the clean table(s) into a new dataframe
            df = pd.DataFrame()
            for table in tables:
                df = df.append(table, ignore_index=True)

            # clean the dataframe
            df = df.dropna(how="all", axis='columns')  # remove empty columns
            df = df.dropna(how="all", axis='rows')  # remove empty rows
            df = df.dropna(how="any", axis="rows")  # remove rows with na values
            df = df.T.reset_index(drop=True).T  # reset columns indexes
            df = df.reset_index(drop=True)  # reset row indexes

            # add column names
            if len(df.columns) < 2:  # if there is only one column
                df.columns = ["subsidiaries"]

            else:
                df.columns = ["sub_name", "jurisdiction_country/state"]  # if there are two columns

            # optional: save to csv
            if save_as_csv is True:
                if filename == "":
                    return print("Please provide a filename in the parameter: filename='***.csv'")  # require "filename"

                else:
                    df.to_csv(filename)
                    with pd.option_context("display.max_rows", None, "display.max_columns", None):  # display all in console
                        return print(df, "\n\nLink to subsidiaries of {ticker}: {url}".format(url = ex21_url, ticker = ticker))

            if save_as_csv is False:
                with pd.option_context("display.max_rows", None, "display.max_columns", None):  # display all in console
                    return print(df, "\n\nLink to subsidiaries of {ticker}: {url}".format(url = ex21_url, ticker = ticker))

        except:
            # some Exhibit 21 is not in HTML table (e.g. Facebook)
            # need some other processing - out of scope of this project
            soup = parse_html(ex21_url)
            text = soup.get_text().strip().split("\n")  # get the text
            return print(text, "\n\nLink to subsidiaries of {ticker}: {url}".format(url = ex21_url, ticker = ticker))



# test the function:
huidan_main("ibm")  # can be any ticker, such as "aapl" (Apple), "sbux" (Starbucks), "amzn" (Amazon)...
                    # an incorrect ticker will also produce a warning
                    # the dataframe can also be saved by adding two parameters: save_to_csv=True and filename="***.csv")
