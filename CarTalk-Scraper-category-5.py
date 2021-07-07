'''
@file       CarTalkWebscraper.py
@date       2020/06/09
@brief      Class to scrape attributes of interest from all topics on the CarTalk Discourse forum
'''

import time
from datetime import datetime
import os

from bs4 import BeautifulSoup
from selenium import webdriver

import pandas as pd
import json


'''
@brief  Webscraper that scrapes attributes of interest from all topics on the CarTalk Discourse forum
'''


class CarTalkWebscraper:
    driver = None                   # Selenium webdriver object
    topicDict = {}                  # Dictionary of all topics and their attributes
    topicDataframe = \
        pd.DataFrame(columns=[      # Pandas dataframe of all topic attributes
            'Topic Title',
            'Category',
            'Tags',
            'Author',
            'Commenters',
            'Leading Comment',
            'Other Comments',
            'Likes',
            'Views'])

    def __init__(self, webdriverPath):
        # Set up webdriver
        options = webdriver.ChromeOptions()
        # Ignore security certificates
        options.add_argument('--ignore-certificate-errors')
        # Use Chrome in Incognito mode
        options.add_argument('--incognito')
        # Run in background
        options.add_argument('--headless')
        self.driver = webdriver.Chrome(
            executable_path=webdriverPath,
            options=options)
        options.add_argument("start-maximized")
        options.add_argument("enable-automation")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-browser-side-navigation")
        options.add_argument("--disable-gpu")

    '''
    @brief      Retrieves a topic title
    @param      topicSoup   BeautifulSoup object that contains the topic page HTML
    @return     topicName   Topic name
    '''

    def get_title(self, topicSoup):
        topicName = topicSoup.find('a', class_='fancy-title')
        if topicName == None:
            return str(0)
        else:
            topicName = topicName.text
        # Remove leading and trailing spaces and newlines
        topicName = topicName.replace('\n', '').strip()
        return topicName

    '''
    @brief      Retrieves a topic's category and tags
    @param      topicSoup   BeautifulSoup object that contains the topic page HTML
    @return     category    Category that the topic belongs to
    @return     tags        List of topic tags
    '''

    def get_category_and_tags(self, topicSoup):
        topicCategoryDiv = topicSoup.find(
            'div', class_='topic-category ember-view')
        categoryAnchors = topicCategoryDiv.find('span', class_='category-name')
        tagAnchors = topicCategoryDiv.find_all(
            'a', class_='discourse-tag bullet')

        category = categoryAnchors.text
        tagList = []
        for anchor in tagAnchors:
            tagList.append(anchor.text)

        # if (len(tagList) == 1):
        #     category = tagList[0]
        #     tags = []
        #     return category, tags
        # else:
        #     category = tagList[0]
        #     tags = tagList[1:]
        return category, tagList

    '''
    @brief      Retrieves a topic's author and commenters
    @param      topicSoup   BeautifulSoup object that contains the topic page HTML
    @return     author      Author username
    @return     commenters  List of unique commenter usernames
    '''

    def get_author_and_commenters(self, topicSoup):
        names = topicSoup.find_all("div", class_="names trigger-user-card")
        authorList = []
        for name in names:
            author = name.span.a.text
            authorList.append(author)

        # Remove redundant names
        authorList = list(set(authorList))

        if (len(authorList) == 1):
            author = authorList[0]
            commenters = []
            return author, commenters
        else:
            author = authorList[0]
            commenters = authorList[1:]
            return author, commenters

    '''
    @brief      Retrieves a topic's comments
    @param      topicSoup       BeautifulSoup object that contains the topic page HTML
    @return     leadingComment  Leading comment (by the author)
    @return     otherComments   List of other comments
    '''

    def get_comments(self, topicSoup):
        postStream = topicSoup.find('div', class_='post-stream')
        postDivs = postStream.find_all('div',
                                       {'class': ['topic-post clearfix regular', 'topic-post clearfix topic-owner regular']})

        comments = []
        for i in range(len(postDivs)):
            comment = postDivs[i].find('div', class_='cooked').text
            comments.append(comment)

        if (len(comments) == 1):
            leadingComment = comments[0]
            otherComments = []
            return leadingComment, otherComments
        else:
            leadingComment = comments[0]
            otherComments = comments[1:]
            return leadingComment, otherComments

    '''
    @brief      Retrieves a topic's number of views
    @param      topicSoup           BeautifulSoup object that contains the topic page HTML
    @return     views.span.text     Number of views as a string
    '''

    def get_views(self, topicSoup):
        views = topicSoup.find('li', class_='secondary views')
        if views == None:
            return str(0)
        return views.span.text

    '''
    @brief      Retrieves a topic's number of likes
    @param      topicSoup           BeautifulSoup object that contains the topic page HTML
    @return     likes.span.text     Number of likes as a string
    '''

    def get_likes(self, topicSoup):
        likes = topicSoup.find('li', class_='secondary likes')
        if likes == None:
            return str(0)
        return likes.span.text

    '''
    @brief      Runs the webscraper application and saves the data in both JSON and CSV files
    @param      baseURL     Link to the CarTalk forum home page
    @return     None
    '''

    def runApplication(self, baseURL):
        # Open Chrome web client using Selenium and retrieve page source
        self.driver.get('https://community.cartalk.com/categories')
        baseHTML = self.driver.page_source
        # Get base HTML text and generate soup object
        baseSoup = BeautifulSoup(baseHTML, 'html.parser')

        # Find all anchor objects that contain category information
        categoryAnchors = baseSoup.find_all('a', class_='category-title-link')

        # Get hyperlink references and append it to the base URL to get the category page URLs

        # categoryPageURLs = []
        # for i in range(len(categoryAnchors)):
        #     href = categoryAnchors[i]['href']
        #     categoryPageURLs.append(baseURL + href)

        # 1st for loop to loop through all categories
        # for categoryURL in categoryPageURLs[2:3]:
        # Access category webpage
        self.driver.get('https://community.cartalk.com/c/buying-selling/13') #https://community.cartalk.com/c/general-discussion/8
        count = 0
        # Load the entire webage by scrolling to the bottom
        lastHeight = self.driver.execute_script(
            "return document.body.scrollHeight")
        while (True):
            # Scroll to bottom of page
            print("here")
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            # Wait for new page segment to load
            time.sleep(1)

            # Calculate new scroll height and compare with last scroll height
            newHeight = self.driver.execute_script(
                "return document.body.scrollHeight")
            if newHeight == lastHeight or count > 100:
                break
            lastHeight = newHeight
            count += 1

        # Generate category soup object
        categoryHTML = self.driver.page_source
        categorySoup = BeautifulSoup(categoryHTML, 'html.parser')

        # Find all anchor objects that contain topic information
        topicAnchors = categorySoup.find_all(
            'a', class_='title raw-link raw-topic-link')

        # Get hyperlink references and append it to the base URL to get the topic page URLs
        topicPageURLs = []
        # print("Finding topic for No.", categoryURL, "category")
        print(len(topicAnchors))
        for i in range(len(topicAnchors)):
            href = topicAnchors[i]['href']
            topicPageURLs.append(baseURL + href)
        print(topicPageURLs)
        # 2nd for loop to loop through all topics in a category
        for topicURL in topicPageURLs:
            print(topicURL)
            # Get topic HTML text and generate topic soup object
            self.driver.get(topicURL)
            topicHTML = self.driver.page_source
            topicSoup = BeautifulSoup(topicHTML, 'html.parser')

            # Scape all topic attributes of interest
            topicTitle = self.get_title(topicSoup)
            category, tags = self.get_category_and_tags(topicSoup)
            author, commenters = self.get_author_and_commenters(topicSoup)
            leadingComment, otherComments = self.get_comments(topicSoup)
            numLikes = self.get_likes(topicSoup)
            numViews = self.get_views(topicSoup)

            # Create attribute dictionary for topic
            attributeDict = {
                'Topic Title':   topicTitle,
                'Category':   category,
                'Tags':   tags,
                'Author':   author,
                'Commenters':   commenters,
                'Leading Comment':   leadingComment,
                'Other Comments':   otherComments,
                'Likes':   numLikes,
                'Views':   numViews}

            # Add the new entry to the topic dictionary and Pandas dataframe
            self.topicDict[topicTitle] = attributeDict
            self.topicDataframe = self.topicDataframe.append(
                attributeDict, ignore_index=True)
            
            if count > 200:
                break
            count+=1

            '''
            print('Topic Title:')
            print(topicTitle)
            print('Category:')
            print(category)
            print('Tags:')
            print(tags)
            print('Author:')
            print(author)
            print('Commenters:')
            print(commenters)
            print('Leading Comment:')
            print(leadingComment)
            
            print('Other Comments:')
            print(otherComments)
            print('Likes:')
            print(numLikes)
            print('Views:')
            print(numViews)
            '''

        # Get unique timestamp of the webscraping
        timeStamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # Save data in JSON and CSV files and store in the save folder as this program
        jsonFilename = 'CarTalk_Topic_Attributes_5' + timeStamp + '.json'
        csvFilename = 'CarTalk_Topic_Attributes_5' + timeStamp + '.csv'

        jsonFileFullPath = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), jsonFilename)
        csvFileFullPath = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), csvFilename)

        with open(jsonFileFullPath, 'w') as f:
            json.dump(self.topicDict, f)

        self.topicDataframe.to_csv(csvFileFullPath)


if __name__ == '__main__':
    # Local path to webdriver

    webdriverPath = '/usr/local/bin/chromedriver'

    # CarTalk forum base URL
    baseURL = 'https://community.cartalk.com'

    # Create CarTalk webscraping object
    CarTalkWebscraper = CarTalkWebscraper(webdriverPath)

    # Run webscraping and save data
    CarTalkWebscraper.runApplication(baseURL)