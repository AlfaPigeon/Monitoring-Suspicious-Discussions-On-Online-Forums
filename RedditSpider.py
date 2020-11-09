import scrapy

class RedditSpider(scrapy.Spider):
    name = 'redditspider'
    start_urls = [
        'https://www.reddit.com/r/leagueoflegends/comments/',
        'https://www.reddit.com/r/lgbt/comments/',
        'https://www.reddit.com/r/Conservative/comments/',
        'https://www.reddit.com/r/ToxicAMWF/comments/',
        'https://www.reddit.com/r/atheism/comments/',
        'https://www.reddit.com/r/unpopularopinion/comments/',
        'https://www.reddit.com/r/Feminism/comments/'
    ]
    COUNT_MAX = 1000
    count = 0

    def parse(self, response):

        for comment in response.css('.comment'):
            text = ""
            for p in comment.css('.usertext-body>.md>p::text').getall():
                text += " " + p
            pagename = comment.css('.subreddit::text').get()
            author = comment.css(".author::text").get()
            title = comment.css(".title::text").get()
            yield {
                "pagename" : pagename,
                "author" : author,
                "title" : title,
                "comment" : text
            }

        next_page = response.css('.next-button>a::attr("href")').get()
        if (next_page is not None):
            yield scrapy.Request(next_page, callback=self.parse)

