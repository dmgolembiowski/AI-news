#!/usr/bine/env python3
"""
Scrapy Documentation:
    https://docs.scrapy.org/en/latest/intro/overview.htm
    Accessed March 13, 2019
"""
import scrapy

URLFILE = "/home/david/ai/AI-news/newscraping/trainingData/urls.txt"

class QuoteSpider(scrapy.Spider):
    start_urls = []
    with open(URLFILE,'r') as f:
        start_urls.append(f.read().split('\n'))
    start_urls = start_urls[0]

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'data-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)

