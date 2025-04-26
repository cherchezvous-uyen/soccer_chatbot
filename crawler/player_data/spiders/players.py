import scrapy
from w3lib.html import remove_tags
import re

class PlayerSpider(scrapy.Spider):
    name = "players"
    allowed_domains = ["transfermarkt.com"]
    start_urls = [
        "https://www.transfermarkt.com/uefa-champions-league/marktwerte/pokalwettbewerb/CL/pos//detailpos/0/altersklasse/alle/plus/1"
    ]

    def parse(self, response): 
        for row in response.css("table.items tbody tr"):
            player_link = row.css("td:nth-child(2) a::attr(href)").get()
            player_name = row.css("td:nth-child(2) a::attr(title)").get()

            if player_link:
                player_id_match = re.search(r'/spieler/(\d+)', player_link)
                if player_id_match:
                    player_id = player_id_match.group(1)

                    yield response.follow(
                        player_link,
                        callback=self.parse_player_profile,
                        meta={"player_name": player_name, "player_id": player_id}
                    )

        # Phân trang
        current_page_li = response.css("ul.tm-pagination li.tm-pagination__list-item--active")
        if current_page_li:
            next_page_li = current_page_li.xpath("following-sibling::li[1]/a/@href").get()
            if next_page_li:
                yield response.follow(next_page_li, callback=self.parse)

    def parse_player_profile(self, response):
        item = {}
        player_name = response.meta.get("player_name")
        player_id = response.meta.get("player_id")
    
        item["player_name"] = player_name

        name_in_home_country = response.xpath('//span[text()="Name in home country:"]/following-sibling::span[1]/text()').get()
        item["name_in_home_country"] = name_in_home_country.strip() if name_in_home_country else player_name

        item["shirt_number"] = response.css("span.data-header__shirt-number::text").get(default="").strip()
        item["birth_date"] = response.css("span[itemprop='birthDate']::text").get(default="").strip()

        # Citizenship
        citizenship = " / ".join(response.xpath('//span[text()="Citizenship:"]/following-sibling::span[1]//text()').getall()).strip().replace("\xa0", " ").replace("\n", "")
        item["citizenship"] = citizenship

        # Birthplace, fallback = citizenship
        birth_place = response.css("span[itemprop='birthPlace']::text").get()
        item["birth_place"] = birth_place.strip() if birth_place else citizenship

        item["height"] = response.css("span[itemprop='height']::text").get(default="").strip()
        item["position"] = response.xpath('//li[contains(@class, "data-header__label") and contains(text(), "Position")]/span[@class="data-header__content"]/text()').get(default="").strip()
        item["current_national_team"] = response.xpath('//li[contains(@class, "data-header__label") and contains(text(), "Current international:")]//a/text()').get(default="").strip()
        item["club"] = response.css(".data-header__club a::text").get(default="").strip()
        item["joined"] = response.xpath('//span[contains(@class, "data-header__label") and contains(text(), "Joined")]/span[@class="data-header__content"]/text()').get(default="").strip()
        item["contract_expires"] = response.xpath('//span[contains(@class, "data-header__label") and contains(text(), "Contract expires")]/span[@class="data-header__content"]/text()').get(default="").strip()
        item["market_value"] = response.css(".data-header__market-value-wrapper::text").get(default="").strip()

        # Awards
        item["awards"] = []
        for award in response.css("a.data-header__success-data"):
            title = award.attrib.get("title", "").strip()
            count = award.css("span.data-header__success-number::text").get(default="1").strip()
            if title:
                item["awards"].append({"title": title, "count": count})

        # Injury
        injury_div = response.css(".verletzungsbox .text")
        if injury_div:
            reason = injury_div.xpath("text()[1]").get(default="").strip()
            return_date = injury_div.css("span.rueckkehr::text").get(default="").strip()
            item["injury_status"] = reason if reason else "Không có thông tin về các vấn đề này"
            item["return_date"] = return_date if return_date else "Không có thông tin về các vấn đề này"
        else:
            item["injury_status"] = "Không có thông tin về các vấn đề này"
            item["return_date"] = "Không có thông tin về các vấn đề này"

        # Extra info
        extra_info_raw = response.css(".tm-player-additional-data .content").get()
        if extra_info_raw:
            clean_text = remove_tags(extra_info_raw)
            lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
            lines = [line for line in lines if not re.search(r"is the (brother|son|father) of", line.lower())]
            item["extra_info"] = lines if lines else ["Không có thông tin về các vấn đề này"]
        else:
            item["extra_info"] = ["Không có thông tin về các vấn đề này"]

        yield item
