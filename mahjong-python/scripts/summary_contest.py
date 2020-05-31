# -*- coding: utf-8 -*-

import json
import sys
from collections import Counter

import requests
from bs4 import BeautifulSoup


def get_table(contest_id):
    response = requests.get('https://botzone.org.cn/contest/detail/' + contest_id)
    contest = json.loads(response.content)
    return BeautifulSoup('<table>' + contest['table'] + '</table>', 'lxml')


def get_block_info(block):
    score = int(block.select_one('div.score').text)
    links = block.select('a')
    user_name = links[0].text.strip()
    bot_name = links[1].text.strip().replace(' ', ':')
    return f'{user_name}/{bot_name}', score


def summary_contest(contest_id):
    match_index = 0
    table = get_table(contest_id)
    records = []

    matches = Counter()
    for row in table.select('tr'):
        blocks = row.select('div.matchresult')
        match_url = row.select_one('td > a').attrs['href']

        assert len(blocks) == 4
        records.extend([[(match_index, match_url), get_block_info(block)] for block in blocks])
        match_index += 1

        matches[tuple(sorted(r[1][0] for r in records[-4:]))] += 1

    print('number of matches:', len(records) // 4)
    scores = {}
    for _, (identity, score) in records:
        info = scores.get(identity)
        if info is None:
            info = scores[identity] = [0, 0]
        info[0] += score
        info[1] += 1

    for rank, (identity, info) in enumerate(sorted(scores.items(), key=lambda x: -x[1][0]), 1):
        print(rank, info, identity, sep='\t')

    print('outstanding matches:')
    for (match_index, match_url), (identity, score) in records:
        if score > 150:
            print(match_index, match_url, score, identity, sep='\t')


if __name__ == '__main__':
    summary_contest(sys.argv[1])
