# -*- coding: utf-8 -*-

# Copyright 2019 Information Retrieval Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


class RatingsMatrix(object):
    """
    Represents a ratings matrix. Base class that doesn't actually load
    any data.

    The implemented methods suppose the subclasses store ratings as a pair
    of dicts, one mapping users to a list of pairs (item, rating) and another
    mapping items to a list of pairs (user, rating).

    If any subclass want to store the data in any other way it should override
    these methods
    """

    users = {}
    items = {}

    def users_set(self):
        """
        Returns a set with all the users that have at least one rating in the
        matrix
        """
        return set(self.users.keys())

    def items_set(self):
        """
        Returns a set with all the items that have been rated at least once
        """
        return set(self.items.keys())

    def rating(self, user, item):
        """
        Returns the rating given by an user to an item.

        Returns None if the user hasn't rated the item

        - user: user to get rating from
        - item: item which rating we want to find out
        """
        for rated_item, rating in self.users[user]:
            if rated_item == item:
                return rating
        return None

    def items_rated_by_user(self, user):
        """
        Returns a set of the items that the user has rated

        - user: get items rated by this user
        """
        return set(item_id for item_id, _ in self.users[user])

    def users_with_rating_for_item(self, item):
        return set(user_id for user_id, _ in self.items[item])

    def user_ratings(self, user):
        """
        Returns all the ratings of the user in a dict mapping item -> rating
        """
        return dict(self.users[user])

    def item_ratings(self, item):
        """
        Returns all the ratings for the item in a dict mapping user -> rating
        """
        return dict(self.items[item])


class TabDataSet(RatingsMatrix):
    """
    Rating matrix loaded from a dataset file with commas or whitespaces
    (e.g. tabs) as separators, with fields: user, item, rating, any extra
    fields.
    """

    def __init__(self, load_from):
        """
        - load_from: movielens dataset to load data from
        """
        with open(load_from) as f_in:
            users = {}
            items = {}
            for line in f_in:
                tokens = re.split(r'[\s,]', line)
                user = int(tokens[0])
                item = int(tokens[1])
                rating = float(tokens[2])
                users.setdefault(user, {})[item] = rating
                items.setdefault(item, {})[user] = rating

        self.users = users
        self.items = items

    def rating(self, user, item):
        if item in self.users[user]:
            return self.users[user][item]
        else:
            return 0

    def items_rated_by_user(self, user):
        return set(self.users[user].keys())

    def users_with_rating_for_item(self, item):
        return set(self.items[item].keys())

    def user_ratings(self, user):
        return self.users[user]

    def item_ratings(self, item):
        return self.items[item]


class QRel(RatingsMatrix):
    """
    Rating matrix loaded from a qrel file.

    Fields: <user_id>\t0\t<item_id>\t<relevance>
    """

    def __init__(self, load_from):
        """
        - load_from: qrel file to load data from
        """
        with open(load_from) as f_in:
            users = {}
            items = {}
            for line in f_in:
                tokens = re.split(r'[\s,]', line)
                user = int(tokens[0])
                item = int(tokens[2])
                rating = float(tokens[3])
                users.setdefault(user, []).append((item, rating))
                items.setdefault(item, []).append((user, rating))

        self.users = users
        self.items = items
