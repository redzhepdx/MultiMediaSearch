from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import textdistance as td
from dateutil.relativedelta import relativedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MetadataRuleEngine:
    """

    :param annotations:
    """

    def __init__(self, annotations: List[Dict[str, Any]]) -> None:
        self.annotations = annotations
        self.game_data_length = len(self.annotations)
        self.time_window_size = 1

        self.genre_similarity_fn = td.Tversky(ks=(0.8, 0.2))
        self.platform_similarity_fn = td.Sorensen()
        self.game_mode_similarity_fn = td.Tversky(ks=(0.8, 0.2))
        self.game_themes_similarity_fn = td.Tversky(ks=(0.8, 0.2))

        self.platform_clusters: Optional[Dict[int, List[int]]] = None
        self.genre_clusters: Optional[Dict[int, List[int]]] = None
        self.game_mode_clusters: Optional[Dict[int, List[int]]] = None
        self.game_themes_clusters: Optional[Dict[int, List[int]]] = None

        self.company_vectorizer = TfidfVectorizer()

    def calc_similarity(self, game_x: Dict[str, Any], game_y: Dict[str, Any]) -> Tuple[float, ...]:
        """
        :param game_x:
        :param game_y:
        :return:
        """
        game_mode_sim_score = -1.0
        game_themes_sim_score = -1.0
        genre_sim_score = self.genre_similarity_fn(game_x["genres"], game_y["genres"])
        platform_sim_score = self.platform_similarity_fn(game_x["platforms"], game_y["platforms"])
        platform_sim_score = float(platform_sim_score > 0.0)

        if game_x.get("modes") and game_y.get("modes"):
            game_mode_sim_score = self.game_mode_similarity_fn(game_x["modes"], game_y["modes"])
        if game_x.get("themes") and game_y.get("themes"):
            game_themes_sim_score = self.game_mode_similarity_fn(game_x["themes"], game_y["themes"])

        return genre_sim_score, platform_sim_score, game_mode_sim_score, game_themes_sim_score

    def temporal_proximity_date_similarity(self, game_x: Dict[str, Any], game_y: Dict[str, Any]) -> float:
        """
        :param game_x:
        :param game_y:
        :return:
        """
        if game_x.get("last_release_date") is None or game_y.get("last_release_date") is None:
            return 0.0

        # unix to datetime
        game_x_release_date = datetime.fromtimestamp(game_x["last_release_date"]).strftime("%Y-%m-%d")
        game_x_release_date = datetime.strptime(game_x_release_date, "%Y-%m-%d")
        game_y_release_date = datetime.fromtimestamp(game_y["last_release_date"]).strftime("%Y-%m-%d")
        game_y_release_date = datetime.strptime(game_y_release_date, "%Y-%m-%d")

        delta = relativedelta(game_x_release_date, game_y_release_date)

        return 1.0 if abs(delta.years) <= self.time_window_size else 0.0

    def structure_by_platform(self) -> None:
        """

        """
        self.platform_clusters = {}
        for annotation in self.annotations:
            for platform in annotation["platforms"]:
                if platform not in self.platform_clusters:
                    self.platform_clusters[platform] = []
                self.platform_clusters[platform].append(annotation["product_id"])

    def structure_by_genres(self):
        """

        """
        self.genre_clusters = {}
        for annotation in self.annotations:
            for genre in annotation["genres"]:
                if genre not in self.genre_clusters:
                    self.genre_clusters[genre] = []
                self.genre_clusters[genre].append(annotation["product_id"])

    def get_game_ids_by_platform(self, platforms: Union[int, List[int]]) -> List[int]:
        """

        :param platforms:
        :return:
        """
        if isinstance(platforms, int):
            return self.platform_clusters[platforms]
        game_ids = set()
        for platform in platforms:
            game_ids.update(self.platform_clusters[platform])
        return list(game_ids)

    def get_game_ids_by_genre(self, genres: Union[int, List[int]]) -> List[int]:
        """

        :param genres:
        :return:
        """
        if isinstance(genres, int):
            return self.genre_clusters[genres]
        game_ids = set()
        for genre in genres:
            game_ids.update(self.genre_clusters[genre])
        return list(game_ids)

    def prepare_company_vectorizer(self, company_descriptions: List[str]) -> None:
        """
        :param company_descriptions:
        :return:
        """
        self.company_vectorizer.fit(company_descriptions)

    def company_similarity(self, game_x: Dict[str, Any], game_y: Dict[str, Any]) -> float:
        """
        :param game_x:
        :param game_y:
        :return:
        """
        game_x_company_desc = game_x["company_description"]
        game_y_company_desc = game_y["company_description"]

        if game_x_company_desc is None or game_y_company_desc is None:
            return 0.0

        game_x_company = self.company_vectorizer.transform([game_y_company_desc])
        game_y_company = self.company_vectorizer.transform([game_y_company_desc])

        similarity = cosine_similarity(game_x_company, game_y_company)
        return similarity[0][0]
